import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
  HarnessError,
  INVALID_CREDENTIAL_CODE,
  QUOTA_EXCEEDED_CODE,
} from '@deepseek-ai/dsh-llm';
import { createSanitizedAgyEnv } from './binary.js';
import { MCP_SERVER_NAME, type McpSessionEndpoint } from './mcp-bridge.js';
import type {
  AcpAvailableModel,
  AcpPromptPart,
  AcpPromptResponse,
  AcpSubscriptionTier,
} from './types.js';

const MAX_NDJSON_LINE_BYTES = 32 * 1024 * 1024; // 32 MiB (avoids 256 KiB model-catalog overflow)
const MAX_STDERR_TAIL_CHARS = 4096;

/**
 * Matches in-band upstream API failure banners emitted by `agy`'s internal `run.go` retry loop,
 * e.g. `⚠ API error (attempt 1): RESOURCE_EXHAUSTED (code 429): Resource has been exhausted...`
 */
const IN_BAND_API_ERROR_RE =
  /^\s*⚠\s*(?:API error \(attempt (\d+)\):\s*)?((?:RESOURCE_EXHAUSTED|PERMISSION_DENIED|TOS_VIOLATION|UNAUTHENTICATED|INVALID_ARGUMENT)\s*\(code \d+\):[^\n]*)/i;

export function classifyInBandAgyApiError(chunkText: string): HarnessError | undefined {
  const match = IN_BAND_API_ERROR_RE.exec(chunkText);
  if (!match) return undefined;
  const detail = match[2]!.trim();
  const upper = detail.toUpperCase();
  if (upper.includes('UNAUTHENTICATED')) {
    return new HarnessError(
      `Antigravity upstream authentication error: ${detail} (run 'agy login' to refresh credentials)`,
      INVALID_CREDENTIAL_CODE,
    );
  }
  if (
    upper.includes('RESOURCE_EXHAUSTED') ||
    upper.includes('429') ||
    upper.includes('TOS_VIOLATION') ||
    upper.includes('PERMISSION_DENIED') ||
    upper.includes('403')
  ) {
    return new HarnessError(
      `Antigravity upstream quota/policy rejection (circuit breaker tripped on attempt 1 to prevent retry storm): ${detail}`,
      QUOTA_EXCEEDED_CODE,
    );
  }
  return new HarnessError(`Antigravity upstream API error: ${detail}`, 'INVALID_ARGS');
}

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
}

function createDeferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

export interface AcpAgentCapabilitiesSummary {
  protocolVersion: number;
  image: boolean;
  loadSession: boolean;
  systemPromptAppend: boolean;
}

export interface AcpNewSessionResult {
  sessionId: string;
  currentModelId?: string;
  availableModels: AcpAvailableModel[];
  tier?: AcpSubscriptionTier;
}

export interface AcpStreamCallbacks {
  onThoughtDelta?: (text: string) => void;
  onTextDelta?: (text: string) => void;
  onUsageUpdate?: (used: number, size: number) => void;
}

interface JsonRpcWireMessage {
  jsonrpc?: string;
  id?: number | string;
  method?: string;
  params?: Record<string, unknown>;
  result?: unknown;
  error?: { code?: number; message?: string; data?: unknown };
}

/**
 * Persistent JSON-RPC 2.0 client over one official `agy acp --stdio` child process.
 */
export class AcpWorkerClient {
  readonly #agyPath: string;
  readonly #cwd: string;
  readonly #startupTimeoutMs: number;
  readonly #extraArgs: readonly string[];

  #proc?: ChildProcessWithoutNullStreams;
  #tempCwd?: string;
  #rpcId = 0;
  readonly #pendingRpcs = new Map<number | string, Deferred<unknown>>();
  #stderrTail = '';
  #closed = false;
  #closeError?: Error;
  #capabilities?: AcpAgentCapabilitiesSummary;

  readonly #sessions = new Map<string, {
    endpoint?: McpSessionEndpoint;
    callbacks?: AcpStreamCallbacks;
    promptId?: number;
  }>();
  #starting?: Promise<AcpAgentCapabilitiesSummary>;

  constructor(opts: {
    agyPath: string;
    cwd?: string;
    startupTimeoutMs: number;
    extraArgs?: readonly string[];
  }) {
    this.#agyPath = opts.agyPath;
    if (opts.cwd) {
      this.#cwd = opts.cwd;
    } else {
      this.#tempCwd = fs.mkdtempSync(path.join(os.tmpdir(), 'dsh-agy-native-'));
      this.#cwd = this.#tempCwd;
    }
    this.#startupTimeoutMs = opts.startupTimeoutMs;
    this.#extraArgs = opts.extraArgs ?? [];
  }

  get alive(): boolean {
    return !this.#closed && this.#proc !== undefined && this.#proc.exitCode === null;
  }

  get capabilities(): AcpAgentCapabilitiesSummary | undefined {
    return this.#capabilities;
  }

  get cwd(): string {
    return this.#cwd;
  }

  async start(): Promise<AcpAgentCapabilitiesSummary> {
    if (this.#capabilities && this.alive) return this.#capabilities;
    if (this.#closed) {
      throw this.#closeError ?? new Error('dsh-agy-native: worker client already closed');
    }

    return this.#starting ??= this.#initialize();
  }

  async #initialize(): Promise<AcpAgentCapabilitiesSummary> {
    const args = ['acp', '--stdio', ...this.#extraArgs];
    const proc = spawn(this.#agyPath, args, {
      cwd: this.#cwd,
      env: createSanitizedAgyEnv(),
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    this.#proc = proc;

    proc.stderr.setEncoding('utf8');
    proc.stderr.on('data', (chunk: string) => {
      this.#stderrTail = (this.#stderrTail + chunk).slice(-MAX_STDERR_TAIL_CHARS);
    });

    proc.on('error', (err) => {
      this.#handleFatalClose(new Error(`Failed to spawn agy (${this.#agyPath}): ${err.message}`));
    });

    proc.on('exit', (code, signal) => {
      const detail = this.#stderrTail.trim();
      const suffix = detail ? ` — stderr: ${detail.slice(-400)}` : '';
      this.#handleFatalClose(
        new Error(
          `Antigravity CLI exited unexpectedly (code=${String(code)}, signal=${String(signal)})${suffix}`,
        ),
      );
    });

    proc.stdin.on('error', (err) => {
      this.#handleFatalClose(new Error(`Antigravity stdin error: ${err.message}`));
    });

    this.#attachStdoutReader(proc);

    const initResult = (await this.#rpc(
      'initialize',
      {
        protocolVersion: 1,
        clientInfo: { name: 'dsh-agy-native', version: '0.2.0' },
        clientCapabilities: {},
      },
      this.#startupTimeoutMs,
    )) as {
      protocolVersion?: number;
      agentCapabilities?: Record<string, unknown>;
    };

    const caps = initResult?.agentCapabilities ?? {};
    const promptCaps = (caps.promptCapabilities ?? {}) as Record<string, unknown>;
    const sessionCaps = (caps.sessionCapabilities ?? caps.session_capabilities ?? {}) as Record<
      string,
      unknown
    >;
    const metaAgy = (((caps._meta as Record<string, unknown> | undefined)?.antigravity ??
      {}) as Record<string, unknown>);
    const sysPromptCap = (metaAgy.systemPrompt ?? {}) as Record<string, unknown>;

    this.#capabilities = {
      protocolVersion: initResult?.protocolVersion ?? 1,
      image: Boolean(promptCaps.image),
      loadSession: Boolean(caps.loadSession || caps.load_session || sessionCaps.load),
      systemPromptAppend: Boolean(sysPromptCap.append),
    };
    return this.#capabilities;
  }

  /**
   * Create a new ACP session while preserving Antigravity's official `systemInstruction`
   * 100% untouched (never sets `excludeDefaultComponents: true` or `systemPrompt.append`,
   * avoiding Cloud Code Assist `systemInstruction` WAF false-positive 429s).
   */
  async createSession(opts: {
    endpoint?: McpSessionEndpoint;
    cwd?: string;
  }): Promise<AcpNewSessionResult> {
    await this.start();

    const hasTools = Boolean(opts.endpoint?.hasTools);
    const mcpServers = hasTools && opts.endpoint ? [opts.endpoint.toAcpMcpServerConfig()] : [];

    const antigravityMeta: Record<string, unknown> = {
      enabledTools: [],
      eagerMcpServers: hasTools ? [MCP_SERVER_NAME] : [],
    };

    const res = (await this.#rpc(
      'session/new',
      {
        cwd: opts.cwd ?? this.#cwd,
        mcpServers,
        _meta: { antigravity: antigravityMeta },
      },
      this.#startupTimeoutMs,
    )) as {
      sessionId: string;
      models?: {
        currentModelId?: string;
        availableModels?: AcpAvailableModel[];
      };
      _meta?: {
        antigravity?: {
          tier?: AcpSubscriptionTier;
        };
      };
    };

    this.#sessions.set(res.sessionId, { endpoint: opts.endpoint });
    return {
      sessionId: res.sessionId,
      currentModelId: res.models?.currentModelId,
      availableModels: res.models?.availableModels ?? [],
      tier: res._meta?.antigravity?.tier,
    };
  }

  /**
   * Resume a persisted ACP session (`session/load`) while suppressing historical update replay
   * and keeping Antigravity's `systemInstruction` 100% untouched.
   */
  async loadSession(opts: {
    sessionId: string;
    endpoint?: McpSessionEndpoint;
    cwd?: string;
  }): Promise<void> {
    await this.start();
    const hasTools = Boolean(opts.endpoint?.hasTools);
    const mcpServers = hasTools && opts.endpoint ? [opts.endpoint.toAcpMcpServerConfig()] : [];

    const antigravityMeta: Record<string, unknown> = {
      enabledTools: [],
      eagerMcpServers: hasTools ? [MCP_SERVER_NAME] : [],
    };

    if (this.#sessions.has(opts.sessionId)) throw new Error('ACP session is already attached');
    this.#sessions.set(opts.sessionId, { endpoint: opts.endpoint });
    try {
      await this.#rpc(
        'session/load',
        {
          sessionId: opts.sessionId,
          cwd: opts.cwd ?? this.#cwd,
          mcpServers,
          _meta: { antigravity: antigravityMeta },
        },
        this.#startupTimeoutMs,
      );
    } catch (error) {
      this.#sessions.delete(opts.sessionId);
      throw error;
    }
  }

  async setModel(sessionId: string, rawModelId: string): Promise<void> {
    await this.#rpc('session/set_model', { sessionId, modelId: rawModelId }, this.#startupTimeoutMs);
  }

  setStreamCallbacks(sessionId: string, callbacks?: AcpStreamCallbacks): void {
    const session = this.#sessions.get(sessionId);
    if (session) session.callbacks = callbacks;
  }

  /**
   * Start `session/prompt` and return its completion promise. Note that when `agy` invokes
   * an MCP tool on this session’s endpoint, this promise remains pending until all MCP tool calls
   * in the turn have resolved and `agy` finishes its final response (or is cancelled).
   */
  prompt(
    sessionId: string,
    parts: readonly AcpPromptPart[],
    timeoutMs?: number,
  ): Promise<AcpPromptResponse> {
    const nextRpcId = this.#rpcId + 1;
    const session = this.#sessions.get(sessionId);
    if (!session) return Promise.reject(new Error('ACP session is not attached'));
    session.promptId = nextRpcId;
    return (
      this.#rpc(
        'session/prompt',
        {
          sessionId,
          prompt: parts,
        },
        timeoutMs,
      ) as Promise<AcpPromptResponse>
    ).finally(() => {
      if (session.promptId === nextRpcId) {
        session.promptId = undefined;
      }
    });
  }

  /** Request graceful cancellation of an in-flight `session/prompt` turn. */
  cancelPrompt(sessionId: string): void {
    if (!this.alive) return;
    try {
      this.#writeJsonRpc({
        method: 'session/cancel',
        params: { sessionId },
      });
    } catch {
      // Ignore write failure on dying stream
    }
  }

  async closeSession(sessionId: string): Promise<void> {
    this.#sessions.delete(sessionId);
    if (!this.alive) return;
    try {
      await this.#rpc('session/close', { sessionId }, 3000);
    } catch {
      // Best-effort session cleanup
    }
  }

  async dispose(): Promise<void> {
    this.#handleFatalClose(new Error('dsh-agy-native: worker disposed'));
    const proc = this.#proc;
    this.#proc = undefined;
    if (proc && proc.exitCode === null) {
      proc.kill('SIGTERM');
      await Promise.race([
        new Promise<void>((resolve) => proc.once('exit', () => resolve())),
        new Promise<void>((resolve) =>
          setTimeout(() => {
            if (proc.exitCode === null) proc.kill('SIGKILL');
            resolve();
          }, 1500),
        ),
      ]);
    }
    if (this.#tempCwd) {
      try {
        fs.rmSync(this.#tempCwd, { recursive: true, force: true });
      } catch {
        // Ignore cleanup failure
      }
      this.#tempCwd = undefined;
    }
  }

  #attachStdoutReader(proc: ChildProcessWithoutNullStreams): void {
    let buffer: Buffer = Buffer.alloc(0);
    proc.stdout.on('data', (chunk: Buffer) => {
      buffer = buffer.length === 0 ? Buffer.from(chunk) : Buffer.concat([buffer, chunk]);
      while (true) {
        const newlineIdx = buffer.indexOf(0x0a); // '\n'
        if (newlineIdx < 0) {
          if (buffer.byteLength > MAX_NDJSON_LINE_BYTES) {
            this.#handleFatalClose(
              new Error(`ACP frame exceeded ${MAX_NDJSON_LINE_BYTES} bytes limit`),
            );
            buffer = Buffer.alloc(0);
          }
          break;
        }
        const lineBuf = buffer.subarray(0, newlineIdx);
        buffer = buffer.subarray(newlineIdx + 1);
        const line = lineBuf.toString('utf8').trim();
        if (!line) continue;
        try {
          const parsed = JSON.parse(line) as JsonRpcWireMessage;
          this.#handleIncomingMessage(parsed);
        } catch (err) {
          this.#handleFatalClose(new Error(`Malformed JSON-RPC frame from agy: ${String(err)}`));
          break;
        }
      }
    });
  }

  #handleIncomingMessage(msg: JsonRpcWireMessage): void {
    const sessionId = msg.params?.sessionId;
    const session = typeof sessionId === 'string' ? this.#sessions.get(sessionId) : undefined;
    if (msg.method === 'session/request_permission') {
      const params = (msg.params ?? {}) as {
        toolCall?: { name?: string; rawInput?: Record<string, unknown> };
        options?: Array<{ kind?: string; optionId?: string }>;
      };
      const allowed = Boolean(
        session?.endpoint?.isAllowedPermissionRequest(
          params.toolCall?.name,
          params.toolCall?.rawInput,
        ),
      );
      const desiredKind = allowed ? 'allow_once' : 'reject_once';
      const selected = params.options?.find((opt) => opt.kind === desiredKind);
      if (msg.id !== undefined) {
        this.#writeJsonRpc({
          id: msg.id,
          result: {
            outcome: selected?.optionId
              ? { outcome: 'selected', optionId: selected.optionId }
              : { outcome: 'cancelled' },
          },
        });
      }
      return;
    }

    if (msg.method === 'session/update') {
      if (!session || typeof sessionId !== 'string' || session.promptId === undefined) return;
      const update = (msg.params?.update ?? {}) as {
        sessionUpdate?: string;
        content?: { type?: string; text?: string };
        used?: number;
        size?: number;
      };
      const kind = update.sessionUpdate;
      if (kind === 'agent_thought_chunk' && update.content?.type === 'text' && update.content.text) {
        session.callbacks?.onThoughtDelta?.(update.content.text);
        return;
      }
      if (kind === 'agent_message_chunk' && update.content?.type === 'text' && update.content.text) {
        const upstreamError = classifyInBandAgyApiError(update.content.text);
        if (upstreamError) {
          // Trip circuit breaker immediately on attempt 1 so `agy` never hammers Google 7 more times
          this.cancelPrompt(sessionId);
          const pending = this.#pendingRpcs.get(session.promptId);
          this.#pendingRpcs.delete(session.promptId);
          pending?.reject(upstreamError);
          return;
        }
        session.callbacks?.onTextDelta?.(update.content.text);
        return;
      }
      if (
        kind === 'usage_update' &&
        typeof update.used === 'number' &&
        typeof update.size === 'number'
      ) {
        session.callbacks?.onUsageUpdate?.(update.used, update.size);
      }
      return;
    }

    if (msg.method) {
      if (msg.id !== undefined) {
        this.#writeJsonRpc({
          id: msg.id,
          error: { code: -32601, message: `Unsupported client method: ${msg.method}` },
        });
      }
      return;
    }

    if (msg.id !== undefined) {
      const pending = this.#pendingRpcs.get(msg.id);
      if (!pending) return;
      this.#pendingRpcs.delete(msg.id);
      if (msg.error) {
        const errMsg = msg.error.message || JSON.stringify(msg.error);
        pending.reject(new Error(`ACP RPC error (${String(msg.error.code ?? 'unknown')}): ${errMsg}`));
      } else {
        pending.resolve(msg.result);
      }
    }
  }

  #rpc(method: string, params: Record<string, unknown>, timeoutMs?: number): Promise<unknown> {
    if (!this.alive || !this.#proc) {
      return Promise.reject(this.#closeError ?? new Error('dsh-agy-native: worker is not running'));
    }
    const id = ++this.#rpcId;
    const deferred = createDeferred<unknown>();
    this.#pendingRpcs.set(id, deferred);

    let timer: NodeJS.Timeout | undefined;
    if (timeoutMs && timeoutMs > 0) {
      timer = setTimeout(() => {
        if (this.#pendingRpcs.delete(id)) {
          deferred.reject(new Error(`ACP RPC '${method}' timed out after ${timeoutMs}ms`));
        }
      }, timeoutMs);
    }

    try {
      this.#writeJsonRpc({ id, method, params });
    } catch (err) {
      if (timer) clearTimeout(timer);
      this.#pendingRpcs.delete(id);
      return Promise.reject(err);
    }

    return deferred.promise.finally(() => {
      if (timer) clearTimeout(timer);
    });
  }

  #writeJsonRpc(payload: Record<string, unknown>): void {
    if (!this.#proc || this.#closed) {
      throw this.#closeError ?? new Error('dsh-agy-native: cannot write to closed worker');
    }
    this.#proc.stdin.write(JSON.stringify({ jsonrpc: '2.0', ...payload }) + '\n');
  }

  #handleFatalClose(error: Error): void {
    if (this.#closed) return;
    this.#closed = true;
    this.#closeError = error;
    for (const session of this.#sessions.values()) session.endpoint?.dispose();
    this.#sessions.clear();
    for (const pending of this.#pendingRpcs.values()) {
      pending.reject(error);
    }
    this.#pendingRpcs.clear();
  }
}
