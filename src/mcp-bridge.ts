import crypto from 'node:crypto';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import type { ToolSchema } from '@deepseek-ai/dsh-llm';
import type { PendingMcpToolCall } from './types.js';

export const MCP_SERVER_NAME = 'dsh';
const TOKEN_HEADER = 'x-dsh-mcp-token';

/** Truncate UTF-8 text to at most `maxBytes` without splitting multi-byte code points. */
export function truncateToolOutputUtf8(text: string, maxBytes: number): string {
  if (maxBytes <= 0) return text;
  const buf = Buffer.from(text, 'utf8');
  if (buf.byteLength <= maxBytes) return text;

  const footer = `\n\n[Output truncated at ${maxBytes} bytes by dsh-agy-native; refine your query or read with offset/limit.]`;
  const footerBytes = Buffer.byteLength(footer, 'utf8');
  const target = Math.max(64, maxBytes - footerBytes);
  let end = Math.min(buf.byteLength, target);
  while (end > 0 && (buf[end]! & 0b1100_0000) === 0b1000_0000) {
    end--;
  }
  return buf.subarray(0, end).toString('utf8') + footer;
}

/** Encode arbitrary DSH tool names into safe MCP tool identifiers (`[a-zA-Z0-9_-]`). */
export function encodeMcpToolName(name: string, usedNames: Set<string>): string {
  const clean = /^[a-zA-Z0-9_-]{1,64}$/.test(name)
    ? name
    : name.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 54);
  const base = clean || 'tool';
  if (!usedNames.has(base)) {
    usedNames.add(base);
    return base;
  }
  const suffix = crypto.createHash('sha1').update(name).digest('hex').slice(0, 6);
  const stem = `${base.slice(0, 56)}_${suffix}`;
  let candidate = stem;
  for (let n = 2; usedNames.has(candidate); n++) {
    const tail = `_${n}`;
    candidate = stem.slice(0, 64 - tail.length) + tail;
  }
  usedNames.add(candidate);
  return candidate;
}

/** Compute a deterministic fingerprint over active tool schemas so we detect schema changes. */
export function computeToolsFingerprint(tools: readonly ToolSchema[] | undefined): string {
  if (!tools || tools.length === 0) return 'none';
  const hash = crypto.createHash('sha256');
  for (const tool of tools) {
    hash.update(tool.name);
    hash.update('\0');
    hash.update(tool.description ?? '');
    hash.update('\0');
    hash.update(JSON.stringify(tool.parameters ?? {}));
    hash.update('\n');
  }
  return hash.digest('hex').slice(0, 24);
}

/** Strip `agy`-injected `toolSummary` unless the DSH tool explicitly declared it. */
export function sanitizeMcpToolArguments(
  rawArgs: unknown,
  schema: ToolSchema | undefined,
): Record<string, unknown> {
  if (!rawArgs || typeof rawArgs !== 'object' || Array.isArray(rawArgs)) {
    return {};
  }
  const copy: Record<string, unknown> = { ...(rawArgs as Record<string, unknown>) };
  if ('toolSummary' in copy) {
    const props = schema?.parameters?.properties;
    const declaresToolSummary =
      props !== null &&
      typeof props === 'object' &&
      !Array.isArray(props) &&
      Object.prototype.hasOwnProperty.call(props, 'toolSummary');
    if (!declaresToolSummary) {
      delete copy.toolSummary;
    }
  }
  return copy;
}

interface JsonRpcRequest {
  jsonrpc?: string;
  id?: string | number | null;
  method?: string;
  params?: Record<string, unknown>;
}

/** Per-ACP-session tool registration handle on the shared loopback MCP server. */
export class McpSessionEndpoint {
  readonly token: string;
  readonly path = `/mcp/${crypto.randomUUID()}`;
  readonly #url: string;
  readonly #maxToolOutputBytes: number;
  readonly #batchWindowMs: number;
  readonly #onDispose: () => void;

  #tools: readonly ToolSchema[] = [];
  #mcpToDsh = new Map<string, ToolSchema>();
  #callSeq = 0;

  readonly #pendingById = new Map<string, PendingMcpToolCall>();
  #stagingBatch: PendingMcpToolCall[] = [];
  #batchTimer?: NodeJS.Timeout;
  #batchListeners: Array<(batch: PendingMcpToolCall[]) => void> = [];

  constructor(opts: {
    token: string;
    url: string;
    maxToolOutputBytes: number;
    batchWindowMs: number;
    onDispose: () => void;
  }) {
    this.token = opts.token;
    this.#url = opts.url;
    this.#maxToolOutputBytes = opts.maxToolOutputBytes;
    this.#batchWindowMs = opts.batchWindowMs;
    this.#onDispose = opts.onDispose;
  }

  /** ACP `session/new` / `session/load` MCP server descriptor (loopback HTTP). */
  toAcpMcpServerConfig(): {
    type: 'http';
    name: string;
    url: string;
    headers: Array<{ name: string; value: string }>;
  } {
    return {
      type: 'http',
      name: MCP_SERVER_NAME,
      url: new URL(this.path, this.#url).href,
      headers: [{ name: TOKEN_HEADER, value: this.token }],
    };
  }

  setTools(tools: readonly ToolSchema[] | undefined): void {
    this.#tools = tools ?? [];
    this.#mcpToDsh.clear();

    const used = new Set<string>();
    for (const tool of this.#tools) {
      const mcpName = encodeMcpToolName(tool.name, used);
      this.#mcpToDsh.set(mcpName, tool);
    }
  }

  get hasTools(): boolean {
    return this.#tools.length > 0;
  }

  /** Encoded names as advertised by this endpoint in MCP tools/list. */
  get bridgeToolNames(): string[] {
    return [...this.#mcpToDsh.keys()];
  }

  get pendingCount(): number {
    return this.#pendingById.size;
  }

  getPendingCalls(): PendingMcpToolCall[] {
    return [...this.#pendingById.values()];
  }

  getPendingCall(toolCallId: string): PendingMcpToolCall | undefined {
    return this.#pendingById.get(toolCallId);
  }

  /** Check whether an ACP `session/request_permission` tool request belongs to our DSH bridge. */
  isAllowedPermissionRequest(toolName?: string, rawInput?: Record<string, unknown>): boolean {
    const prefix = `mcp_${MCP_SERVER_NAME}_`;
    if (toolName?.startsWith(prefix)) return this.#mcpToDsh.has(toolName.slice(prefix.length));
    // Only the ACP MCP dispatcher takes server/tool routing fields as input.
    if (toolName !== 'call_mcp_tool') return false;
    const serverName =
      (rawInput?.ServerName as string | undefined) ??
      (rawInput?.serverName as string | undefined) ??
      (rawInput?.server_name as string | undefined);
    const innerTool =
      (rawInput?.ToolName as string | undefined) ??
      (rawInput?.toolName as string | undefined) ??
      (rawInput?.tool_name as string | undefined);
    if (serverName === MCP_SERVER_NAME && innerTool && this.#mcpToDsh.has(innerTool)) {
      return true;
    }
    return false;
  }

  /** Wait for the next coalesced batch of MCP tool calls during an active prompt turn. */
  onNextBatch(listener: (batch: PendingMcpToolCall[]) => void): () => void {
    this.#batchListeners.push(listener);
    return () => {
      const idx = this.#batchListeners.indexOf(listener);
      if (idx >= 0) this.#batchListeners.splice(idx, 1);
    };
  }

  /** Resolve one suspended MCP tool call with DSH's `ToolResultMessage` content. */
  resolveToolCall(toolCallId: string, result: { text: string; isError?: boolean }): boolean {
    const pending = this.#pendingById.get(toolCallId);
    if (!pending) return false;
    this.#pendingById.delete(toolCallId);
    pending.resolve({
      text: truncateToolOutputUtf8(result.text, this.#maxToolOutputBytes),
      isError: result.isError,
    });
    return true;
  }

  /** Cancel all suspended MCP tool calls (e.g. on user abort or session teardown). */
  cancelAllPending(reason = 'Cancelled'): void {
    if (this.#batchTimer) {
      clearTimeout(this.#batchTimer);
      this.#batchTimer = undefined;
    }
    this.#stagingBatch = [];
    for (const pending of this.#pendingById.values()) {
      pending.cancel(reason);
    }
    this.#pendingById.clear();
  }

  dispose(): void {
    this.cancelAllPending('MCP endpoint disposed');
    this.#batchListeners = [];
    this.#onDispose();
  }

  /** Handle one JSON-RPC message sent by `agy` over Streamable HTTP. */
  handleRpc(msg: JsonRpcRequest, res: http.ServerResponse): void {
    if (msg.id === undefined || msg.id === null) {
      res.writeHead(202).end();
      return;
    }

    const sendResult = (result: unknown) => {
      if (res.writableEnded) return;
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ jsonrpc: '2.0', id: msg.id, result }));
    };

    const sendError = (code: number, message: string) => {
      if (res.writableEnded) return;
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ jsonrpc: '2.0', id: msg.id, error: { code, message } }));
    };

    if (msg.method === 'initialize') {
      const requestedVersion =
        typeof msg.params?.protocolVersion === 'string'
          ? msg.params.protocolVersion
          : '2025-03-26';
      sendResult({
        protocolVersion: requestedVersion,
        capabilities: { tools: {} },
        serverInfo: { name: MCP_SERVER_NAME, version: '0.2.0' },
      });
      return;
    }

    if (msg.method === 'ping') {
      sendResult({});
      return;
    }

    if (msg.method === 'tools/list') {
      const tools = [...this.#mcpToDsh.entries()].map(([mcpName, schema]) => ({
        name: mcpName,
        description: schema.description ?? '',
        inputSchema: schema.parameters ?? { type: 'object', properties: {} },
      }));
      sendResult({ tools });
      return;
    }

    if (msg.method === 'tools/call') {
      const mcpToolName = typeof msg.params?.name === 'string' ? msg.params.name : '';
      const schema = this.#mcpToDsh.get(mcpToolName);
      if (!schema) {
        sendResult({
          content: [{ type: 'text', text: `Unknown tool: ${mcpToolName}` }],
          isError: true,
        });
        return;
      }

      const sanitizedArgs = sanitizeMcpToolArguments(msg.params?.arguments, schema);
      const argumentsJson = JSON.stringify(sanitizedArgs);
      const toolCallId = `agy_call_${++this.#callSeq}_${crypto.randomBytes(4).toString('hex')}`;

      let settled = false;
      const pending: PendingMcpToolCall = {
        toolCallId,
        toolName: schema.name,
        argumentsJson,
        receivedAt: Date.now(),
        resolve: ({ text, isError }) => {
          if (settled) return;
          settled = true;
          this.#pendingById.delete(toolCallId);
          sendResult({
            content: [{ type: 'text', text }],
            ...(isError ? { isError: true } : {}),
          });
        },
        cancel: (reason = 'Tool execution cancelled') => {
          if (settled) return;
          settled = true;
          this.#pendingById.delete(toolCallId);
          sendError(-32800, reason);
        },
      };

      res.on('close', () => {
        if (!settled) {
          settled = true;
          this.#pendingById.delete(toolCallId);
        }
      });

      this.#pendingById.set(toolCallId, pending);
      this.#enqueueForBatch(pending);
      return;
    }

    sendError(-32601, `Unsupported MCP method: ${String(msg.method)}`);
  }

  #enqueueForBatch(call: PendingMcpToolCall): void {
    this.#stagingBatch.push(call);
    if (this.#batchWindowMs <= 0) {
      this.#flushBatch();
      return;
    }
    if (!this.#batchTimer) {
      this.#batchTimer = setTimeout(() => {
        this.#batchTimer = undefined;
        this.#flushBatch();
      }, this.#batchWindowMs);
    }
  }

  #flushBatch(): void {
    if (this.#stagingBatch.length === 0) return;
    const batch = this.#stagingBatch;
    this.#stagingBatch = [];
    for (const listener of [...this.#batchListeners]) {
      listener(batch);
    }
  }
}

/**
 * Shared loopback Streamable-HTTP MCP bridge bound exclusively to `127.0.0.1:0`.
 */
export class LoopbackMcpBridge {
  #server?: http.Server;
  #url?: string;
  #listenPromise?: Promise<string>;
  readonly #endpoints = new Map<string, McpSessionEndpoint>();

  async ensureStarted(): Promise<string> {
    if (this.#url) return this.#url;
    if (this.#listenPromise) return this.#listenPromise;

    this.#listenPromise = new Promise<string>((resolve, reject) => {
      const server = http.createServer((req, res) => {
        this.#handleHttpRequest(req, res);
      });
      server.once('error', reject);
      server.listen(0, '127.0.0.1', () => {
        server.removeListener('error', reject);
        const addr = server.address() as AddressInfo;
        this.#server = server;
        this.#url = `http://127.0.0.1:${addr.port}/mcp`;
        server.unref();
        resolve(this.#url);
      });
    });

    return this.#listenPromise;
  }

  async createSessionEndpoint(opts: {
    maxToolOutputBytes: number;
    batchWindowMs: number;
  }): Promise<McpSessionEndpoint> {
    const url = await this.ensureStarted();
    const token = crypto.randomBytes(24).toString('hex');
    const endpoint = new McpSessionEndpoint({
      token,
      url,
      maxToolOutputBytes: opts.maxToolOutputBytes,
      batchWindowMs: opts.batchWindowMs,
      onDispose: () => {
        this.#endpoints.delete(token);
      },
    });
    this.#endpoints.set(token, endpoint);
    return endpoint;
  }

  async close(): Promise<void> {
    for (const endpoint of this.#endpoints.values()) {
      endpoint.dispose();
    }
    this.#endpoints.clear();
    const srv = this.#server;
    this.#server = undefined;
    this.#url = undefined;
    this.#listenPromise = undefined;
    if (srv) {
      await new Promise<void>((resolve) => srv.close(() => resolve()));
    }
  }

  #handleHttpRequest(req: http.IncomingMessage, res: http.ServerResponse): void {
    const rawToken = req.headers[TOKEN_HEADER];
    const token = Array.isArray(rawToken) ? rawToken[0] : rawToken;
    const endpoint = token ? this.#endpoints.get(token) : undefined;
    if (!endpoint || req.url !== endpoint.path) {
      res.writeHead(401, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: 'Unauthorized MCP session token' }));
      return;
    }

    if (req.method === 'DELETE') {
      res.writeHead(200).end();
      return;
    }

    if (req.method === 'GET') {
      res.writeHead(405, { allow: 'POST, DELETE' }).end();
      return;
    }

    if (req.method !== 'POST') {
      res.writeHead(405, { allow: 'POST, DELETE' }).end();
      return;
    }

    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer) => chunks.push(chunk));
    req.on('end', () => {
      if (this.#endpoints.get(endpoint.token) !== endpoint) {
        res.writeHead(401).end();
        return;
      }
      try {
        const body = Buffer.concat(chunks).toString('utf8');
        const parsed = JSON.parse(body) as JsonRpcRequest | JsonRpcRequest[];
        if (Array.isArray(parsed)) {
          // Handle JSON-RPC batch (rare, usually notifications + single request)
          const firstWithId = parsed.find((item) => item.id !== undefined && item.id !== null);
          if (!firstWithId) {
            res.writeHead(202).end();
            return;
          }
          endpoint.handleRpc(firstWithId, res);
          return;
        }
        endpoint.handleRpc(parsed, res);
      } catch (err) {
        res.writeHead(400, { 'content-type': 'application/json' });
        res.end(
          JSON.stringify({
            jsonrpc: '2.0',
            id: null,
            error: { code: -32700, message: `Invalid JSON: ${String(err)}` },
          }),
        );
      }
    });
  }
}
