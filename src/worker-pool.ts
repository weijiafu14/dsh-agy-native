import { ToolCallId } from '@deepseek-ai/dsh-llm';
import type {
  GenerateOptions,
  StreamChunk,
  TokenUsage,
} from '@deepseek-ai/dsh-llm';
import { AcpWorkerClient, type AcpAgentCapabilitiesSummary, type AcpStreamCallbacks } from './acp-client.js';
import { resolveAgyBinary, supportsProcessPerSession } from './binary.js';
import type { AgyModelCatalog } from './catalog.js';
import {
  computeToolsFingerprint,
  type LoopbackMcpBridge,
  type McpSessionEndpoint,
} from './mcp-bridge.js';
import {
  buildFullReplayPromptParts,
  buildIncrementalPromptParts,
  computeContextDigests,
  computeHistoryPrefixDigest,
  extractReplayMetadata,
  extractSystemPrompt,
  formatHarnessGuidelines,
  planSessionSync,
  type ImageBytesLoader,
} from './session-sync.js';
import type {
  AcpPromptResponse,
  AcpPromptUsage,
  AgyReplayMetadata,
  PendingMcpToolCall,
  ResolvedAgyAcpConfig,
} from './types.js';

type TurnQueueEvent =
  | { kind: 'thought'; text: string }
  | { kind: 'text'; text: string }
  | { kind: 'tool-batch'; calls: PendingMcpToolCall[] }
  | { kind: 'prompt-done'; response: AcpPromptResponse }
  | { kind: 'error'; error: Error }
  | { kind: 'aborted'; reason: Error };

class AsyncEventQueue<T> {
  readonly #items: T[] = [];
  readonly #waiters: Array<(result: IteratorResult<T>) => void> = [];
  #closed = false;

  push(item: T): void {
    if (this.#closed) return;
    const waiter = this.#waiters.shift();
    if (waiter) {
      waiter({ value: item, done: false });
    } else {
      this.#items.push(item);
    }
  }

  close(): void {
    if (this.#closed) return;
    this.#closed = true;
    while (this.#waiters.length > 0) {
      const waiter = this.#waiters.shift()!;
      waiter({ value: undefined as unknown as T, done: true });
    }
  }

  async next(): Promise<IteratorResult<T>> {
    if (this.#items.length > 0) {
      return { value: this.#items.shift()!, done: false };
    }
    if (this.#closed) {
      return { value: undefined as unknown as T, done: true };
    }
    return new Promise<IteratorResult<T>>((resolve) => {
      this.#waiters.push(resolve);
    });
  }
}

export function mapAcpUsageToDsh(usage?: AcpPromptUsage): TokenUsage | undefined {
  if (!usage) return undefined;
  const inputTokens = Math.max(0, Math.round(usage.inputTokens ?? 0));
  const outputTokens = Math.max(0, Math.round(usage.outputTokens ?? 0));
  const cacheReadTokens =
    typeof usage.cachedReadTokens === 'number'
      ? Math.max(0, Math.round(usage.cachedReadTokens))
      : undefined;
  const cacheWriteTokens =
    typeof usage.cachedWriteTokens === 'number'
      ? Math.max(0, Math.round(usage.cachedWriteTokens))
      : undefined;
  const reasoningTokens =
    typeof usage.thoughtTokens === 'number'
      ? Math.max(0, Math.round(usage.thoughtTokens))
      : undefined;
  const totalTokens =
    typeof usage.totalTokens === 'number'
      ? Math.max(0, Math.round(usage.totalTokens))
      : inputTokens + outputTokens + (cacheReadTokens ?? 0) + (cacheWriteTokens ?? 0);

  return {
    inputTokens,
    outputTokens,
    totalTokens,
    ...(typeof cacheReadTokens === 'number' ? { cacheReadTokens } : {}),
    ...(typeof cacheWriteTokens === 'number' ? { cacheWriteTokens } : {}),
    ...(typeof reasoningTokens === 'number' ? { reasoningTokens } : {}),
  };
}

export interface InFlightTurnUsageState {
  emittedInput: number;
  emittedOutput: number;
  emittedCacheRead: number;
  emittedCacheWrite: number;
  emittedSteps: number;
  toolBatchCount: number;
  pendingStepUsed?: number;
  lastStepUsed?: number;
}

export function createInFlightTurnUsageState(): InFlightTurnUsageState {
  return {
    emittedInput: 0,
    emittedOutput: 0,
    emittedCacheRead: 0,
    emittedCacheWrite: 0,
    emittedSteps: 0,
    toolBatchCount: 0,
  };
}

/**
 * Estimate single-step `TokenUsage` for an intermediate tool-calling step (`j < K`)
 * from ACP's real-time `usage_update.used` count so DSH's context-occupancy meter
 * reflects single-step context pressure while preserving cumulative session totals.
 */
export function estimateIntermediateStepUsage(opts: {
  stepUsedTokens: number;
  emittedOutputChars: number;
  cachedPromptBaseline: number;
}): { usage: TokenUsage; nextCachedPromptBaseline: number } {
  const stepUsed = Math.max(0, Math.round(opts.stepUsedTokens));
  const estOutput = Math.min(
    Math.floor(stepUsed * 0.25),
    Math.max(1, Math.ceil(Math.max(0, opts.emittedOutputChars) / 4)),
  );
  const stepPrompt = Math.max(0, stepUsed - estOutput);
  const baseline = Math.max(0, Math.round(opts.cachedPromptBaseline));
  const stepCacheRead = Math.min(stepPrompt, baseline);
  const stepInput = stepPrompt - stepCacheRead;
  const nextCachedPromptBaseline = Math.max(baseline, Math.floor(stepPrompt * 0.8));

  return {
    usage: {
      inputTokens: stepInput,
      outputTokens: estOutput,
      totalTokens: stepUsed,
      ...(stepCacheRead > 0 ? { cacheReadTokens: stepCacheRead } : {}),
    },
    nextCachedPromptBaseline,
  };
}

/**
 * Reconcile ACP's cumulative `session/prompt` end-of-turn usage against intermediate
 * step usages emitted during tool loops so that:
 * 1. Across steps `1..K`, `sum(step.usage)` equals ACP's cumulative turn totals;
 * 2. Final step `K` reports only step `K`'s single-step prompt size (avoiding `K × context` 100% spikes).
 */
export function reconcileFinalStepUsage(
  rawUsage: AcpPromptUsage | undefined,
  turnState: InFlightTurnUsageState | undefined,
): { usage: TokenUsage | undefined; nextCachedPromptBaseline: number } {
  const full = mapAcpUsageToDsh(rawUsage);
  if (!full) {
    return { usage: undefined, nextCachedPromptBaseline: 0 };
  }

  const fullCacheRead = full.cacheReadTokens ?? 0;
  const fullCacheWrite = full.cacheWriteTokens ?? 0;
  const fullInput = full.inputTokens;
  const fullOutput = full.outputTokens;
  const fullPrompt = fullInput + fullCacheRead + fullCacheWrite;
  const totalSteps = (turnState?.toolBatchCount ?? 0) + 1;

  if (!turnState || totalSteps <= 1) {
    return {
      usage: full,
      nextCachedPromptBaseline: Math.floor(fullPrompt * 0.85),
    };
  }

  if (turnState.emittedSteps === 0) {
    const targetPrompt =
      turnState.lastStepUsed && turnState.lastStepUsed > 0
        ? Math.min(
            fullPrompt,
            Math.max(0, Math.round(turnState.lastStepUsed - fullOutput / totalSteps)),
          )
        : Math.round(fullPrompt / totalSteps);
    const scale = fullPrompt > 0 ? targetPrompt / fullPrompt : 0;
    const stepCacheRead = Math.round(fullCacheRead * scale);
    const stepCacheWrite = Math.round(fullCacheWrite * scale);
    const stepInput = Math.max(0, targetPrompt - stepCacheRead - stepCacheWrite);
    const stepOutput = Math.round(fullOutput / totalSteps);
    return {
      usage: {
        inputTokens: stepInput,
        outputTokens: stepOutput,
        totalTokens: targetPrompt + stepOutput,
        ...(typeof full.cacheReadTokens === 'number' ? { cacheReadTokens: stepCacheRead } : {}),
        ...(typeof full.cacheWriteTokens === 'number' ? { cacheWriteTokens: stepCacheWrite } : {}),
        ...(typeof full.reasoningTokens === 'number'
          ? { reasoningTokens: full.reasoningTokens }
          : {}),
      },
      nextCachedPromptBaseline: Math.floor(targetPrompt * 0.85),
    };
  }

  const emittedPrompt =
    turnState.emittedInput + turnState.emittedCacheRead + turnState.emittedCacheWrite;
  const missedIntermediateSteps = Math.max(0, turnState.toolBatchCount - turnState.emittedSteps);
  let targetPrompt = Math.max(0, fullPrompt - emittedPrompt);
  if (missedIntermediateSteps > 0) {
    targetPrompt =
      turnState.lastStepUsed && turnState.lastStepUsed > 0
        ? Math.min(targetPrompt, Math.round(turnState.lastStepUsed))
        : Math.round(targetPrompt / (missedIntermediateSteps + 1));
  }

  let remInput = Math.max(0, fullInput - turnState.emittedInput);
  let remCacheRead = Math.max(0, fullCacheRead - turnState.emittedCacheRead);
  let remCacheWrite = Math.max(0, fullCacheWrite - turnState.emittedCacheWrite);
  const remPromptSum = remInput + remCacheRead + remCacheWrite;

  if (remPromptSum > targetPrompt && remPromptSum > 0) {
    const scale = targetPrompt / remPromptSum;
    remCacheRead = Math.round(remCacheRead * scale);
    remCacheWrite = Math.round(remCacheWrite * scale);
    remInput = Math.max(0, targetPrompt - remCacheRead - remCacheWrite);
  }

  const remOutput = Math.max(0, fullOutput - turnState.emittedOutput);
  const stepPrompt = remInput + remCacheRead + remCacheWrite;

  return {
    usage: {
      inputTokens: remInput,
      outputTokens: remOutput,
      totalTokens: stepPrompt + remOutput,
      ...(typeof full.cacheReadTokens === 'number' || remCacheRead > 0
        ? { cacheReadTokens: remCacheRead }
        : {}),
      ...(typeof full.cacheWriteTokens === 'number' || remCacheWrite > 0
        ? { cacheWriteTokens: remCacheWrite }
        : {}),
      ...(typeof full.reasoningTokens === 'number'
        ? { reasoningTokens: full.reasoningTokens }
        : {}),
    },
    nextCachedPromptBaseline:
      fullCacheRead > 0 || stepPrompt > 0 ? Math.floor(stepPrompt * 0.85) : 0,
  };
}

/** Owns one ACP transport, including its initialization and crash replacement. */
class SharedTransport {
  #client?: AcpWorkerClient;
  #opening?: Promise<AcpWorkerClient>;
  #closed = false;

  async get(cfg: ResolvedAgyAcpConfig): Promise<AcpWorkerClient> {
    if (this.#closed) throw new Error('ACP transport disposed');
    if (this.#opening) return this.#opening;
    if (this.#client?.alive) return this.#client;
    this.#opening = this.#open(cfg).finally(() => { this.#opening = undefined; });
    return this.#opening;
  }

  async #open(cfg: ResolvedAgyAcpConfig): Promise<AcpWorkerClient> {
    await this.#client?.dispose();
    const agyPath = resolveAgyBinary(cfg.agyPath);
    const client = new AcpWorkerClient({
      agyPath,
      cwd: cfg.workingDirectory,
      startupTimeoutMs: cfg.startupTimeoutMs,
      extraArgs: await supportsProcessPerSession(agyPath) ? ['--process-per-session'] : [],
    });
    this.#client = client;
    try {
      await client.start();
      return client;
    } catch (error) {
      await client.dispose();
      throw error;
    }
  }

  async dispose(): Promise<void> {
    this.#closed = true;
    await this.#opening?.catch(() => {});
    await this.#client?.dispose();
  }
}

/** Session-local stream, bridge and replay state. */
export class SessionRuntime {
  readonly id: string;
  readonly #getConfig: () => ResolvedAgyAcpConfig;
  readonly #mcpBridge: LoopbackMcpBridge;
  readonly #catalog: AgyModelCatalog;
  readonly #loadImage?: ImageBytesLoader;

  readonly #transport: SharedTransport;
  readonly #ownsTransport: boolean;
  readonly #onIdle?: () => void;
  #client?: AcpWorkerClient;
  #endpoint?: McpSessionEndpoint;
  #activeAcpSessionId?: string;
  #activeRawModelId?: string;
  #lastToolsFingerprint?: string;
  #lastContextDigests?: Record<string, string>;
  #lastHistoryPrefixDigest?: string;
  #cachedPromptBaseline = 0;
  #inFlightPrompt?: Promise<AcpPromptResponse>;
  #inFlightTurnUsage?: InFlightTurnUsageState;

  #busy = false;
  #lastUsedAt = Date.now();
  #idleTimer?: NodeJS.Timeout;
  #boundKey?: string;

  constructor(opts: {
    id: string;
    transport?: SharedTransport;
    onIdle?: () => void;
    getConfig: () => ResolvedAgyAcpConfig;
    mcpBridge: LoopbackMcpBridge;
    catalog: AgyModelCatalog;
    loadImage?: ImageBytesLoader;
  }) {
    this.id = opts.id;
    this.#transport = opts.transport ?? new SharedTransport();
    this.#ownsTransport = !opts.transport;
    this.#onIdle = opts.onIdle;
    this.#getConfig = opts.getConfig;
    this.#mcpBridge = opts.mcpBridge;
    this.#catalog = opts.catalog;
    this.#loadImage = opts.loadImage;
  }

  get busy(): boolean {
    return this.#busy;
  }

  get hasPendingToolTurn(): boolean {
    return Boolean(this.#inFlightPrompt && (this.#endpoint?.pendingCount ?? 0) > 0);
  }

  get lastUsedAt(): number {
    return this.#lastUsedAt;
  }

  get boundKey(): string | undefined {
    return this.#boundKey;
  }

  setBoundKey(key: string | undefined): void {
    this.#boundKey = key;
  }

  get activeAcpSessionId(): string | undefined {
    return this.#activeAcpSessionId;
  }

  /** Ensure `agy` is started and fetch catalog models via a lightweight probe session if needed. */
  async probeCatalog(): Promise<AcpAgentCapabilitiesSummary> {
    const cfg = this.#getConfig();
    const client = await this.#ensureClient(cfg);
    const caps = await client.start();
    if (!this.#catalog.snapshot) {
      const sess = await client.createSession({ cwd: cfg.workingDirectory });
      if (sess.availableModels.length > 0) {
        this.#catalog.update(sess.availableModels, sess.tier);
      }
      await client.closeSession(sess.sessionId);
    }
    this.#scheduleIdleTimer(cfg.idleTtlMs);
    return caps;
  }

  async *streamTurn(options: GenerateOptions<string>): AsyncGenerator<StreamChunk> {
    if (options.signal?.aborted) {
      throw options.signal.reason ?? new Error('Request aborted before start');
    }

    this.#clearIdleTimer();
    this.#busy = true;
    const now = Date.now();
    if (now - this.#lastUsedAt > 270_000) {
      this.#cachedPromptBaseline = 0;
    }
    this.#lastUsedAt = now;

    const cfg = this.#getConfig();
    let targetRawModelId = this.#catalog.resolveRawModelId(
      options.model,
      options.reasoningEffort,
    );
    const toolsFp = computeToolsFingerprint(options.tools);
    const systemPrompt = extractSystemPrompt(options.messages, options.system);

    try {
      const client = await this.#ensureClient(cfg);

      const pendingIds = new Set(this.#endpoint?.getPendingCalls().map((c) => c.toolCallId));
      const syncPlan = planSessionSync({
        messages: options.messages,
        activeAcpSessionId: this.#activeAcpSessionId,
        lastToolsFingerprint: this.#lastToolsFingerprint,
        currentToolsFingerprint: toolsFp,
        lastContextDigests: this.#lastContextDigests,
        lastHistoryPrefixDigest: this.#lastHistoryPrefixDigest,
        pendingToolCallIds: pendingIds,
        allowResumeFromReplay: cfg.resumeSessions && !!options.sessionId && options.purpose === undefined,
        dshSessionId: options.sessionId,
        forwardContextUpdates: cfg.forwardContextUpdates,
      });

      if (syncPlan.kind !== 'resolve-pending-tools') {
        const replace = syncPlan.kind !== 'warm-incremental';
        await this.#abortInFlightTurn(replace);
        if (replace) {
          this.#endpoint?.dispose();
          this.#endpoint = undefined;
        }
      }
      let endpoint = await this.#ensureEndpoint(cfg);
      endpoint.setTools(options.tools);
      let resumed = false;
      if (syncPlan.kind === 'resume-incremental' && client.capabilities?.loadSession) {
        try {
          await client.loadSession({ sessionId: syncPlan.resumeAcpSessionId, endpoint, cwd: cfg.workingDirectory });
          this.#activeAcpSessionId = syncPlan.resumeAcpSessionId;
          resumed = true;
        } catch {
          // Even a timed-out load may have received the URL. Revoke it before cold replay.
          endpoint.dispose();
          this.#endpoint = undefined;
          endpoint = await this.#ensureEndpoint(cfg);
          endpoint.setTools(options.tools);
        }
      }
      const queue = new AsyncEventQueue<TurnQueueEvent>();
      const detachBatch = endpoint.onNextBatch((calls) => {
        queue.push({ kind: 'tool-batch', calls });
      });

      const callbacks: AcpStreamCallbacks = {
        onThoughtDelta: (text) => queue.push({ kind: 'thought', text }),
        onTextDelta: (text) => queue.push({ kind: 'text', text }),
        onUsageUpdate: (used, size) => {
          this.#catalog.recordContextWindow(targetRawModelId, size);
          if (used > 0 && this.#inFlightTurnUsage) {
            this.#inFlightTurnUsage.pendingStepUsed = used;
            this.#inFlightTurnUsage.lastStepUsed = used;
          }
        },
      };

      const onAbort = () => {
        const reason =
          options.signal?.reason instanceof Error
            ? options.signal.reason
            : new Error(String(options.signal?.reason ?? 'Request aborted'));
        endpoint.cancelAllPending(reason.message);
        if (this.#activeAcpSessionId) {
          client.cancelPrompt(this.#activeAcpSessionId);
        }
        queue.push({ kind: 'aborted', reason });
      };
      options.signal?.addEventListener('abort', onAbort, { once: true });

      let delivered = false;
      try {
        if (syncPlan.kind === 'resolve-pending-tools' && this.#inFlightPrompt) {
          client.setStreamCallbacks(this.#activeAcpSessionId!, callbacks);
          this.#attachPromptCompletion(this.#inFlightPrompt, queue);
          for (const res of syncPlan.resolutions) {
            endpoint.resolveToolCall(res.toolCallId, {
              text: res.text,
              isError: res.isError,
            });
          }
        } else {
          const routingBridgeToolNames =
            options.purpose !== undefined && !endpoint.hasTools
              ? undefined
              : endpoint.bridgeToolNames;

          let promptParts;
          if (syncPlan.kind === 'warm-incremental' || (syncPlan.kind === 'resume-incremental' && resumed)) {
            if (resumed || this.#activeRawModelId !== targetRawModelId) {
              await client.setModel(this.#activeAcpSessionId!, targetRawModelId);
              this.#activeRawModelId = targetRawModelId;
            }
            promptParts = await buildIncrementalPromptParts({
              tailMessages: syncPlan.tailMessages,
              updatedContextBlocks: syncPlan.updatedContextBlocks,
              loadImage: this.#loadImage,
              signal: options.signal,
            });
            if (resumed) {
              const routing = formatHarnessGuidelines(undefined, routingBridgeToolNames ?? false);
              if (routing) promptParts.unshift({ type: 'text', text: routing });
            }
          } else {
            targetRawModelId = await this.#openFreshSession(client, endpoint, options, cfg);
            promptParts = await buildFullReplayPromptParts({
              messages: options.messages,
              systemPrompt,
              bridgeToolNames: routingBridgeToolNames,
              loadImage: this.#loadImage,
              signal: options.signal,
            });
          }

          this.#inFlightTurnUsage = createInFlightTurnUsageState();
          if (options.signal?.aborted) throw options.signal.reason;
          client.setStreamCallbacks(this.#activeAcpSessionId!, callbacks);
          const promptPromise = client.prompt(
            this.#activeAcpSessionId!,
            promptParts,
            cfg.requestTimeoutMs > 0 ? cfg.requestTimeoutMs : undefined,
          );
          this.#inFlightPrompt = promptPromise;
          this.#attachPromptCompletion(promptPromise, queue);
        }

        // Stream events out as DSH StreamChunks
        let nextBlockIndex = 0;
        let stepOutputChars = 0;
        let openBlock:
          | { index: number; type: 'reasoning' | 'text'; text: string }
          | undefined;

        const closeOpenBlock = (): StreamChunk | undefined => {
          if (!openBlock) return undefined;
          const chunk: StreamChunk = {
            type: 'block-end',
            index: openBlock.index,
            block:
              openBlock.type === 'reasoning'
                ? { type: 'reasoning', text: openBlock.text }
                : { type: 'text', text: openBlock.text },
          };
          openBlock = undefined;
          return chunk;
        };

        const buildReplayEnvelope = () => {
          const historyPrefixDigest = computeHistoryPrefixDigest(
            options.messages,
            options.messages.length,
          );
          const meta: AgyReplayMetadata = {
            version: 1,
            dshSessionId: options.sessionId,
            acpSessionId: this.#activeAcpSessionId ?? '',
            rawModelId: targetRawModelId,
            syncedMessageCount: options.messages.length + 1,
            toolsFingerprint: toolsFp,
            contextDigests: computeContextDigests(options.messages),
            historyPrefixDigest,
          };
          this.#lastToolsFingerprint = toolsFp;
          this.#lastContextDigests = meta.contextDigests;
          this.#lastHistoryPrefixDigest = historyPrefixDigest;
          return { response: meta };
        };

        while (true) {
          const step = await queue.next();
          if (step.done) break;
          const ev = step.value;

          if (ev.kind === 'thought') {
            if (openBlock && openBlock.type !== 'reasoning') {
              yield closeOpenBlock()!;
            }
            if (!openBlock) {
              openBlock = { index: nextBlockIndex++, type: 'reasoning', text: '' };
              yield { type: 'block-start', index: openBlock.index, blockType: 'reasoning' };
            }
            openBlock.text += ev.text;
            stepOutputChars += ev.text.length;
            yield { type: 'reasoning-delta', index: openBlock.index, text: ev.text };
            continue;
          }

          if (ev.kind === 'text') {
            if (openBlock && openBlock.type !== 'text') {
              yield closeOpenBlock()!;
            }
            if (!openBlock) {
              openBlock = { index: nextBlockIndex++, type: 'text', text: '' };
              yield { type: 'block-start', index: openBlock.index, blockType: 'text' };
            }
            openBlock.text += ev.text;
            stepOutputChars += ev.text.length;
            yield { type: 'text-delta', index: openBlock.index, text: ev.text };
            continue;
          }

          if (ev.kind === 'tool-batch') {
            const closed = closeOpenBlock();
            if (closed) yield closed;

            for (const call of ev.calls) {
              const idx = nextBlockIndex++;
              const callId = ToolCallId(call.toolCallId);
              stepOutputChars += call.toolName.length + call.argumentsJson.length;
              yield { type: 'block-start', index: idx, blockType: 'tool-call' };
              yield {
                type: 'tool-call-delta',
                index: idx,
                id: callId,
                name: call.toolName,
                argumentsDelta: call.argumentsJson,
              };
              yield {
                type: 'block-end',
                index: idx,
                block: {
                  type: 'tool-call',
                  id: callId,
                  name: call.toolName,
                  arguments: call.argumentsJson,
                },
              };
            }

            const turnUsage = this.#inFlightTurnUsage;
            if (turnUsage) {
              turnUsage.toolBatchCount += 1;
              if (turnUsage.pendingStepUsed !== undefined && turnUsage.pendingStepUsed > 0) {
                const stepUsed = turnUsage.pendingStepUsed;
                turnUsage.pendingStepUsed = undefined;
                const { usage: stepUsage, nextCachedPromptBaseline } =
                  estimateIntermediateStepUsage({
                    stepUsedTokens: stepUsed,
                    emittedOutputChars: stepOutputChars,
                    cachedPromptBaseline: this.#cachedPromptBaseline,
                  });
                this.#cachedPromptBaseline = nextCachedPromptBaseline;
                turnUsage.emittedInput += stepUsage.inputTokens;
                turnUsage.emittedOutput += stepUsage.outputTokens;
                turnUsage.emittedCacheRead += stepUsage.cacheReadTokens ?? 0;
                turnUsage.emittedCacheWrite += stepUsage.cacheWriteTokens ?? 0;
                turnUsage.emittedSteps += 1;
                yield { type: 'usage', usage: stepUsage };
              }
            }

            delivered = true;
            yield {
              type: 'finish',
              reason: { kind: 'tool-calls' },
              replayState: buildReplayEnvelope(),
            };
            queue.close();
            break;
          }

          if (ev.kind === 'prompt-done') {
            const turnUsage = this.#inFlightTurnUsage;
            this.#inFlightTurnUsage = undefined;
            this.#inFlightPrompt = undefined;
            const closed = closeOpenBlock();
            if (closed) yield closed;

            const { usage, nextCachedPromptBaseline } = reconcileFinalStepUsage(
              ev.response.usage,
              turnUsage,
            );
            if (usage) {
              this.#cachedPromptBaseline = nextCachedPromptBaseline;
              yield { type: 'usage', usage };
            }

            const stopReason =
              ev.response.stopReason === 'max_tokens'
                ? ({ kind: 'max-tokens' } as const)
                : ({ kind: 'stop' } as const);

            delivered = true;
            yield {
              type: 'finish',
              reason: stopReason,
              replayState: buildReplayEnvelope(),
            };
            queue.close();
            break;
          }

          if (ev.kind === 'aborted') {
            this.#inFlightTurnUsage = undefined;
            this.#inFlightPrompt = undefined;
            queue.close();
            throw ev.reason;
          }

          if (ev.kind === 'error') {
            this.#inFlightTurnUsage = undefined;
            this.#inFlightPrompt = undefined;
            queue.close();
            throw ev.error;
          }
        }
      } finally {
        options.signal?.removeEventListener('abort', onAbort);
        detachBatch();
        queue.close();
        if (this.#activeAcpSessionId) client.setStreamCallbacks(this.#activeAcpSessionId);
        if (!delivered) {
          endpoint.dispose();
          this.#endpoint = undefined;
          await this.#abortInFlightTurn(true);
        }
      }
    } finally {
      this.#busy = false;
      this.#lastUsedAt = Date.now();
      this.#scheduleIdleTimer(cfg.idleTtlMs);
    }
  }

  async dispose(): Promise<void> {
    this.#clearIdleTimer();
    this.#endpoint?.dispose();
    this.#endpoint = undefined;
    this.#inFlightTurnUsage = undefined;
    this.#inFlightPrompt = undefined;
    const sessionId = this.#activeAcpSessionId;
    this.#clearActiveSessionState();
    const client = this.#client;
    this.#client = undefined;
    if (sessionId) await client?.closeSession(sessionId);
    if (this.#ownsTransport) await this.#transport.dispose();
  }

  async #ensureClient(cfg: ResolvedAgyAcpConfig): Promise<AcpWorkerClient> {
    const client = await this.#transport.get(cfg);
    if (client !== this.#client) {
      this.#endpoint?.dispose();
      this.#endpoint = undefined;
      this.#clearActiveSessionState();
      this.#inFlightPrompt = undefined;
      this.#inFlightTurnUsage = undefined;
      this.#client = client;
    }
    return client;
  }

  async #ensureEndpoint(cfg: ResolvedAgyAcpConfig): Promise<McpSessionEndpoint> {
    if (this.#endpoint) return this.#endpoint;
    this.#endpoint = await this.#mcpBridge.createSessionEndpoint({
      maxToolOutputBytes: cfg.maxToolOutputBytes,
      batchWindowMs: cfg.batchWindowMs,
    });
    return this.#endpoint;
  }

  async #openFreshSession(
    client: AcpWorkerClient,
    endpoint: McpSessionEndpoint,
    options: Pick<GenerateOptions<string>, 'model' | 'reasoningEffort'>,
    cfg: ResolvedAgyAcpConfig,
  ): Promise<string> {
    if (this.#activeAcpSessionId) {
      await client.closeSession(this.#activeAcpSessionId);
      this.#clearActiveSessionState();
    }

    const created = await client.createSession({
      endpoint,
      cwd: cfg.workingDirectory,
    });

    this.#activeAcpSessionId = created.sessionId;
    if (created.availableModels.length > 0) {
      this.#catalog.update(created.availableModels, created.tier);
    }

    const targetRawModelId = this.#catalog.resolveRawModelId(
      options.model,
      options.reasoningEffort,
    );

    const modelAvailable =
      created.availableModels.length === 0 ||
      created.availableModels.some((m) => m.modelId === targetRawModelId);
    if (!modelAvailable) {
      throw new Error(
        `dsh-agy-native: model '${targetRawModelId}' is not available on this Google subscription (${created.tier?.name ?? 'unknown tier'}). Available: ${created.availableModels.map((m) => m.modelId).join(', ')}`,
      );
    }

    if (created.currentModelId !== targetRawModelId) {
      await client.setModel(created.sessionId, targetRawModelId);
    }

    this.#activeRawModelId = targetRawModelId;
    return targetRawModelId;
  }

  async #abortInFlightTurn(closeExistingSession: boolean): Promise<void> {
    this.#endpoint?.cancelAllPending('Turn superseded');
    if (this.#inFlightPrompt && this.#activeAcpSessionId && this.#client?.alive) {
      this.#client.cancelPrompt(this.#activeAcpSessionId);
      await Promise.race([
        this.#inFlightPrompt.catch(() => undefined),
        new Promise<void>((r) => setTimeout(r, 500)),
      ]);
    }
    this.#inFlightTurnUsage = undefined;
    this.#inFlightPrompt = undefined;
    if (closeExistingSession) {
      if (this.#activeAcpSessionId && this.#client?.alive) {
        await this.#client.closeSession(this.#activeAcpSessionId);
      }
      this.#clearActiveSessionState();
    }
  }

  #clearActiveSessionState(): void {
    this.#activeAcpSessionId = undefined;
    this.#activeRawModelId = undefined;
    this.#lastToolsFingerprint = undefined;
    this.#lastContextDigests = undefined;
    this.#lastHistoryPrefixDigest = undefined;
    this.#cachedPromptBaseline = 0;
  }

  #attachPromptCompletion(
    promptPromise: Promise<AcpPromptResponse>,
    queue: AsyncEventQueue<TurnQueueEvent>,
  ): void {
    promptPromise.then(
      (response) => {
        queue.push({ kind: 'prompt-done', response });
      },
      (err: unknown) => {
        const error = err instanceof Error ? err : new Error(String(err));
        queue.push({ kind: 'error', error });
      },
    );
  }

  #scheduleIdleTimer(idleTtlMs: number): void {
    this.#clearIdleTimer();
    if (idleTtlMs <= 0) return;
    this.#idleTimer = setTimeout(() => {
      if (!this.#busy && !this.hasPendingToolTurn) {
        this.#onIdle?.();
        void this.dispose();
      } else {
        this.#scheduleIdleTimer(idleTtlMs);
      }
    }, idleTtlMs);
    this.#idleTimer.unref?.();
  }

  #clearIdleTimer(): void {
    if (this.#idleTimer) {
      clearTimeout(this.#idleTimer);
      this.#idleTimer = undefined;
    }
  }
}

interface AgyPoolOptions {
  getConfig: () => ResolvedAgyAcpConfig;
  mcpBridge: LoopbackMcpBridge;
  catalog: AgyModelCatalog;
  loadImage?: ImageBytesLoader;
}

/** Multiplex independent DSH conversations over one long-lived ACP connection. */
export class AgyWorkerPool {
  readonly #sessions = new Map<string | symbol, SessionRuntime>();
  #transport = new SharedTransport();
  readonly #locks = new Map<SessionRuntime, Promise<void>>();
  readonly #opts: AgyPoolOptions;
  #probe?: Promise<void>;

  constructor(opts: AgyPoolOptions) {
    this.#opts = opts;
  }

  async ensureCatalogLoaded(): Promise<void> {
    if (this.#opts.catalog.snapshot) return;
    return this.#probe ??= this.#probeCatalog().finally(() => { this.#probe = undefined; });
  }

  async #probeCatalog(): Promise<void> {
    const probe = new SessionRuntime({ ...this.#opts, transport: this.#transport, id: 'catalog' });
    try {
      await probe.probeCatalog();
    } finally {
      await probe.dispose();
    }
  }

  async *stream(options: GenerateOptions<string>): AsyncGenerator<StreamChunk> {
    if (options.signal?.aborted) throw options.signal.reason;
    await this.ensureCatalogLoaded();
    let key: string | symbol = options.sessionId && options.purpose === undefined
      ? String(options.sessionId) : Symbol('request');
    // Unkeyed tool continuations can rejoin only another unkeyed request.
    if (typeof key === 'symbol' && options.purpose === undefined) {
      for (let i = options.messages.length - 1; i >= 0; i--) {
        const replay = extractReplayMetadata(options.messages[i]!);
        if (!replay) continue;
        for (const [candidate, runtime] of this.#sessions) {
          if (typeof candidate === 'symbol' && runtime.activeAcpSessionId === replay.acpSessionId) key = candidate;
        }
        break;
      }
    }
    let slot = this.#sessions.get(key);
    if (!slot) {
      slot = new SessionRuntime({
        ...this.#opts, transport: this.#transport, id: String(key),
        onIdle: () => { this.#sessions.delete(key); },
      });
      this.#sessions.set(key, slot);
    }
    const previous = this.#locks.get(slot);
    let release!: () => void;
    const lock = new Promise<void>(resolve => { release = resolve; });
    this.#locks.set(slot, lock);
    await previous;
    try {
      yield* slot.streamTurn(options);
    } finally {
      release();
      if (this.#locks.get(slot) === lock) {
        this.#locks.delete(slot);
        if (typeof key === 'symbol' && !slot.hasPendingToolTurn) {
          this.#sessions.delete(key);
          await slot.dispose();
        }
      }
    }
  }

  async dispose(): Promise<void> {
    await Promise.allSettled([...this.#sessions.values()].map(slot => slot.dispose()));
    this.#sessions.clear();
    await this.#transport.dispose();
    // The host can restart this adapter instance after disposal. Reopen lazily on first use.
    this.#transport = new SharedTransport();
  }
}

// Preserve the public constructor exported by ./core.
export { SessionRuntime as AgyWorkerSlot };
