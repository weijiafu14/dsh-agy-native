import fs from 'node:fs';
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
  buildMcpToolResultContent,
  computeHistoryPrefixDigest,
  extractReplayMetadata,
  extractSystemPrompt,
  formatHarnessGuidelines,
  planSessionSync,
  type ImageBytesLoader,
} from './session-sync.js';
import {
  convertDshMessagesToAgySteps,
  hydrateAgyConversationDb,
  resolveAgyConversationDbPath,
  splitMessagesForTrajectorySync,
} from './trajectory-db.js';
import type {
  AcpPromptPart,
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

async function waitForSqliteWalClear(dbPath: string, maxWaitMs = 800): Promise<void> {
  const walPath = `${dbPath}-wal`;
  const deadline = Date.now() + maxWaitMs;
  while (fs.existsSync(walPath) && Date.now() < deadline) {
    await new Promise<void>((r) => setTimeout(r, 25));
  }
}

export function mapAcpUsageToDsh(usage?: AcpPromptUsage): TokenUsage | undefined {
  if (!usage) return undefined;
  const inputTokens = Math.max(0, Math.round(usage.inputTokens ?? 0));
  const rawOutput = Math.max(0, Math.round(usage.outputTokens ?? 0));
  const cacheReadTokens =
    typeof usage.cachedReadTokens === 'number'
      ? Math.max(0, Math.round(usage.cachedReadTokens))
      : 0;
  const cacheWriteTokens =
    typeof usage.cachedWriteTokens === 'number'
      ? Math.max(0, Math.round(usage.cachedWriteTokens))
      : 0;
  const reasoningTokens =
    typeof usage.thoughtTokens === 'number'
      ? Math.max(0, Math.round(usage.thoughtTokens))
      : undefined;
  const knownPrompt = inputTokens + cacheReadTokens + cacheWriteTokens;
  const rawTotal =
    typeof usage.totalTokens === 'number' ? Math.max(0, Math.round(usage.totalTokens)) : undefined;
  // Preserve @deepseek-ai/dsh-token-meter invariants:
  // `reasoningTokens <= outputTokens` and `totalTokens === knownPrompt + outputTokens`.
  const outputTokens =
    reasoningTokens !== undefined &&
    reasoningTokens > 0 &&
    rawTotal !== undefined &&
    rawTotal === knownPrompt + rawOutput + reasoningTokens
      ? rawOutput + reasoningTokens
      : Math.max(rawOutput, reasoningTokens ?? 0);
  const totalTokens = knownPrompt + outputTokens;

  return {
    inputTokens,
    outputTokens,
    totalTokens,
    cacheReadTokens,
    cacheWriteTokens,
    ...(typeof reasoningTokens === 'number' ? { reasoningTokens } : {}),
  };
}

export const IMPLICIT_CACHE_TTL_MS = 3_300_000;

export interface InFlightTurnUsageState {
  emittedInput: number;
  emittedOutput: number;
  emittedCacheRead: number;
  emittedCacheWrite: number;
  emittedReasoning: number;
  emittedSteps: number;
  toolBatchCount: number;
  pendingStepUsed?: number;
  lastStepUsed?: number;
  finalStepObservedUsed?: number;
  compactionDetected?: boolean;
  estimatedPrevTotalTokens?: number;
  pendingInputDeltaTokens?: number;
}

export function createInFlightTurnUsageState(): InFlightTurnUsageState {
  return {
    emittedInput: 0,
    emittedOutput: 0,
    emittedCacheRead: 0,
    emittedCacheWrite: 0,
    emittedReasoning: 0,
    emittedSteps: 0,
    toolBatchCount: 0,
  };
}

function estimatePromptPartsTokens(parts: readonly { type: string; text?: string }[]): number {
  let chars = 0;
  for (const part of parts) {
    if (part.type === 'text' && part.text) {
      chars += part.text.length;
    } else {
      chars += 1024;
    }
  }
  return Math.max(1, Math.ceil(chars / 4));
}

/**
 * Estimate single-step `TokenUsage` for an intermediate tool-calling step (`j < K`)
 * from ACP's real-time `usage_update.used` count (or prompt-delta fallback when omitted)
 * so DSH's context-occupancy meter reflects single-step context pressure while preserving
 * exact end-of-turn cumulative billing and cache-hit ratios.
 */
export function estimateIntermediateStepUsage(opts: {
  stepUsedTokens: number;
  emittedOutputChars: number;
  emittedThoughtChars?: number;
  cachedPromptBaseline: number;
  authoritative?: boolean;
}): { usage: TokenUsage; nextCachedPromptBaseline: number; compacted: boolean } {
  const stepUsed = Math.max(0, Math.round(opts.stepUsedTokens));
  const estOutput =
    stepUsed === 0
      ? 0
      : Math.min(
          Math.floor(stepUsed * 0.25),
          Math.max(1, Math.ceil(Math.max(0, opts.emittedOutputChars) / 4)),
        );
  const thoughtChars = Math.max(0, opts.emittedThoughtChars ?? 0);
  const totalOutChars = Math.max(1, opts.emittedOutputChars);
  const estReasoning =
    thoughtChars > 0 && estOutput > 0
      ? Math.min(estOutput, Math.max(1, Math.round((estOutput * thoughtChars) / totalOutChars)))
      : 0;
  const stepPrompt = Math.max(0, stepUsed - estOutput);
  const baseline = Math.max(0, Math.round(opts.cachedPromptBaseline));

  // Detect mid-session / mid-turn internal `agy` `CHECKPOINT` compaction where prompt size
  // drops sharply (>30% drop from a >=16k baseline): the compacted summary invalidates the
  // old prefix cache on its first step and establishes `stepPrompt` as the new baseline.
  const compacted =
    baseline >= 16_000 && stepPrompt > 0 && stepPrompt < Math.floor(baseline * 0.7);
  const stepCacheRead = compacted ? 0 : Math.min(stepPrompt, baseline);
  const stepInput = stepPrompt - stepCacheRead;
  // When `stepUsedTokens` came from an authoritative `usage_update` frame, anchor directly to
  // `stepPrompt` so any prior heuristic overshoot cannot ratchet `#cachedPromptBaseline` upward.
  const nextCachedPromptBaseline =
    compacted || opts.authoritative ? stepPrompt : Math.max(baseline, stepPrompt);

  return {
    usage: {
      inputTokens: stepInput,
      outputTokens: estOutput,
      totalTokens: stepPrompt + estOutput,
      cacheReadTokens: stepCacheRead,
      cacheWriteTokens: 0,
      ...(opts.emittedThoughtChars !== undefined ? { reasoningTokens: estReasoning } : {}),
    },
    nextCachedPromptBaseline,
    compacted,
  };
}

/**
 * Reconcile ACP's cumulative `session/prompt` end-of-turn usage against intermediate
 * step usages emitted during tool loops so that:
 * 1. Across steps `1..K`, `sum(step.usage)` equals ACP's cumulative turn totals (including cache hit %);
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
      nextCachedPromptBaseline: fullPrompt,
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
    const stepReasoning =
      typeof full.reasoningTokens === 'number'
        ? Math.min(stepOutput, Math.round(full.reasoningTokens / totalSteps))
        : undefined;
    return {
      usage: {
        inputTokens: stepInput,
        outputTokens: stepOutput,
        totalTokens: targetPrompt + stepOutput,
        cacheReadTokens: stepCacheRead,
        cacheWriteTokens: stepCacheWrite,
        ...(stepReasoning !== undefined ? { reasoningTokens: stepReasoning } : {}),
      },
      nextCachedPromptBaseline: targetPrompt,
    };
  }

  const remOutput = Math.max(0, fullOutput - turnState.emittedOutput);
  const finalObservedPrompt =
    turnState.finalStepObservedUsed && turnState.finalStepObservedUsed > 0
      ? Math.max(0, Math.round(turnState.finalStepObservedUsed - remOutput))
      : undefined;

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
  // If `agy` executed a hidden internal `CHECKPOINT` summarization call during the turn,
  // `fullPrompt - emittedPrompt` includes that hidden 500k call even though the post-checkpoint
  // final step ran on a much smaller prompt (`finalObservedPrompt`). Clamp only on genuine
  // checkpoint cliffs so normal sparse-update turns preserve 100% exact cumulative totals.
  const avgEmittedStepPrompt =
    turnState.emittedSteps > 0 ? emittedPrompt / turnState.emittedSteps : 0;
  const checkpointCliff =
    Boolean(turnState.compactionDetected) ||
    (avgEmittedStepPrompt >= 16_000 &&
      finalObservedPrompt !== undefined &&
      finalObservedPrompt > 0 &&
      finalObservedPrompt < Math.floor(avgEmittedStepPrompt * 0.7));
  if (
    checkpointCliff &&
    finalObservedPrompt !== undefined &&
    finalObservedPrompt > 0 &&
    targetPrompt > Math.round(finalObservedPrompt * 1.25)
  ) {
    targetPrompt = finalObservedPrompt;
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

  // Prevent final-step `pressureTokens` from collapsing to 0 when intermediate heuristic sub-steps
  // (e.g. split tool dispatches or mid-turn checkpoint resets) already over-emitted cumulative buckets.
  const referenceSingleStepPrompt =
    finalObservedPrompt ??
    (turnState.lastStepUsed && turnState.lastStepUsed > remOutput
      ? Math.round(turnState.lastStepUsed - remOutput)
      : 0);
  if (
    referenceSingleStepPrompt > 0 &&
    remInput + remCacheRead + remCacheWrite < Math.floor(referenceSingleStepPrompt * 0.5)
  ) {
    const cacheReadShare = fullPrompt > 0 ? fullCacheRead / fullPrompt : 0;
    const cacheWriteShare = fullPrompt > 0 ? fullCacheWrite / fullPrompt : 0;
    remCacheRead = Math.round(referenceSingleStepPrompt * cacheReadShare);
    remCacheWrite = Math.round(referenceSingleStepPrompt * cacheWriteShare);
    remInput = Math.max(0, referenceSingleStepPrompt - remCacheRead - remCacheWrite);
  }

  const stepPrompt = remInput + remCacheRead + remCacheWrite;
  const remReasoning =
    typeof full.reasoningTokens === 'number'
      ? Math.min(remOutput, Math.max(0, full.reasoningTokens - turnState.emittedReasoning))
      : turnState.emittedSteps > 0
        ? 0
        : undefined;

  const observedFinalPrompt =
    finalObservedPrompt && finalObservedPrompt > 0
      ? finalObservedPrompt
      : turnState.lastStepUsed && turnState.lastStepUsed > remOutput
        ? Math.max(stepPrompt, Math.round(turnState.lastStepUsed - remOutput))
        : stepPrompt;

  return {
    usage: {
      inputTokens: remInput,
      outputTokens: remOutput,
      totalTokens: stepPrompt + remOutput,
      cacheReadTokens: remCacheRead,
      cacheWriteTokens: remCacheWrite,
      ...(remReasoning !== undefined ? { reasoningTokens: remReasoning } : {}),
    },
    nextCachedPromptBaseline: stepPrompt > 0 || fullCacheRead > 0 ? observedFinalPrompt : 0,
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
    // Cloud Code implicit prompt caches on warm ACP sessions routinely survive 15–55 minutes.
    // Clearing at 4.5 minutes caused irreversible `uncachedInputTokens` spikes on step 1 of resumed loops.
    if (now - this.#lastUsedAt > IMPLICIT_CACHE_TTL_MS) {
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

      const pendingIds = this.#endpoint?.getDispatchedCallIds() ?? new Set<string>();
      const syncPlan = planSessionSync({
        messages: options.messages,
        explicitSystem: options.system,
        activeAcpSessionId: this.#activeAcpSessionId,
        lastToolsFingerprint: this.#lastToolsFingerprint,
        currentToolsFingerprint: toolsFp,
        lastHistoryPrefixDigest: this.#lastHistoryPrefixDigest,
        pendingToolCallIds: pendingIds,
        allowResumeFromReplay:
          cfg.resumeSessions &&
          !!options.sessionId &&
          (options.purpose === undefined || options.purpose === 'compaction'),
        dshSessionId: options.sessionId,
        purpose: options.purpose,
      });

      const priorWarmSessionId = this.#activeAcpSessionId;
      if (syncPlan.kind !== 'resolve-pending-tools') {
        const replace = syncPlan.kind !== 'warm-incremental';
        await this.#abortInFlightTurn(replace);
        if (replace || !this.#activeAcpSessionId) {
          this.#endpoint?.dispose();
          this.#endpoint = undefined;
        }
      }
      let endpoint = await this.#ensureEndpoint(cfg);
      endpoint.setTools(options.tools);
      let resumed = false;
      const reloadSessionId =
        syncPlan.kind === 'resume-incremental'
          ? syncPlan.resumeAcpSessionId
          : syncPlan.kind === 'warm-incremental' && !this.#activeAcpSessionId
            ? priorWarmSessionId
            : undefined;
      if (reloadSessionId && client.capabilities?.loadSession) {
        try {
          await client.loadSession({ sessionId: reloadSessionId, endpoint, cwd: cfg.workingDirectory });
          this.#activeAcpSessionId = reloadSessionId;
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
            this.#inFlightTurnUsage.finalStepObservedUsed = used;
          }
        },
      };

      const onAbort = () => {
        const reason =
          options.signal?.reason instanceof Error
            ? options.signal.reason
            : new Error(String(options.signal?.reason ?? 'Request aborted'));
        if (this.#activeAcpSessionId) {
          client.cancelPrompt(this.#activeAcpSessionId);
        }
        endpoint.cancelAllPending(reason.message);
        queue.push({ kind: 'aborted', reason });
      };
      options.signal?.addEventListener('abort', onAbort, { once: true });

      let delivered = false;
      try {
        if (syncPlan.kind === 'resolve-pending-tools' && this.#inFlightPrompt) {
          client.setStreamCallbacks(this.#activeAcpSessionId!, callbacks);
          this.#attachPromptCompletion(this.#inFlightPrompt, queue);
          let resolvedChars = 0;
          for (const res of syncPlan.resolutions) {
            const content = await buildMcpToolResultContent({
              content: res.content,
              trailingNotes: res.trailingNotes,
              loadImage: this.#loadImage,
              signal: options.signal,
            });
            for (const item of content) {
              resolvedChars += item.type === 'text' ? item.text.length : 1024;
            }
            endpoint.resolveToolCall(res.toolCallId, {
              content,
              isError: res.isError,
            });
          }
          if (this.#inFlightTurnUsage) {
            this.#inFlightTurnUsage.pendingInputDeltaTokens = Math.max(
              1,
              Math.ceil(resolvedChars / 4),
            );
          }
        } else {
          const routingBridgeToolNames =
            options.purpose !== undefined && !endpoint.hasTools
              ? undefined
              : endpoint.bridgeToolNames;

          let latestAnchorMeta: AgyReplayMetadata | undefined;
          for (let i = options.messages.length - 1; i >= 0; i--) {
            const candidate = extractReplayMetadata(options.messages[i]!);
            if (candidate) {
              latestAnchorMeta = candidate;
              break;
            }
          }
          const restoreBaselineFromAnchor = (acpSessionId: string) => {
            if (
              this.#cachedPromptBaseline === 0 &&
              latestAnchorMeta?.acpSessionId === acpSessionId &&
              typeof latestAnchorMeta.cachedPromptBaseline === 'number' &&
              latestAnchorMeta.cachedPromptBaseline > 0 &&
              typeof latestAnchorMeta.baselineUpdatedAt === 'number' &&
              now - latestAnchorMeta.baselineUpdatedAt <= IMPLICIT_CACHE_TTL_MS
            ) {
              this.#cachedPromptBaseline = latestAnchorMeta.cachedPromptBaseline;
            }
          };

          let promptParts: AcpPromptPart[] | undefined;
          if (
            (syncPlan.kind === 'warm-incremental' && this.#activeAcpSessionId) ||
            (syncPlan.kind === 'resume-incremental' && resumed && this.#activeAcpSessionId)
          ) {
            if (resumed || this.#activeRawModelId !== targetRawModelId) {
              await client.setModel(this.#activeAcpSessionId, targetRawModelId);
              this.#activeRawModelId = targetRawModelId;
            }
            restoreBaselineFromAnchor(this.#activeAcpSessionId);
            promptParts = await buildIncrementalPromptParts({
              tailMessages: syncPlan.tailMessages,
              priorSystemPrompt: syncPlan.priorSystemPrompt,
              loadImage: this.#loadImage,
              signal: options.signal,
            });
            if (resumed && options.purpose !== 'compaction') {
              const routing = formatHarnessGuidelines(undefined, routingBridgeToolNames ?? false);
              if (routing) promptParts.unshift({ type: 'text', text: routing });
            }
          } else {
            targetRawModelId = await this.#openFreshSession(client, endpoint, options, cfg);
            const split = splitMessagesForTrajectorySync(options.messages);

            if (split.hasPriorTurns && client.capabilities?.loadSession && this.#activeAcpSessionId) {
              const freshSessionId = this.#activeAcpSessionId;
              const dbPath = resolveAgyConversationDbPath(freshSessionId);
              if (fs.existsSync(dbPath)) {
                const convertedSteps = convertDshMessagesToAgySteps({
                  historyMessages: split.historyMessages,
                  explicitSystem: options.system,
                  tools: options.tools,
                  bridgeToolNames: routingBridgeToolNames ?? false,
                });
                if (convertedSteps.length > 0) {
                  try {
                    await client.closeSession(freshSessionId);
                    await waitForSqliteWalClear(dbPath);
                    hydrateAgyConversationDb({
                      dbPath,
                      sessionId: freshSessionId,
                      steps: convertedSteps,
                    });
                    await client.loadSession({
                      sessionId: freshSessionId,
                      endpoint,
                      cwd: cfg.workingDirectory,
                    });
                    await client.setModel(freshSessionId, targetRawModelId);
                    this.#activeAcpSessionId = freshSessionId;
                    this.#activeRawModelId = targetRawModelId;

                    const priorSystemPrompt = extractSystemPrompt(
                      options.messages,
                      options.system,
                      split.historyMessages.length,
                    );
                    promptParts = await buildIncrementalPromptParts({
                      tailMessages: split.tailMessages,
                      priorSystemPrompt,
                      loadImage: this.#loadImage,
                      signal: options.signal,
                    });
                    if (options.purpose !== 'compaction') {
                      const routing = formatHarnessGuidelines(
                        undefined,
                        routingBridgeToolNames ?? false,
                      );
                      if (routing) promptParts.unshift({ type: 'text', text: routing });
                    }
                  } catch {
                    targetRawModelId = await this.#openFreshSession(
                      client,
                      endpoint,
                      options,
                      cfg,
                    );
                  }
                }
              }
            }

            if (!promptParts) {
              promptParts = await buildFullReplayPromptParts({
                messages: options.messages,
                systemPrompt,
                bridgeToolNames: routingBridgeToolNames,
                loadImage: this.#loadImage,
                signal: options.signal,
              });
            }
          }

          const turnUsage = createInFlightTurnUsageState();
          turnUsage.estimatedPrevTotalTokens = this.#cachedPromptBaseline;
          turnUsage.pendingInputDeltaTokens = estimatePromptPartsTokens(promptParts);
          this.#inFlightTurnUsage = turnUsage;
          if (options.signal?.aborted) throw options.signal.reason;
          client.setStreamCallbacks(this.#activeAcpSessionId!, callbacks);
          const promptPromise = client.prompt(
            this.#activeAcpSessionId!,
            promptParts,
            cfg.requestTimeoutMs > 0 ? cfg.requestTimeoutMs : undefined,
          );
          this.#inFlightPrompt = promptPromise;
          promptPromise.then(
            () => {
              if (this.#inFlightPrompt === promptPromise) {
                this.#inFlightPrompt = undefined;
              }
            },
            () => {
              if (this.#inFlightPrompt === promptPromise) {
                this.#inFlightPrompt = undefined;
              }
            },
          );
          this.#attachPromptCompletion(promptPromise, queue);
        }

        // Stream events out as DSH StreamChunks
        let nextBlockIndex = 0;
        let stepOutputChars = 0;
        let stepThoughtChars = 0;
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
            options.system,
          );
          const meta: AgyReplayMetadata = {
            version: 1,
            dshSessionId: options.sessionId,
            acpSessionId: this.#activeAcpSessionId ?? '',
            rawModelId: targetRawModelId,
            syncedMessageCount: options.messages.length + 1,
            toolsFingerprint: toolsFp,
            historyPrefixDigest,
            ...(this.#cachedPromptBaseline > 0
              ? {
                  cachedPromptBaseline: this.#cachedPromptBaseline,
                  baselineUpdatedAt: Date.now(),
                }
              : {}),
          };
          this.#lastToolsFingerprint = toolsFp;
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
            stepThoughtChars += ev.text.length;
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
              const estOutTokens = Math.max(1, Math.ceil(stepOutputChars / 4));
              const fallbackStepUsed =
                (turnUsage.estimatedPrevTotalTokens ?? this.#cachedPromptBaseline) +
                (turnUsage.pendingInputDeltaTokens ?? 1) +
                estOutTokens;
              const hasAuthoritativeUsed =
                turnUsage.pendingStepUsed !== undefined && turnUsage.pendingStepUsed > 0;
              const stepUsed = hasAuthoritativeUsed
                ? turnUsage.pendingStepUsed!
                : fallbackStepUsed;
              turnUsage.pendingStepUsed = undefined;
              turnUsage.finalStepObservedUsed = undefined;
              turnUsage.pendingInputDeltaTokens = 0;

              if (stepUsed > 0) {
                const { usage: stepUsage, nextCachedPromptBaseline, compacted } =
                  estimateIntermediateStepUsage({
                    stepUsedTokens: stepUsed,
                    emittedOutputChars: stepOutputChars,
                    emittedThoughtChars: stepThoughtChars,
                    cachedPromptBaseline: this.#cachedPromptBaseline,
                    authoritative: hasAuthoritativeUsed,
                  });
                if (compacted) {
                  turnUsage.compactionDetected = true;
                }
                this.#cachedPromptBaseline = nextCachedPromptBaseline;
                const stepPromptTokens =
                  stepUsage.inputTokens +
                  (stepUsage.cacheReadTokens ?? 0) +
                  (stepUsage.cacheWriteTokens ?? 0);
                const visibleOutTokens = Math.max(
                  0,
                  stepUsage.outputTokens - (stepUsage.reasoningTokens ?? 0),
                );
                // Reasoning (`thought`) tokens are not re-injected verbatim as raw prompt text
                // on subsequent steps; exclude them so fallback steps never drift upward.
                turnUsage.estimatedPrevTotalTokens = stepPromptTokens + visibleOutTokens;
                turnUsage.lastStepUsed = stepUsed;
                turnUsage.emittedInput += stepUsage.inputTokens;
                turnUsage.emittedOutput += stepUsage.outputTokens;
                turnUsage.emittedCacheRead += stepUsage.cacheReadTokens ?? 0;
                turnUsage.emittedCacheWrite += stepUsage.cacheWriteTokens ?? 0;
                turnUsage.emittedReasoning += stepUsage.reasoningTokens ?? 0;
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
        if (!delivered || options.purpose === 'compaction') {
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
    let settledInTime = true;
    const inFlight = this.#inFlightPrompt;
    if (inFlight && this.#activeAcpSessionId && this.#client?.alive) {
      settledInTime = false;
      // Signal session/cancel BEFORE rejecting suspended MCP calls so `agy` exits with
      // `stopReason: 'cancelled'` instead of triggering an extra LLM retry round-trip.
      this.#client.cancelPrompt(this.#activeAcpSessionId);
      this.#endpoint?.cancelAllPending('Turn superseded');
      await Promise.race([
        inFlight.then(
          () => {
            settledInTime = true;
          },
          () => {
            settledInTime = true;
          },
        ),
        new Promise<void>((r) => {
          const timer = setTimeout(r, 5_000);
          timer.unref?.();
        }),
      ]);
    } else {
      this.#endpoint?.cancelAllPending('Turn superseded');
    }
    this.#inFlightTurnUsage = undefined;
    this.#inFlightPrompt = undefined;
    if (closeExistingSession || !settledInTime) {
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
    const usesConversationSession =
      Boolean(options.sessionId) &&
      (options.purpose === undefined || options.purpose === 'compaction');
    let key: string | symbol = usesConversationSession
      ? String(options.sessionId)
      : Symbol('request');
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
