import type { ReasoningEffortId } from '@deepseek-ai/dsh-llm';

/** Recognized reasoning effort levels in ascending order of compute effort. */
export const EFFORT_ORDER = ['low', 'medium', 'high', 'xhigh'] as const;
export type AgyEffortSlug = (typeof EFFORT_ORDER)[number];

/** Raw model entry advertised by `agy acp` in `session/new`. */
export interface AcpAvailableModel {
  modelId: string;
  name: string;
  description?: string;
}

/** Subscription tier metadata returned in `session/new._meta.antigravity.tier`. */
export interface AcpSubscriptionTier {
  id?: string;
  name?: string;
  upgradeUri?: string;
}

/** Grouped model family route exposed to DSH selectors and `resolveModel()`. */
export interface CatalogModelEntry {
  /** Base family ID exposed in DSH (e.g. `gemini-4-carbon`) or standalone ID. */
  id: string;
  /** Clean display name without effort suffix (e.g. `Gemini 4 Carbon`). */
  name: string;
  /** Description shown in model selectors. */
  description?: string;
  /** Map from effort slug (`low` | `medium` | `high` | `xhigh`) to raw `agy` modelId. */
  effortToRawModelId: ReadonlyMap<AgyEffortSlug, string>;
  /** Default effort slug when `effortToRawModelId` is non-empty. */
  defaultEffort?: AgyEffortSlug;
  /** Standalone raw modelId when the model has no effort variants. */
  standaloneRawModelId?: string;
  /** Context window in tokens. */
  contextWindow: number;
}

/** Snapshot of discovered models and account tier from `agy acp`. */
export interface CatalogSnapshot {
  updatedAt: number;
  tier?: AcpSubscriptionTier;
  rawModels: readonly AcpAvailableModel[];
  entries: readonly CatalogModelEntry[];
}

/** Content part accepted by ACP `session/prompt`. */
export type AcpPromptPart =
  | {
      type: 'text';
      text: string;
    }
  | {
      type: 'image';
      data: string;
      mimeType: string;
    };

/** Usage counters returned by ACP `session/prompt`. */
export interface AcpPromptUsage {
  inputTokens?: number;
  outputTokens?: number;
  cachedReadTokens?: number;
  cachedWriteTokens?: number;
  thoughtTokens?: number;
  totalTokens?: number;
}

/** Final response of ACP `session/prompt`. */
export interface AcpPromptResponse {
  stopReason?: string;
  usage?: AcpPromptUsage;
}

/** Suspended MCP `tools/call` invocation awaiting DSH execution. */
export interface PendingMcpToolCall {
  /** Stable DSH tool call ID correlated across assistant and tool messages. */
  toolCallId: string;
  /** Unprefixed DSH tool name (exact match in `options.tools`). */
  toolName: string;
  /** Sanitized JSON string of tool arguments. */
  argumentsJson: string;
  /** Timestamp when the MCP bridge received this invocation. */
  receivedAt: number;
  /** Resolve the suspended HTTP MCP call back to `agy`. */
  resolve: (result: { text: string; isError?: boolean }) => void;
  /** Reject/abort the suspended HTTP MCP call when cancelled or torn down. */
  cancel: (reason?: string) => void;
}

/** Adapter-private state persisted inside DSH's `ReplayEnvelope.response`. */
export interface AgyReplayMetadata {
  version: 1;
  dshSessionId?: string;
  acpSessionId: string;
  rawModelId: string;
  syncedMessageCount: number;
  lastAnchorId?: string;
  toolsFingerprint: string;
  contextDigests?: Record<string, string>;
  historyPrefixDigest?: string;
}

/** Resolved runtime configuration for `AgyAcpAdapter`. */
export interface ResolvedAgyAcpConfig {
  providerId: string;
  providerName: string;
  agyPath?: string;
  workingDirectory?: string;
  /** @deprecated Session concurrency is no longer capped by the adapter. */
  poolSize?: number;
  idleTtlMs: number;
  batchWindowMs: number;
  startupTimeoutMs: number;
  requestTimeoutMs: number;
  maxToolOutputBytes: number;
  resumeSessions: boolean;
  forwardContextUpdates: boolean;
  cacheDir?: string;
}

/** Diagnostic report produced by `runDoctor()`. */
export interface AgyDoctorReport {
  ok: boolean;
  agyPath?: string;
  agyVersion?: string;
  stateDir: string;
  stateDirWritable: boolean;
  protocolVersion?: number;
  capabilities?: {
    image: boolean;
    loadSession: boolean;
    systemPromptAppend: boolean;
  };
  tier?: AcpSubscriptionTier;
  rawModelCount: number;
  familyModels: Array<{
    id: string;
    name: string;
    efforts: ReasoningEffortId[];
    defaultEffort?: ReasoningEffortId;
  }>;
  error?: string;
}
