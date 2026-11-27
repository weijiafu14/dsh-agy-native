import z from '@deepseek-ai/schemastery';
import { isVolatile } from '@deepseek-ai/cosmokit';
import type { ResolvedAgyAcpConfig } from './types.js';

export const DEFAULT_PROVIDER_ID = 'agy-acp';
export const DEFAULT_PROVIDER_NAME = 'Antigravity (AGY ACP)';
/**
 * Cap advertised DSH `contextWindow` at 500k by default so DSH's visible compaction
 * (`thresholdRatio = 0.8` -> 400k) preempts `agy`'s silent ~500k internal `CHECKPOINT` freeze.
 */
export const DEFAULT_MAX_CONTEXT_WINDOW = 500_000;

export const Config = z.object({
  providerId: z
    .string()
    .default(DEFAULT_PROVIDER_ID)
    .description('Provider route ID registered in DeepSeek Harness (e.g. agy-acp).'),
  providerName: z
    .string()
    .default(DEFAULT_PROVIDER_NAME)
    .volatile()
    .description('Display name shown in DSH model selectors and settings.'),
  agyPath: z
    .string()
    .default('')
    .volatile()
    .description('Optional explicit path to the official `agy` binary. Leave empty for auto-discovery (~/.local/bin/agy or PATH).'),
  workingDirectory: z
    .string()
    .default('')
    .volatile()
    .description('Optional working directory passed to `agy acp` (leave empty to use an isolated temp directory).'),
  poolSize: z
    .natural()
    .min(1)
    .max(64)
    .volatile()
    .hidden()
    .description('Deprecated: sessions are multiplexed over one ACP transport without a slot cap.'),
  idleMinutes: z
    .natural()
    .min(0)
    .max(1440)
    .default(60)
    .volatile()
    .description('Minutes of inactivity before an idle ACP session is closed (0 = keep warm indefinitely).'),
  batchWindowMs: z
    .natural()
    .min(0)
    .max(500)
    .default(25)
    .volatile()
    .description('Micro-batch window (ms) to coalesce concurrent MCP tool calls into a single DSH parallel batch.'),
  startupTimeoutMs: z
    .natural()
    .min(3000)
    .max(120000)
    .default(30000)
    .volatile()
    .description('Timeout (ms) for `agy acp` startup handshake and `session/new`.'),
  requestTimeoutMs: z
    .natural()
    .min(0)
    .max(3600000)
    .default(0)
    .volatile()
    .description('Optional per-turn prompt timeout in milliseconds (0 = no adapter-imposed timeout; defer to DSH cancellation).'),
  maxToolOutputBytes: z
    .natural()
    .min(0)
    .max(67108864)
    .default(0)
    .volatile()
    .description('Optional UTF-8 byte cap per tool output over the MCP bridge (0 = unlimited; defer to DSH pagination and spill handling).'),
  maxContextWindow: z
    .natural()
    .min(0)
    .max(2097152)
    .default(DEFAULT_MAX_CONTEXT_WINDOW)
    .volatile()
    .description('Effective context window cap advertised to DSH so DSH compaction runs before `agy` hits its silent ~500k internal checkpoint (0 = uncapped raw model window).'),
  resumeSessions: z
    .boolean()
    .default(true)
    .volatile()
    .description('Attempt lossless ACP `session/load` recovery across worker restarts before falling back to structured transcript replay.'),
  forwardContextUpdates: z
    .boolean()
    .volatile()
    .hidden()
    .description('Deprecated: prefix mutations now rebuild cleanly while in-history updates append at the tail.'),
});

export type RawPluginConfig = Record<string, unknown>;

export function unwrapVolatileConfig(raw: RawPluginConfig = {}): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(raw).map(([key, value]) => [key, isVolatile(value) ? value.get() : value]),
  );
}

export function resolveAgyAcpConfig(raw: RawPluginConfig = {}): ResolvedAgyAcpConfig {
  const plain = unwrapVolatileConfig(raw);
  const providerId =
    typeof plain.providerId === 'string' && plain.providerId.trim()
      ? plain.providerId.trim()
      : DEFAULT_PROVIDER_ID;
  const providerName =
    typeof plain.providerName === 'string' && plain.providerName.trim()
      ? plain.providerName.trim()
      : DEFAULT_PROVIDER_NAME;
  const agyPath =
    typeof plain.agyPath === 'string' && plain.agyPath.trim() ? plain.agyPath.trim() : undefined;
  const workingDirectory =
    typeof plain.workingDirectory === 'string' && plain.workingDirectory.trim()
      ? plain.workingDirectory.trim()
      : undefined;

  const idleMinutes = Math.max(0, Number(plain.idleMinutes ?? 60));
  const idleTtlMs = Number.isFinite(idleMinutes) ? Math.round(idleMinutes * 60_000) : 3_600_000;
  const batchWindowMs = Math.min(500, Math.max(0, Number(plain.batchWindowMs ?? 25)));
  const startupTimeoutMs = Math.max(3000, Number(plain.startupTimeoutMs ?? 30_000) || 30_000);
  const requestTimeoutMs = Math.max(0, Number(plain.requestTimeoutMs ?? 0) || 0);
  const rawMaxBytes = Number(plain.maxToolOutputBytes ?? 0);
  const maxToolOutputBytes =
    Number.isFinite(rawMaxBytes) && rawMaxBytes > 0 ? Math.floor(rawMaxBytes) : 0;
  const rawMaxContextWindow = Number(plain.maxContextWindow ?? DEFAULT_MAX_CONTEXT_WINDOW);
  const maxContextWindow =
    Number.isFinite(rawMaxContextWindow) && rawMaxContextWindow >= 0
      ? Math.round(rawMaxContextWindow)
      : DEFAULT_MAX_CONTEXT_WINDOW;
  const resumeSessions = plain.resumeSessions !== false;
  const cacheDir = typeof plain.cacheDir === 'string' ? plain.cacheDir : undefined;

  return {
    providerId,
    providerName,
    agyPath,
    workingDirectory,
    idleTtlMs,
    batchWindowMs,
    startupTimeoutMs,
    requestTimeoutMs,
    maxToolOutputBytes,
    maxContextWindow,
    resumeSessions,
    cacheDir,
  };
}
