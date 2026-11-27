import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { ReasoningEffortId } from '@deepseek-ai/dsh-llm';
import type { LlmModelBadge, LlmModelInfo, LlmResolvedModelInfo } from '@deepseek-ai/dsh-llm';
import {
  EFFORT_ORDER,
  type AcpAvailableModel,
  type AcpSubscriptionTier,
  type AgyEffortSlug,
  type CatalogModelEntry,
  type CatalogSnapshot,
} from './types.js';

const EFFORT_LABELS: Record<AgyEffortSlug, { name: string; description: string }> = {
  minimal: { name: 'Minimal', description: 'Ultra-low latency with minimal thinking' },
  low: { name: 'Low', description: 'Fastest responses with minimal thinking budget' },
  medium: { name: 'Medium', description: 'Balanced reasoning depth and latency' },
  high: { name: 'High', description: 'Deep multi-step reasoning for complex engineering tasks' },
  extra: { name: 'Extra', description: 'Extended reasoning budget above High' },
  xhigh: { name: 'XHigh', description: 'Maximum extended thinking budget' },
  max: { name: 'Max', description: 'Unbounded maximum reasoning depth' },
};

const EFFORT_PATTERN = EFFORT_ORDER.join('|');
const EFFORT_SUFFIX_RE = new RegExp(`^(.+)-(${EFFORT_PATTERN})$`, 'i');
const PAREN_EFFORT_NAME_RE = new RegExp(`\\s*\\((${EFFORT_PATTERN})\\)\\s*$`, 'i');

/**
 * Default effective context window cap advertised to DSH so DSH's own visible compaction
 * (`0.8 × 500k = 400k`) runs cleanly before `agy` hits its silent ~500k internal `CHECKPOINT`.
 */
export const AGY_PRE_CHECKPOINT_CONTEXT_CAP = 500_000;

export function clampEffectiveContextWindow(
  rawWindow: number,
  maxContextWindow = AGY_PRE_CHECKPOINT_CONTEXT_CAP,
): number {
  const safeRaw = Math.max(1, Math.round(rawWindow));
  if (maxContextWindow > 0) {
    return Math.min(safeRaw, Math.round(maxContextWindow));
  }
  return safeRaw;
}

export function defaultContextWindowForModel(
  modelId: string,
  maxContextWindow = AGY_PRE_CHECKPOINT_CONTEXT_CAP,
): number {
  const lower = modelId.toLowerCase();
  let raw = 131_072;
  if (lower.startsWith('gemini-')) raw = 1_048_576;
  else if (lower.startsWith('claude-')) raw = 200_000;
  return clampEffectiveContextWindow(raw, maxContextWindow);
}

function chooseDefaultEffort(
  familyId: string,
  efforts: ReadonlyMap<AgyEffortSlug, string>,
  firstSeen: AgyEffortSlug,
): AgyEffortSlug {
  if (familyId.includes('hydrogen') && efforts.has('high')) return 'high';
  if (efforts.has('medium')) return 'medium';
  if (efforts.has(firstSeen)) return firstSeen;
  for (const slug of EFFORT_ORDER) {
    if (efforts.has(slug)) return slug;
  }
  return firstSeen;
}

/**
 * Group raw ACP `availableModels` into clean DSH model families with `reasoningEfforts`
 * while preserving standalone models (`gemini-pro-orchestrator`, single-effort models, etc.).
 */
export function buildCatalogEntries(
  rawModels: readonly AcpAvailableModel[],
  observedContextWindows?: ReadonlyMap<string, number>,
  maxContextWindow = AGY_PRE_CHECKPOINT_CONTEXT_CAP,
): CatalogModelEntry[] {
  interface DraftGroup {
    familyId: string;
    baseName: string;
    rawItems: AcpAvailableModel[];
    efforts: Map<AgyEffortSlug, string>;
    firstEffort?: AgyEffortSlug;
  }

  const orderedKeys: string[] = [];
  const groups = new Map<string, DraftGroup>();

  for (const raw of rawModels) {
    if (!raw.modelId) continue;
    const match = EFFORT_SUFFIX_RE.exec(raw.modelId);
    if (match) {
      const familyId = match[1]!;
      const effort = match[2]!.toLowerCase() as AgyEffortSlug;
      let group = groups.get(familyId);
      if (!group) {
        group = {
          familyId,
          baseName: (raw.name || familyId).replace(PAREN_EFFORT_NAME_RE, '').trim(),
          rawItems: [],
          efforts: new Map(),
          firstEffort: effort,
        };
        groups.set(familyId, group);
        orderedKeys.push(familyId);
      }
      group.rawItems.push(raw);
      group.efforts.set(effort, raw.modelId);
    } else {
      const key = `standalone:${raw.modelId}`;
      if (!groups.has(key)) {
        groups.set(key, {
          familyId: raw.modelId,
          baseName: raw.name || raw.modelId,
          rawItems: [raw],
          efforts: new Map(),
        });
        orderedKeys.push(key);
      }
    }
  }

  const entries: CatalogModelEntry[] = [];
  for (const key of orderedKeys) {
    const group = groups.get(key)!;
    if (group.efforts.size >= 2) {
      const sortedEfforts = new Map<AgyEffortSlug, string>();
      for (const slug of EFFORT_ORDER) {
        const rawId = group.efforts.get(slug);
        if (rawId) sortedEfforts.set(slug, rawId);
      }
      const defaultEffort = chooseDefaultEffort(
        group.familyId,
        sortedEfforts,
        group.firstEffort ?? 'medium',
      );
      const defaultRawId = sortedEfforts.get(defaultEffort) ?? group.rawItems[0]!.modelId;
      const rawContextWindow =
        observedContextWindows?.get(defaultRawId) ??
        observedContextWindows?.get(group.familyId) ??
        defaultContextWindowForModel(group.familyId, 0);
      const contextWindow = clampEffectiveContextWindow(rawContextWindow, maxContextWindow);

      entries.push({
        id: group.familyId,
        name: group.baseName,
        description: group.rawItems[0]?.description,
        effortToRawModelId: sortedEfforts,
        defaultEffort,
        contextWindow,
      });
    } else {
      const raw = group.rawItems[0]!;
      const rawContextWindow =
        observedContextWindows?.get(raw.modelId) ?? defaultContextWindowForModel(raw.modelId, 0);
      const contextWindow = clampEffectiveContextWindow(rawContextWindow, maxContextWindow);
      entries.push({
        id: raw.modelId,
        name: raw.name || raw.modelId,
        description: raw.description,
        effortToRawModelId: new Map(),
        standaloneRawModelId: raw.modelId,
        contextWindow,
      });
    }
  }

  return entries;
}

function tierBadge(tier?: AcpSubscriptionTier): LlmModelBadge | undefined {
  if (!tier?.name) return undefined;
  const lower = tier.name.toLowerCase();
  if (lower.includes('ultra')) return { text: 'Ultra', tone: 'success' };
  if (lower.includes('pro')) return { text: 'Pro', tone: 'info' };
  return { text: tier.name, tone: 'neutral' };
}

export interface AgyModelCatalogOptions {
  cacheDir?: string;
  getMaxContextWindow?: () => number;
}

/**
 * In-memory and disk-backed catalog of models discovered from official `agy acp`.
 */
export class AgyModelCatalog {
  #snapshot?: CatalogSnapshot;
  readonly #contextWindows = new Map<string, number>();
  readonly #cacheFilePath?: string;
  readonly #getMaxContextWindow: () => number;

  constructor(opts?: string | AgyModelCatalogOptions) {
    const cacheDir = typeof opts === 'string' ? opts : opts?.cacheDir;
    this.#getMaxContextWindow =
      (typeof opts === 'object' && opts?.getMaxContextWindow) ||
      (() => AGY_PRE_CHECKPOINT_CONTEXT_CAP);
    const baseDir = cacheDir ?? path.join(os.homedir(), '.dsh', 'cache');
    this.#cacheFilePath = path.join(baseDir, 'dsh-agy-native-catalog.json');
    this.#loadFromDisk();
  }

  get snapshot(): CatalogSnapshot | undefined {
    return this.#snapshot;
  }

  get tier(): AcpSubscriptionTier | undefined {
    return this.#snapshot?.tier;
  }

  recordContextWindow(modelId: string, size: number): void {
    if (!Number.isFinite(size) || size <= 0) return;
    this.#contextWindows.set(modelId, size);
    if (this.#snapshot) {
      this.#snapshot = {
        ...this.#snapshot,
        entries: buildCatalogEntries(
          this.#snapshot.rawModels,
          this.#contextWindows,
          this.#getMaxContextWindow(),
        ),
      };
    }
  }

  update(rawModels: readonly AcpAvailableModel[], tier?: AcpSubscriptionTier): CatalogSnapshot {
    const entries = buildCatalogEntries(
      rawModels,
      this.#contextWindows,
      this.#getMaxContextWindow(),
    );
    const next: CatalogSnapshot = {
      updatedAt: Date.now(),
      tier: tier ?? this.#snapshot?.tier,
      rawModels: [...rawModels],
      entries,
    };
    this.#snapshot = next;
    this.#saveToDisk(next);
    return next;
  }

  toLlmModelInfos(provider: string): LlmModelInfo[] {
    if (!this.#snapshot) return [];
    const badge = tierBadge(this.#snapshot.tier);
    return this.#snapshot.entries.map((entry) => ({
      provider,
      id: entry.id,
      name: entry.name,
      ...(entry.description ? { description: entry.description } : {}),
      inputModalities: ['text', 'image'] as const,
      ...(badge ? { badge } : {}),
    }));
  }

  /**
   * Map a DSH `(model, reasoningEffort)` pair to the exact ACP `modelId` accepted by `agy`.
   * Accepts both grouped family IDs (`gemini-4-carbon` + `high`) and raw IDs (`gemini-4-carbon-high`).
   */
  resolveRawModelId(model: string, reasoningEffort?: string): string {
    const entries = this.#snapshot?.entries ?? [];
    const family = entries.find((e) => e.id === model);
    if (family) {
      if (family.effortToRawModelId.size > 0) {
        if (reasoningEffort) {
          const matched = family.effortToRawModelId.get(reasoningEffort.toLowerCase() as AgyEffortSlug);
          if (matched) return matched;
        }
        const fallbackEffort = family.defaultEffort ?? 'medium';
        return (
          family.effortToRawModelId.get(fallbackEffort) ??
          family.effortToRawModelId.values().next().value!
        );
      }
      return family.standaloneRawModelId ?? family.id;
    }

    // Check if caller passed a raw ID directly or if a single-variant model matches base prefix
    const rawMatch = this.#snapshot?.rawModels.find((m) => m.modelId === model);
    if (rawMatch) return rawMatch.modelId;

    // Fallback synthesis when catalog hasn't been populated yet or unlisted model requested
    if (reasoningEffort && !EFFORT_SUFFIX_RE.test(model)) {
      const candidate = `${model}-${reasoningEffort.toLowerCase()}`;
      if (!this.#snapshot || this.#snapshot.rawModels.some((m) => m.modelId === candidate)) {
        return candidate;
      }
    }
    return model;
  }

  toResolvedModelInfo(provider: string, model: string): LlmResolvedModelInfo {
    const maxWin = this.#getMaxContextWindow();
    const entries = this.#snapshot?.entries ?? [];
    const family = entries.find((e) => e.id === model);
    if (family) {
      const efforts = [...family.effortToRawModelId.keys()].map((slug) => ({
        id: ReasoningEffortId(slug),
        name: EFFORT_LABELS[slug].name,
        description: EFFORT_LABELS[slug].description,
      }));
      const defaultRawId =
        (family.defaultEffort ? family.effortToRawModelId.get(family.defaultEffort) : undefined) ??
        family.standaloneRawModelId ??
        family.id;
      const rawWindow =
        this.#contextWindows.get(defaultRawId) ??
        this.#contextWindows.get(family.id) ??
        defaultContextWindowForModel(family.id, 0);
      const contextWindow = clampEffectiveContextWindow(rawWindow, maxWin);
      return {
        provider,
        id: family.id,
        name: family.name,
        ...(family.description ? { description: family.description } : {}),
        inputModalities: ['text', 'image'] as const,
        systemPromptUpdate: 'in-history' as const,
        context: { contextWindow },
        ...(efforts.length > 0
          ? {
              reasoning: {
                efforts,
                ...(family.defaultEffort
                  ? { defaultEffort: ReasoningEffortId(family.defaultEffort) }
                  : {}),
              },
            }
          : {}),
      };
    }

    // Raw model ID passed directly (e.g. `gemini-4-carbon-medium` in subagent config)
    const raw = this.#snapshot?.rawModels.find((m) => m.modelId === model);
    const rawWindow = this.#contextWindows.get(model) ?? defaultContextWindowForModel(model, 0);
    const contextWindow = clampEffectiveContextWindow(rawWindow, maxWin);
    return {
      provider,
      id: model,
      name: raw?.name ?? model,
      ...(raw?.description ? { description: raw.description } : {}),
      inputModalities: ['text', 'image'] as const,
      systemPromptUpdate: 'in-history' as const,
      context: { contextWindow },
    };
  }

  #loadFromDisk(): void {
    if (!this.#cacheFilePath) return;
    try {
      if (!fs.existsSync(this.#cacheFilePath)) return;
      const parsed = JSON.parse(fs.readFileSync(this.#cacheFilePath, 'utf8')) as {
        updatedAt?: number;
        tier?: AcpSubscriptionTier;
        rawModels?: AcpAvailableModel[];
      };
      if (!Array.isArray(parsed.rawModels) || parsed.rawModels.length === 0) return;
      this.#snapshot = {
        updatedAt: parsed.updatedAt ?? Date.now(),
        tier: parsed.tier,
        rawModels: parsed.rawModels,
        entries: buildCatalogEntries(
          parsed.rawModels,
          this.#contextWindows,
          this.#getMaxContextWindow(),
        ),
      };
    } catch {
      // Ignore corrupt cache files
    }
  }

  #saveToDisk(snapshot: CatalogSnapshot): void {
    if (!this.#cacheFilePath) return;
    try {
      const dir = path.dirname(this.#cacheFilePath);
      fs.mkdirSync(dir, { recursive: true });
      const tmpPath = `${this.#cacheFilePath}.${process.pid}.tmp`;
      fs.writeFileSync(
        tmpPath,
        JSON.stringify(
          {
            updatedAt: snapshot.updatedAt,
            tier: snapshot.tier,
            rawModels: snapshot.rawModels,
          },
          null,
          2,
        ),
        'utf8',
      );
      fs.renameSync(tmpPath, this.#cacheFilePath);
    } catch {
      // Cache persistence is advisory; ignore write errors
    }
  }
}
