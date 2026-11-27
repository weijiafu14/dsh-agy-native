import {
  LlmAdapter,
  type GenerateOptions,
  type LlmModelInfo,
  type LlmProviderInfo,
  type LlmResolvedModelInfo,
  type StreamChunk,
} from '@deepseek-ai/dsh-llm';
import { AgyModelCatalog } from './catalog.js';
import { LoopbackMcpBridge } from './mcp-bridge.js';
import type { ImageBytesLoader } from './session-sync.js';
import type { ResolvedAgyAcpConfig } from './types.js';
import { AgyWorkerPool } from './worker-pool.js';

/**
 * Native DeepSeek Harness `LlmAdapter` backed exclusively by the local official
 * Antigravity CLI (`agy acp --stdio`) and a loopback Streamable-HTTP Eager MCP bridge.
 */
export class AgyAcpAdapter extends LlmAdapter {
  readonly #getConfig: () => ResolvedAgyAcpConfig;
  readonly #catalog: AgyModelCatalog;
  readonly #mcpBridge: LoopbackMcpBridge;
  readonly #pool: AgyWorkerPool;
  #lastTierName?: string;
  #onCatalogError?: (error: string | undefined) => void;
  #onTierUpdated?: (tierName: string | undefined) => void;

  constructor(opts: {
    getConfig: () => ResolvedAgyAcpConfig;
    catalog?: AgyModelCatalog;
    mcpBridge?: LoopbackMcpBridge;
    loadImage?: ImageBytesLoader;
    onCatalogError?: (error: string | undefined) => void;
    onTierUpdated?: (tierName: string | undefined) => void;
  }) {
    super();
    this.#getConfig = opts.getConfig;
    const initialCfg = opts.getConfig();
    this.#catalog =
      opts.catalog ??
      new AgyModelCatalog({
        cacheDir: initialCfg.cacheDir,
        getMaxContextWindow: () => this.#getConfig().maxContextWindow,
      });
    this.#lastTierName = this.#catalog.tier?.name;
    this.#mcpBridge = opts.mcpBridge ?? new LoopbackMcpBridge();
    this.#onCatalogError = opts.onCatalogError;
    this.#onTierUpdated = opts.onTierUpdated;
    this.#pool = new AgyWorkerPool({
      getConfig: this.#getConfig,
      mcpBridge: this.#mcpBridge,
      catalog: this.#catalog,
      loadImage: opts.loadImage,
    });
  }

  get catalog(): AgyModelCatalog {
    return this.#catalog;
  }

  override providerInfo(provider: string): LlmProviderInfo {
    const cfg = this.#getConfig();
    const tierName = this.#catalog.tier?.name;
    const displayName =
      tierName && cfg.providerName === 'Antigravity (AGY ACP)'
        ? `Antigravity (${tierName})`
        : cfg.providerName;
    return {
      id: provider,
      name: displayName,
    };
  }

  override async listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    try {
      await this.#pool.ensureCatalogLoaded();
      this.#notifyTierIfChanged();
      this.#onCatalogError?.(undefined);
    } catch (err) {
      this.#onCatalogError?.(err instanceof Error ? err.message : String(err));
      if (!this.#catalog.snapshot) return [];
    }
    return this.#catalog.toLlmModelInfos(provider);
  }

  #notifyTierIfChanged(): void {
    const nextTier = this.#catalog.tier?.name;
    if (nextTier !== this.#lastTierName) {
      this.#lastTierName = nextTier;
      this.#onTierUpdated?.(nextTier);
    }
  }

  override async resolveModel(
    provider: string,
    model: string,
    signal?: AbortSignal,
  ): Promise<LlmResolvedModelInfo> {
    if (signal?.aborted) {
      throw signal.reason ?? new Error('Model resolution aborted');
    }
    if (!this.#catalog.snapshot) {
      try {
        await Promise.race([
          this.#pool.ensureCatalogLoaded(),
          ...(signal
            ? [
                new Promise<never>((_, reject) => {
                  signal.addEventListener(
                    'abort',
                    () => reject(signal.reason ?? new Error('Model resolution aborted')),
                    { once: true },
                  );
                }),
              ]
            : []),
        ]);
        this.#onCatalogError?.(undefined);
      } catch (err) {
        if (signal?.aborted) throw err;
        this.#onCatalogError?.(err instanceof Error ? err.message : String(err));
      }
    }
    return this.#catalog.toResolvedModelInfo(provider, model);
  }

  override stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    return this.#pool.stream(options);
  }

  async dispose(): Promise<void> {
    await this.#pool.dispose();
    await this.#mcpBridge.close();
  }
}
