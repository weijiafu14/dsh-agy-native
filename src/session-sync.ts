import crypto from 'node:crypto';
import type { ImageAttachmentRef } from '@deepseek-ai/dsh-attachment';
import type {
  AssistantMessage,
  ContentBlock,
  Message,
  RequestMessage,
  ToolResultMessage,
} from '@deepseek-ai/dsh-llm';
import type {
  AcpPromptPart,
  AgyReplayMetadata,
  McpToolResultContentItem,
} from './types.js';

export type ImageBytesLoader = (
  ref: ImageAttachmentRef,
  signal?: AbortSignal,
) => Promise<{ base64: string; mimeType: string } | undefined>;

function isDurableMessage(msg: RequestMessage): msg is Message {
  return typeof (msg as { id?: unknown }).id === 'string';
}

export function extractReplayMetadata(msg: RequestMessage): AgyReplayMetadata | undefined {
  if (msg.role !== 'assistant' || !isDurableMessage(msg)) return undefined;
  const replay = (msg as AssistantMessage).source?.replayState as
    | { response?: unknown }
    | undefined;
  const raw = replay?.response;
  if (!raw || typeof raw !== 'object') return undefined;
  const candidate = raw as Partial<AgyReplayMetadata>;
  if (candidate.version !== 1 || typeof candidate.acpSessionId !== 'string') return undefined;
  return candidate as AgyReplayMetadata;
}

const THIRD_PARTY_PERSONA_LINE_RE =
  /^You are an? (?:AI|coding) (?:agent|assistant) powered by [^\n]+\.?\s*$/gim;

/** Strip third-party persona headers (including model-name banners that change on model switch). */
export function cleanSystemPromptText(rawSystemPrompt: string | undefined): string {
  return (rawSystemPrompt ?? '')
    .replace(THIRD_PARTY_PERSONA_LINE_RE, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function systemMessageText(msg: RequestMessage): string {
  return msg.content
    .filter((b): b is Extract<ContentBlock, { type: 'text' }> => b.type === 'text')
    .map((b) => b.text)
    .join('\n')
    .trim();
}

/**
 * Extract effective system prompt from `options.system` and `role: 'system'` messages.
 * Under `systemPromptUpdate: 'in-history'`, the latest `system` node holds the effective prompt
 * (and an empty latest node after cleared nodes means no active prompt).
 */
export function extractSystemPrompt(
  messages: readonly RequestMessage[],
  explicitSystem?: string,
  endExclusive = messages.length,
): string | undefined {
  const parts: string[] = [];
  if (explicitSystem?.trim()) {
    parts.push(explicitSystem.trim());
  }
  const limit = Math.max(0, Math.min(endExclusive, messages.length));
  let latestSystemText: string | undefined;
  for (let i = 0; i < limit; i++) {
    const msg = messages[i]!;
    if (msg.role !== 'system') continue;
    latestSystemText = systemMessageText(msg);
  }
  if (latestSystemText) {
    parts.push(latestSystemText);
  }
  return parts.length > 0 ? parts.join('\n\n') : undefined;
}

/**
 * Neutralize third-party agent persona headers and format DSH operational guidelines
 * for inclusion in the initial user turn (keeping `agy`'s `systemInstruction` 100% untouched).
 */
export function formatHarnessGuidelines(
  rawSystemPrompt: string | undefined,
  bridgeToolNames: readonly string[] | boolean,
): string | undefined {
  const cleaned = cleanSystemPromptText(rawSystemPrompt);

  const sections: string[] = [];
  if (bridgeToolNames !== false) {
    const names = bridgeToolNames === true ? undefined : bridgeToolNames;
    const routing = [
      'Use only tools mounted by the DSH MCP server for this request. Do not call tools on ambient MCP servers, even when their names match.',
    ];
    if (names?.length === 0) {
      routing.push(
        'No DSH tools are available for this request. Do not generate Antigravity Artifacts under `brain/`; respond directly.',
      );
    } else if (names) {
      routing.push(
        `Direct calls available in this request: ${names.map((name) => `mcp_dsh_${name}`).join(', ')}.`,
      );
      routing.push(
        'If a listed tool runs a program with a DSH SDK, invoke that entry point directly. Inside the program, use the exact DSH tool names in the supplied SDK without the outer mcp_dsh_ prefix.',
      );
      routing.push(
        'Do not generate Antigravity Artifacts under `brain/`; respond directly or use the `mcp_dsh_*` tools.',
      );
    } else {
      routing.push(
        'Invoke the advertised mcp_dsh_<tool_name> directly. Do not generate Antigravity Artifacts under `brain/`; respond directly or use the `mcp_dsh_*` tools.',
      );
    }
    sections.push(routing.join(' '));
  }
  if (cleaned) {
    sections.push(cleaned);
  }
  if (sections.length === 0) return undefined;
  return `<workspace_guidelines>\n${sections.join('\n\n')}\n</workspace_guidelines>`;
}

const AMBIENT_SOURCE_KINDS = new Set([
  'runtime-context',
  'time-context',
  'skill-catalog',
  'agent-instructions',
  'user-approval',
  'model-selection',
  'working-directory',
  'plan-mode',
  'ptc-mode',
]);

/**
 * Identify passive ambient context snapshots injected alongside a turn (using DSH's structured
 * `source` metadata rather than matching English prose) so they are ordered before the active request.
 * Active non-human triggers (`tool-jobs`, `subagent-settled`, `agent-message`, `schedule`, `goal`,
 * `webhook`, etc.) return `false` and remain primary turn inputs.
 */
export function isAmbientContextMessage(msg: RequestMessage): boolean {
  if (msg.role !== 'user') return false;
  if (msg.content.some((b) => b.type === 'image' || b.type === 'file')) return false;
  const src = (msg as { source?: { kind?: unknown; form?: unknown; slot?: unknown } }).source;
  if (src && typeof src === 'object' && typeof src.kind === 'string') {
    if (AMBIENT_SOURCE_KINDS.has(src.kind)) return true;
    if (src.form === 'snapshot' || src.form === 'catalog' || src.form === 'instructions') {
      return true;
    }
    if (src.kind === 'context-composition' && src.slot === 'background') {
      return true;
    }
    return false;
  }
  const trimmed = blocksToPlainText(msg.content).trimStart();
  return trimmed.startsWith('<system-reminder>');
}

/** Serialize non-image content blocks of a message into deterministic text for digesting or replay. */
export function blocksToPlainText(blocks: readonly ContentBlock[]): string {
  const pieces: string[] = [];
  for (const block of blocks) {
    switch (block.type) {
      case 'text':
        if (block.text) pieces.push(block.text);
        break;
      case 'image':
        pieces.push(
          `[Image ${block.offloaded ? 'offloaded' : 'attached'}: ${block.attachment.name ?? block.attachment.attachmentId} (${block.attachment.width}x${block.attachment.height})]`,
        );
        break;
      case 'file':
        pieces.push(
          `[File attached: ${block.attachment.name} (${block.attachment.bytes} bytes)]`,
        );
        break;
      case 'tool-call':
        pieces.push(
          `<tool_call id="${block.id}" name="${block.name}">${block.arguments}</tool_call>`,
        );
        break;
      case 'tool-addition':
        pieces.push(`[Tool activated: ${block.toolName}]`);
        break;
      case 'tool-removal':
        pieces.push(`[Tool deactivated: ${block.toolName}]`);
        break;
      case 'reasoning':
        break;
    }
  }
  return pieces.join('\n');
}

function appendHashedField(hash: crypto.Hash, tag: string, value: string): void {
  hash.update(tag);
  hash.update('\0');
  hash.update(String(Buffer.byteLength(value, 'utf8')));
  hash.update('\0');
  hash.update(value);
  hash.update('\0');
}

/**
 * Compute a deterministic digest over the entire prefix `[0, endExclusive)` (including cleaned
 * `system` instructions, `developer` context, `user` messages/snapshots, `assistant` turns, and
 * `tool` results).
 *
 * Stripping third-party model-name persona lines from `system` ensures hot model switching keeps
 * an identical prefix digest, while any genuine prefix mutation (compaction, tool-result pruning,
 * image offload, rewind, or in-place instruction replacement) deterministically triggers a clean rebuild.
 */
export function computeHistoryPrefixDigest(
  messages: readonly RequestMessage[],
  endExclusive = messages.length,
  explicitSystem?: string,
): string {
  const limit = Math.max(0, Math.min(endExclusive, messages.length));
  const hash = crypto.createHash('sha256');

  const cleanedExplicit = cleanSystemPromptText(explicitSystem);
  if (cleanedExplicit) {
    appendHashedField(hash, 'explicit-system', cleanedExplicit);
  }

  for (let i = 0; i < limit; i++) {
    const msg = messages[i]!;

    if (msg.role === 'system') {
      const cleaned = cleanSystemPromptText(systemMessageText(msg));
      if (cleaned) {
        appendHashedField(hash, 'system', cleaned);
      }
      continue;
    }

    if (msg.role === 'developer') {
      const plain = blocksToPlainText(msg.content).trim();
      if (plain) {
        appendHashedField(hash, 'developer', plain);
      }
      continue;
    }

    if (msg.role === 'user') {
      const plain = blocksToPlainText(msg.content).trim();
      if (!plain) continue;
      appendHashedField(hash, 'user', plain);
      continue;
    }

    if (msg.role === 'assistant') {
      const plain = blocksToPlainText(msg.content);
      appendHashedField(hash, 'assistant', plain);
      continue;
    }

    if (msg.role === 'tool') {
      const toolMsg = msg as ToolResultMessage;
      const plain = blocksToPlainText(toolMsg.content);
      const id = String(toolMsg.toolCallId ?? '');
      const status = toolMsg.isError ? '1' : '0';
      hash.update('tool\0');
      hash.update(String(Buffer.byteLength(id, 'utf8')));
      hash.update('\0');
      hash.update(id);
      hash.update('\0');
      hash.update(status);
      hash.update('\0');
      hash.update(String(Buffer.byteLength(plain, 'utf8')));
      hash.update('\0');
      hash.update(plain);
      hash.update('\0');
    }
  }

  return hash.digest('hex').slice(0, 24);
}

function hasMutatedHistoryPrefix(
  messages: readonly RequestMessage[],
  endExclusive: number,
  expectedPrefixDigest?: string,
  anchorPrefixDigest?: string,
  explicitSystem?: string,
): boolean {
  if (!expectedPrefixDigest && !anchorPrefixDigest) return false;
  const currentPrefixDigest = computeHistoryPrefixDigest(messages, endExclusive, explicitSystem);
  if (expectedPrefixDigest && currentPrefixDigest !== expectedPrefixDigest) return true;
  if (anchorPrefixDigest && currentPrefixDigest !== anchorPrefixDigest) return true;
  return false;
}

function appendTextPart(parts: AcpPromptPart[], text: string): void {
  if (!text) return;
  const last = parts[parts.length - 1];
  if (last && last.type === 'text') {
    last.text = `${last.text}\n${text}`;
  } else {
    parts.push({ type: 'text', text });
  }
}

async function appendBlocksWithImages(
  parts: AcpPromptPart[],
  blocks: readonly ContentBlock[],
  loadImage?: ImageBytesLoader,
  signal?: AbortSignal,
): Promise<void> {
  for (const block of blocks) {
    switch (block.type) {
      case 'text':
        if (block.text) appendTextPart(parts, block.text);
        break;
      case 'image': {
        if (block.offloaded || !loadImage) {
          appendTextPart(
            parts,
            `[Image ${block.offloaded ? 'offloaded' : 'reference'}: ${block.attachment.name ?? block.attachment.attachmentId} (${block.attachment.width}x${block.attachment.height})]`,
          );
          break;
        }
        const loaded = await loadImage(block.attachment, signal);
        if (loaded) {
          parts.push({
            type: 'image',
            data: loaded.base64,
            mimeType: loaded.mimeType,
          });
        } else {
          appendTextPart(
            parts,
            `[Image unavailable: ${block.attachment.name ?? block.attachment.attachmentId}]`,
          );
        }
        break;
      }
      case 'file':
        appendTextPart(
          parts,
          `[File: ${block.attachment.name} (${block.attachment.bytes} bytes)]`,
        );
        break;
      case 'tool-call':
        appendTextPart(
          parts,
          `<tool_call id="${block.id}" name="${block.name}">\n${block.arguments}\n</tool_call>`,
        );
        break;
      case 'tool-addition':
        appendTextPart(parts, `[Tool enabled: ${block.toolName}]`);
        break;
      case 'tool-removal':
        appendTextPart(parts, `[Tool disabled: ${block.toolName}]`);
        break;
      case 'reasoning':
        break;
    }
  }
}

/**
 * Convert a DSH `ToolResultMessage`'s blocks (including live images from e.g. `read_image`)
 * plus any mid-turn trailing messages into structured MCP `tools/call` content blocks.
 * Trailing notes are emitted in a separate content item so the tool output itself is never mutated.
 */
export async function buildMcpToolResultContent(opts: {
  content: readonly ContentBlock[];
  trailingNotes?: readonly string[];
  loadImage?: ImageBytesLoader;
  signal?: AbortSignal;
}): Promise<McpToolResultContentItem[]> {
  const items: McpToolResultContentItem[] = [];

  for (const block of opts.content) {
    if (block.type === 'text') {
      const last = items[items.length - 1];
      if (last && last.type === 'text') {
        last.text = block.text ? (last.text ? `${last.text}\n${block.text}` : block.text) : last.text;
      } else {
        items.push({ type: 'text', text: block.text });
      }
      continue;
    }

    if (block.type === 'image') {
      if (!block.offloaded && opts.loadImage) {
        const loaded = await opts.loadImage(block.attachment, opts.signal);
        if (loaded) {
          items.push({
            type: 'image',
            data: loaded.base64,
            mimeType: loaded.mimeType,
          });
          continue;
        }
      }
      const fallback = `[Image ${block.offloaded ? 'offloaded' : 'reference'}: ${block.attachment.name ?? block.attachment.attachmentId} (${block.attachment.width}x${block.attachment.height})]`;
      items.push({ type: 'text', text: fallback });
      continue;
    }

    const serialized = blocksToPlainText([block]);
    if (serialized) {
      items.push({ type: 'text', text: serialized });
    }
  }

  if (items.length === 0) {
    items.push({ type: 'text', text: '' });
  }

  if (opts.trailingNotes && opts.trailingNotes.length > 0) {
    items.push({
      type: 'text',
      text: `<dsh_context_note>\n${opts.trailingNotes.join('\n\n')}\n</dsh_context_note>`,
    });
  }

  return items;
}

/** Convert incremental tail messages into ACP prompt parts, placing ambient snapshots before primary inputs. */
export async function buildIncrementalPromptParts(opts: {
  tailMessages: readonly RequestMessage[];
  priorSystemPrompt?: string;
  loadImage?: ImageBytesLoader;
  signal?: AbortSignal;
}): Promise<AcpPromptPart[]> {
  const parts: AcpPromptPart[] = [];
  let effectiveCleanedSystem = cleanSystemPromptText(opts.priorSystemPrompt);
  let latestTailSystem: string | undefined;

  const primaryUserMessages: RequestMessage[] = [];
  const ambientSnapshots: string[] = [];

  for (const msg of opts.tailMessages) {
    if (msg.role === 'system') {
      latestTailSystem = cleanSystemPromptText(systemMessageText(msg));
      continue;
    }
    if (msg.role === 'developer') {
      const devText = blocksToPlainText(msg.content).trim();
      if (devText) {
        appendTextPart(parts, `<developer_context>\n${devText}\n</developer_context>`);
      }
      continue;
    }
    if (msg.role === 'user') {
      if (isAmbientContextMessage(msg)) {
        const plain = blocksToPlainText(msg.content).trim();
        if (plain) ambientSnapshots.push(plain);
      } else {
        primaryUserMessages.push(msg);
      }
      continue;
    }
    if (msg.role === 'assistant') {
      const asstText = blocksToPlainText(msg.content).trim();
      if (asstText) {
        appendTextPart(parts, `<prior_assistant_turn>\n${asstText}\n</prior_assistant_turn>`);
      }
      continue;
    }
    if (msg.role === 'tool') {
      const toolMsg = msg as ToolResultMessage;
      const out = blocksToPlainText(toolMsg.content);
      appendTextPart(
        parts,
        `<tool_result id="${toolMsg.toolCallId}" status="${toolMsg.isError ? 'error' : 'ok'}">\n${out}\n</tool_result>`,
      );
    }
  }

  if (latestTailSystem !== undefined && latestTailSystem !== effectiveCleanedSystem) {
    effectiveCleanedSystem = latestTailSystem;
    if (latestTailSystem) {
      appendTextPart(
        parts,
        `<workspace_guidelines_update>\n${latestTailSystem}\n</workspace_guidelines_update>`,
      );
    }
  }

  if (ambientSnapshots.length > 0) {
    appendTextPart(parts, `<runtime_context>\n${ambientSnapshots.join('\n\n')}\n</runtime_context>`);
  }

  for (const userMsg of primaryUserMessages) {
    if (primaryUserMessages.length > 1) appendTextPart(parts, '<user_message>');
    await appendBlocksWithImages(parts, userMsg.content, opts.loadImage, opts.signal);
    if (primaryUserMessages.length > 1) appendTextPart(parts, '</user_message>');
  }

  if (parts.length === 0) {
    parts.push({ type: 'text', text: 'Continue.' });
  }
  return parts;
}

/**
 * Replay completed history as text and send current inputs with their live attachments.
 */
export async function buildFullReplayPromptParts(opts: {
  messages: readonly RequestMessage[];
  systemPrompt?: string;
  /** Legacy callers may supply presence only. */
  hasTools?: boolean;
  bridgeToolNames?: readonly string[];
  loadImage?: ImageBytesLoader;
  signal?: AbortSignal;
}): Promise<AcpPromptPart[]> {
  const parts: AcpPromptPart[] = [];
  const effectiveSystem =
    opts.systemPrompt ?? extractSystemPrompt(opts.messages);
  const guidelines = formatHarnessGuidelines(
    effectiveSystem,
    opts.bridgeToolNames ?? Boolean(opts.hasTools),
  );
  if (guidelines) {
    appendTextPart(parts, guidelines);
  }

  const nonSystem = opts.messages.filter((m) => m.role !== 'system');
  if (nonSystem.length === 0) {
    if (parts.length === 0) parts.push({ type: 'text', text: 'Hello.' });
    return parts;
  }

  let splitIdx = nonSystem.length;
  while (splitIdx > 0) {
    const role = nonSystem[splitIdx - 1]!.role;
    if (role === 'assistant' || role === 'tool') break;
    splitIdx--;
  }

  if (splitIdx > 0) {
    appendTextPart(
      parts,
      'Prior conversation history (treat prior tool calls and results as already completed; address the latest request at the end):\n<conversation_history>',
    );
    for (let i = 0; i < splitIdx; i++) {
      const msg = nonSystem[i]!;
      if (msg.role === 'user') {
        const plain = blocksToPlainText(msg.content).trim();
        if (!plain) continue;
        if (isAmbientContextMessage(msg)) {
          appendTextPart(parts, `<runtime_context>\n${plain}\n</runtime_context>`);
        } else {
          appendTextPart(parts, '<user_turn>');
          appendTextPart(parts, plain);
          appendTextPart(parts, '</user_turn>');
        }
      } else if (msg.role === 'developer') {
        const text = blocksToPlainText(msg.content).trim();
        if (text) appendTextPart(parts, `<developer_context>\n${text}\n</developer_context>`);
      } else if (msg.role === 'assistant') {
        const text = blocksToPlainText(msg.content).trim();
        if (text) appendTextPart(parts, `<assistant_turn>\n${text}\n</assistant_turn>`);
      } else if (msg.role === 'tool') {
        const toolMsg = msg as ToolResultMessage;
        const text = blocksToPlainText(toolMsg.content);
        appendTextPart(
          parts,
          `<tool_result id="${toolMsg.toolCallId}" status="${toolMsg.isError ? 'error' : 'ok'}">\n${text}\n</tool_result>`,
        );
      }
    }
    appendTextPart(parts, '</conversation_history>');
  }

  appendTextPart(parts, '<current_input>');
  const current = await buildIncrementalPromptParts({
    tailMessages: nonSystem.slice(splitIdx),
    priorSystemPrompt: effectiveSystem,
    loadImage: opts.loadImage,
    signal: opts.signal,
  });
  parts.push(...current);
  appendTextPart(parts, '</current_input>');
  return parts;
}

export interface PendingToolResolution {
  toolCallId: string;
  content: readonly ContentBlock[];
  text: string;
  isError?: boolean;
  trailingNotes?: readonly string[];
}

export type SessionSyncDecision =
  | {
      kind: 'resolve-pending-tools';
      resolutions: PendingToolResolution[];
    }
  | {
      kind: 'warm-incremental';
      tailMessages: RequestMessage[];
      priorSystemPrompt?: string;
    }
  | {
      kind: 'resume-incremental';
      resumeAcpSessionId: string;
      tailMessages: RequestMessage[];
      priorSystemPrompt?: string;
    }
  | {
      kind: 'cold-full-replay';
      reason: string;
    };

/**
 * Deterministic 3-way synchronization state machine between DSH's `options.messages` and an ACP session:
 * 1. `resolve-pending-tools`: settle dispatched MCP tool calls with multimodal content + trailing notes.
 * 2. `warm-incremental` / `resume-incremental`: prefix & toolset unchanged -> send only tail additions.
 * 3. `cold-full-replay`: prefix mutated/compacted or toolset changed -> open fresh session and replay.
 */
export function planSessionSync(opts: {
  messages: readonly RequestMessage[];
  explicitSystem?: string;
  activeAcpSessionId?: string;
  lastSyncedAnchorId?: string;
  lastToolsFingerprint?: string;
  currentToolsFingerprint: string;
  lastHistoryPrefixDigest?: string;
  pendingToolCallIds: ReadonlySet<string>;
  allowResumeFromReplay: boolean;
  dshSessionId?: string;
  purpose?: string;
}): SessionSyncDecision {
  const { messages, pendingToolCallIds } = opts;

  // Compaction (`purpose: 'compaction'`): reuse the existing warm (or resumable) ACP session with
  // its full toolset intact so the entire conversation prefix hits Google's KV cache and only the
  // trailing compaction directive is sent incrementally (avoiding cold-replay single-message limits).
  // Even when triggered mid-turn with suspended `pendingToolCallIds`, cancelling the in-flight prompt
  // on the same warm ACP session preserves the exact Conversation ID and full prefix KV cache.
  if (
    opts.purpose === 'compaction' &&
    messages.length >= 2 &&
    messages[messages.length - 1]!.role === 'user'
  ) {
    const directiveMsg = messages[messages.length - 1]!;
    const directiveText = blocksToPlainText(directiveMsg.content).trim();
    const cutoffMsg = messages[messages.length - 2]!;
    const cutoffPlain =
      cutoffMsg.role !== 'system' ? blocksToPlainText(cutoffMsg.content).trim() : '';
    const cutoffSnippet =
      cutoffPlain.length > 200 ? `...${cutoffPlain.slice(-200)}` : cutoffPlain;

    let compAnchorIdx = -1;
    let compAnchorReplay: AgyReplayMetadata | undefined;
    for (let i = messages.length - 2; i >= 0; i--) {
      const meta = extractReplayMetadata(messages[i]!);
      if (meta) {
        if (!opts.activeAcpSessionId || meta.acpSessionId === opts.activeAcpSessionId) {
          compAnchorIdx = i;
          compAnchorReplay = meta;
          break;
        }
      }
    }

    const priorSystemPrompt = extractSystemPrompt(
      messages,
      opts.explicitSystem,
      messages.length - 1,
    );

    if (opts.activeAcpSessionId) {
      const effectiveToolsFp = opts.lastToolsFingerprint ?? compAnchorReplay?.toolsFingerprint;
      const toolsMatch = !effectiveToolsFp || effectiveToolsFp === opts.currentToolsFingerprint;
      const prefixIntact =
        compAnchorIdx < 0 ||
        !compAnchorReplay ||
        !hasMutatedHistoryPrefix(
          messages,
          compAnchorIdx,
          compAnchorReplay.historyPrefixDigest,
          compAnchorReplay.historyPrefixDigest,
          opts.explicitSystem,
        );
      if (toolsMatch && prefixIntact) {
        const hasPendingToolCalls = pendingToolCallIds.size > 0;
        const spansPendingToolResults =
          hasPendingToolCalls &&
          messages
            .slice(0, messages.length - 1)
            .some(
              (m) =>
                m.role === 'tool' &&
                pendingToolCallIds.has((m as ToolResultMessage).toolCallId),
            );
        const compactionSpanDigest = computeHistoryPrefixDigest(
          messages,
          messages.length - 1,
          opts.explicitSystem,
        );
        const hasRetainedTailInSession =
          !spansPendingToolResults &&
          Boolean(
            hasPendingToolCalls ||
              (opts.lastHistoryPrefixDigest &&
                opts.lastHistoryPrefixDigest !== compactionSpanDigest),
          );
        const excludedClause = hasPendingToolCalls
          ? 'Exclude any subsequent messages or interrupted tool calls in this session, as newer messages are retained verbatim after the checkpoint.'
          : 'Exclude any subsequent messages in this session, as newer messages are retained verbatim after the checkpoint.';
        const scopePrefix =
          hasRetainedTailInSession && cutoffSnippet
            ? `[Compaction boundary: Condense ONLY the earlier conversation span up through the ${cutoffMsg.role} message ending with ${JSON.stringify(cutoffSnippet)}. ${excludedClause}]\n\n`
            : '';
        const compactionUserMsg: RequestMessage = {
          role: 'user',
          content: [
            {
              type: 'text',
              text: `${scopePrefix}${directiveText}\n\n(Important: Do not invoke any mcp_dsh_* tools; output ONLY the requested Markdown checkpoint text.)`,
            },
          ],
        };
        const unsyncedBeforeDirective =
          compAnchorIdx >= 0 &&
          (spansPendingToolResults ||
            (!hasRetainedTailInSession &&
              !messages[compAnchorIdx]!.content.some((b) => b.type === 'tool-call')))
            ? messages.slice(compAnchorIdx + 1, messages.length - 1)
            : [];
        return {
          kind: 'warm-incremental',
          tailMessages: [...unsyncedBeforeDirective, compactionUserMsg],
          priorSystemPrompt,
        };
      }
    } else if (
      opts.allowResumeFromReplay &&
      compAnchorIdx >= 0 &&
      compAnchorReplay?.acpSessionId &&
      (!compAnchorReplay.dshSessionId || compAnchorReplay.dshSessionId === opts.dshSessionId) &&
      (!compAnchorReplay.toolsFingerprint ||
        compAnchorReplay.toolsFingerprint === opts.currentToolsFingerprint) &&
      !hasMutatedHistoryPrefix(
        messages,
        compAnchorIdx,
        compAnchorReplay.historyPrefixDigest,
        compAnchorReplay.historyPrefixDigest,
        opts.explicitSystem,
      )
    ) {
      const scopePrefix = cutoffSnippet
        ? `[Compaction boundary: Condense ONLY the earlier conversation span up through the ${cutoffMsg.role} message ending with ${JSON.stringify(cutoffSnippet)}. Exclude any subsequent messages in this session, as newer messages are retained verbatim after the checkpoint.]\n\n`
        : '';
      const compactionUserMsg: RequestMessage = {
        role: 'user',
        content: [
          {
            type: 'text',
            text: `${scopePrefix}${directiveText}\n\n(Important: Do not invoke any mcp_dsh_* tools; output ONLY the requested Markdown checkpoint text.)`,
          },
        ],
      };
      return {
        kind: 'resume-incremental',
        resumeAcpSessionId: compAnchorReplay.acpSessionId,
        tailMessages: [compactionUserMsg],
        priorSystemPrompt,
      };
    }
  }

  // 1. Mid-turn suspended MCP tool calls: match dispatched ToolResultMessages by toolCallId
  if (pendingToolCallIds.size > 0 && opts.activeAcpSessionId) {
    const resultByCallId = new Map<string, ToolResultMessage>();
    const trailingNotes: string[] = [];

    // Scan from the last assistant message carrying any of the dispatched pending toolCallIds
    let toolTurnStartIdx = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i]!;
      if (
        msg.role === 'assistant' &&
        msg.content.some((b) => b.type === 'tool-call' && pendingToolCallIds.has(b.id))
      ) {
        toolTurnStartIdx = i;
        break;
      }
    }

    if (toolTurnStartIdx >= 0) {
      for (let i = toolTurnStartIdx + 1; i < messages.length; i++) {
        const msg = messages[i]!;
        if (msg.role === 'tool' && pendingToolCallIds.has(msg.toolCallId)) {
          resultByCallId.set(msg.toolCallId, msg);
        } else if (msg.role === 'developer' || msg.role === 'user') {
          const text = blocksToPlainText(msg.content).trim();
          if (text) trailingNotes.push(text);
        } else if (msg.role === 'system') {
          const priorSys = cleanSystemPromptText(
            extractSystemPrompt(messages, opts.explicitSystem, toolTurnStartIdx),
          );
          const nextSys = cleanSystemPromptText(systemMessageText(msg));
          if (nextSys && nextSys !== priorSys) {
            trailingNotes.push(nextSys);
          }
        }
      }

      if (resultByCallId.size > 0) {
        const toolTurnReplay = extractReplayMetadata(messages[toolTurnStartIdx]!);
        if (
          hasMutatedHistoryPrefix(
            messages,
            toolTurnStartIdx,
            opts.lastHistoryPrefixDigest ?? toolTurnReplay?.historyPrefixDigest,
            toolTurnReplay?.historyPrefixDigest,
            opts.explicitSystem,
          )
        ) {
          return { kind: 'cold-full-replay', reason: 'history-prefix-mutated-or-compacted' };
        }

        const readyIds = [...pendingToolCallIds].filter((id) => resultByCallId.has(id));
        const resolutions: PendingToolResolution[] = readyIds.map((id, idx) => {
          const toolMsg = resultByCallId.get(id)!;
          const isLast = idx === readyIds.length - 1;
          return {
            toolCallId: id,
            content: toolMsg.content,
            text: blocksToPlainText(toolMsg.content),
            isError: toolMsg.isError,
            ...(isLast && trailingNotes.length > 0 ? { trailingNotes } : {}),
          };
        });
        return { kind: 'resolve-pending-tools', resolutions };
      }
    }
    // If DSH aborted/rewound past the pending tool calls, fall through to cold replay
    return { kind: 'cold-full-replay', reason: 'pending-tool-mismatch-or-rewind' };
  }

  // Locate the latest assistant message synced with an ACP session
  let anchorIdx = -1;
  let anchorReplay: AgyReplayMetadata | undefined;

  if (opts.activeAcpSessionId && opts.lastSyncedAnchorId) {
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i]!;
      if (isDurableMessage(msg) && msg.id === opts.lastSyncedAnchorId) {
        anchorIdx = i;
        anchorReplay = extractReplayMetadata(msg);
        break;
      }
    }
  }

  if (anchorIdx < 0) {
    for (let i = messages.length - 1; i >= 0; i--) {
      const meta = extractReplayMetadata(messages[i]!);
      if (meta) {
        if (!opts.activeAcpSessionId || meta.acpSessionId === opts.activeAcpSessionId) {
          anchorIdx = i;
          anchorReplay = meta;
          break;
        }
      }
    }
  }

  if (anchorIdx < 0) {
    return { kind: 'cold-full-replay', reason: 'no-prior-synced-anchor' };
  }

  const isWarmSession = Boolean(
    opts.activeAcpSessionId &&
      (!anchorReplay || anchorReplay.acpSessionId === opts.activeAcpSessionId),
  );

  const effectiveToolsFp = isWarmSession
    ? (opts.lastToolsFingerprint ?? anchorReplay?.toolsFingerprint)
    : anchorReplay?.toolsFingerprint;
  const toolsChanged = Boolean(
    effectiveToolsFp && effectiveToolsFp !== opts.currentToolsFingerprint,
  );

  const tailMessages = messages.slice(anchorIdx + 1);
  if (tailMessages.length === 0) {
    return { kind: 'cold-full-replay', reason: 'empty-incremental-tail' };
  }

  const expectedPrefixDigest = isWarmSession
    ? (opts.lastHistoryPrefixDigest ?? anchorReplay?.historyPrefixDigest)
    : anchorReplay?.historyPrefixDigest;
  if (
    hasMutatedHistoryPrefix(
      messages,
      anchorIdx,
      expectedPrefixDigest,
      anchorReplay?.historyPrefixDigest,
      opts.explicitSystem,
    )
  ) {
    return { kind: 'cold-full-replay', reason: 'history-prefix-mutated-or-compacted' };
  }

  const priorSystemPrompt = extractSystemPrompt(messages, opts.explicitSystem, anchorIdx + 1);

  if (isWarmSession && !toolsChanged) {
    return {
      kind: 'warm-incremental',
      tailMessages,
      priorSystemPrompt,
    };
  }

  // Forks carry a parent's dshSessionId and must not mutate the parent's ACP session.
  // Legacy envelopes recorded before dshSessionId existed (undefined) remain resumable.
  // When active tool declarations change mid-session, `resume-incremental` closes the old
  // per-session worker and reloads the same `acpSessionId` with the updated MCP catalog.
  // A suspended tool turn still needs trajectory rebuild: its HTTP request died with the transport.
  const sameOrLegacyDshSession =
    !anchorReplay?.dshSessionId || anchorReplay.dshSessionId === opts.dshSessionId;
  const resumableAcpSessionId = isWarmSession
    ? opts.activeAcpSessionId
    : opts.allowResumeFromReplay
      ? anchorReplay?.acpSessionId
      : undefined;

  if (
    resumableAcpSessionId &&
    sameOrLegacyDshSession &&
    !messages[anchorIdx]!.content.some((block) => block.type === 'tool-call')
  ) {
    return {
      kind: 'resume-incremental',
      resumeAcpSessionId: resumableAcpSessionId,
      tailMessages,
      priorSystemPrompt,
    };
  }

  return {
    kind: 'cold-full-replay',
    reason: toolsChanged ? 'tool-declarations-changed' : 'worker-restarted-without-resume',
  };
}
