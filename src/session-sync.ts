import crypto from 'node:crypto';
import type { ImageAttachmentRef } from '@deepseek-ai/dsh-attachment';
import type {
  AssistantMessage,
  ContentBlock,
  Message,
  RequestMessage,
  ToolResultMessage,
} from '@deepseek-ai/dsh-llm';
import type { AcpPromptPart, AgyReplayMetadata } from './types.js';

export type ImageBytesLoader = (
  ref: ImageAttachmentRef,
  signal?: AbortSignal,
) => Promise<{ base64: string; mimeType: string } | undefined>;

function sha1Short(text: string): string {
  return crypto.createHash('sha1').update(text).digest('hex').slice(0, 16);
}

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

/** Extract combined system prompt from `options.system` and `role: 'system'` messages. */
export function extractSystemPrompt(
  messages: readonly RequestMessage[],
  explicitSystem?: string,
): string | undefined {
  const parts: string[] = [];
  if (explicitSystem?.trim()) {
    parts.push(explicitSystem.trim());
  }
  for (const msg of messages) {
    if (msg.role !== 'system') continue;
    const text = msg.content
      .filter((b): b is Extract<ContentBlock, { type: 'text' }> => b.type === 'text')
      .map((b) => b.text)
      .join('\n')
      .trim();
    if (text) parts.push(text);
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
  const cleaned = (rawSystemPrompt ?? '')
    .replace(THIRD_PARTY_PERSONA_LINE_RE, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

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

/** Check whether a user message is an automatic DSH runtime notice / reminder rather than human input. */
export function isEphemeralRuntimeNotice(text: string): boolean {
  const trimmed = text.trim();
  return (
    trimmed.startsWith('<system-reminder>') ||
    trimmed.startsWith('Time sampled while preparing turn ') ||
    trimmed.startsWith('Current runtime context.') ||
    trimmed.startsWith('You are repeating the exact same tool call') ||
    trimmed.startsWith('The user changed the approval policy ')
  );
}

/** Serialize non-image content blocks of a message into deterministic text for digesting or tool results. */
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

/** Compute per-slot digests of `developer` and `user` context surfaces for Context OS diffing. */
export function computeContextDigests(messages: readonly RequestMessage[]): Record<string, string> {
  const digests: Record<string, string> = {};
  let ordinal = 0;
  for (const msg of messages) {
    if (msg.role !== 'developer' && msg.role !== 'user') continue;
    const text = blocksToPlainText(msg.content).trim();
    if (!text || isEphemeralRuntimeNotice(text)) continue;
    const key = isDurableMessage(msg) ? `${msg.role}:${msg.id}` : `${msg.role}#${ordinal++}`;
    digests[key] = sha1Short(text);
  }
  return digests;
}

/**
 * Compute a deterministic digest over conversational history messages (`user`, `assistant`, `tool`)
 * in `[0, endExclusive)` to detect history compaction or in-place mutations (e.g. billion-context,
 * DSH compaction, tool-result pruning, image offloading).
 *
 * Excludes `system`, `developer`, and attachment-free ephemeral runtime notices so that
 * `dsh-context-os` `developer` surface updates continue using incremental sync without cold replays.
 */
export function computeHistoryPrefixDigest(
  messages: readonly RequestMessage[],
  endExclusive = messages.length,
): string {
  const limit = Math.max(0, Math.min(endExclusive, messages.length));
  const hash = crypto.createHash('sha256');

  for (let i = 0; i < limit; i++) {
    const msg = messages[i]!;
    if (msg.role === 'system' || msg.role === 'developer') continue;

    if (msg.role === 'user') {
      const plain = blocksToPlainText(msg.content).trim();
      const hasAttachment = msg.content.some((b) => b.type === 'image' || b.type === 'file');
      if (!hasAttachment && (!plain || isEphemeralRuntimeNotice(plain))) continue;
      hash.update('user\0');
      hash.update(String(Buffer.byteLength(plain, 'utf8')));
      hash.update('\0');
      hash.update(plain);
      hash.update('\0');
      continue;
    }

    if (msg.role === 'assistant') {
      const plain = blocksToPlainText(msg.content);
      hash.update('assistant\0');
      hash.update(String(Buffer.byteLength(plain, 'utf8')));
      hash.update('\0');
      hash.update(plain);
      hash.update('\0');
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
): boolean {
  if (!expectedPrefixDigest && !anchorPrefixDigest) return false;
  const currentPrefixDigest = computeHistoryPrefixDigest(messages, endExclusive);
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

/** Convert incremental tail messages (typically the new user turn + any developer context) into ACP parts. */
export async function buildIncrementalPromptParts(opts: {
  tailMessages: readonly RequestMessage[];
  updatedContextBlocks?: readonly string[];
  loadImage?: ImageBytesLoader;
  signal?: AbortSignal;
}): Promise<AcpPromptPart[]> {
  const parts: AcpPromptPart[] = [];

  if (opts.updatedContextBlocks && opts.updatedContextBlocks.length > 0) {
    appendTextPart(
      parts,
      `<updated_context_surface>\n${opts.updatedContextBlocks.join('\n\n')}\n</updated_context_surface>`,
    );
  }

  const humanUserMessages: RequestMessage[] = [];
  const runtimeNotices: string[] = [];

  for (const msg of opts.tailMessages) {
    if (msg.role === 'system') continue;
    if (msg.role === 'developer') {
      const devText = blocksToPlainText(msg.content).trim();
      if (devText) {
        appendTextPart(parts, `<developer_context>\n${devText}\n</developer_context>`);
      }
      continue;
    }
    if (msg.role === 'user') {
      const plain = blocksToPlainText(msg.content).trim();
      const hasImage = msg.content.some((b) => b.type === 'image');
      if (!hasImage && plain && isEphemeralRuntimeNotice(plain)) {
        runtimeNotices.push(plain);
      } else {
        humanUserMessages.push(msg);
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

  if (runtimeNotices.length > 0) {
    appendTextPart(parts, `<runtime_context>\n${runtimeNotices.join('\n\n')}\n</runtime_context>`);
  }

  for (const userMsg of humanUserMessages) {
    if (humanUserMessages.length > 1) appendTextPart(parts, '<user_message>');
    await appendBlocksWithImages(parts, userMsg.content, opts.loadImage, opts.signal);
    if (humanUserMessages.length > 1) appendTextPart(parts, '</user_message>');
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
  const guidelines = formatHarnessGuidelines(opts.systemPrompt, opts.bridgeToolNames ?? Boolean(opts.hasTools));
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
        const hasImage = msg.content.some((b) => b.type === 'image');
        if (!hasImage && plain && isEphemeralRuntimeNotice(plain)) continue;
        appendTextPart(parts, '<user_turn>');
        appendTextPart(parts, plain);
        appendTextPart(parts, '</user_turn>');
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
    loadImage: opts.loadImage,
    signal: opts.signal,
  });
  parts.push(...current);
  appendTextPart(parts, '</current_input>');
  return parts;
}

export type SessionSyncDecision =
  | {
      kind: 'resolve-pending-tools';
      resolutions: Array<{ toolCallId: string; text: string; isError?: boolean }>;
    }
  | {
      kind: 'warm-incremental';
      tailMessages: RequestMessage[];
      updatedContextBlocks: string[];
    }
  | {
      kind: 'resume-incremental';
      resumeAcpSessionId: string;
      tailMessages: RequestMessage[];
      updatedContextBlocks: string[];
    }
  | {
      kind: 'cold-full-replay';
      reason: string;
    };

/**
 * Decide how to synchronize DSH's `options.messages` onto an ACP worker session,
 * remaining resilient to `dsh-context-os` `surfaceOp: 'replace'` mutations and runtime notices.
 */
export function planSessionSync(opts: {
  messages: readonly RequestMessage[];
  activeAcpSessionId?: string;
  lastSyncedAnchorId?: string;
  lastToolsFingerprint?: string;
  currentToolsFingerprint: string;
  lastContextDigests?: Record<string, string>;
  lastHistoryPrefixDigest?: string;
  pendingToolCallIds: ReadonlySet<string>;
  allowResumeFromReplay: boolean;
  dshSessionId?: string;
  forwardContextUpdates: boolean;
}): SessionSyncDecision {
  const { messages, pendingToolCallIds } = opts;

  // 1. Mid-turn suspended MCP tool calls: match ToolResultMessages by toolCallId
  if (pendingToolCallIds.size > 0 && opts.activeAcpSessionId) {
    const resultByCallId = new Map<string, ToolResultMessage>();
    const trailingNotes: string[] = [];

    // Scan from the last assistant message carrying any of the pending toolCallIds
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
        } else if (msg.role === 'developer') {
          const text = blocksToPlainText(msg.content).trim();
          if (text) trailingNotes.push(text);
        } else if (msg.role === 'user') {
          const text = blocksToPlainText(msg.content).trim();
          if (text && !isEphemeralRuntimeNotice(text)) {
            trailingNotes.push(text);
          }
        }
      }

      if (resultByCallId.size === pendingToolCallIds.size) {
        const toolTurnReplay = extractReplayMetadata(messages[toolTurnStartIdx]!);
        if (
          hasMutatedHistoryPrefix(
            messages,
            toolTurnStartIdx,
            opts.lastHistoryPrefixDigest ?? toolTurnReplay?.historyPrefixDigest,
            toolTurnReplay?.historyPrefixDigest,
          )
        ) {
          return { kind: 'cold-full-replay', reason: 'history-prefix-mutated-or-compacted' };
        }

        const orderedIds = [...pendingToolCallIds];
        const resolutions = orderedIds.map((id, idx) => {
          const toolMsg = resultByCallId.get(id)!;
          let text = blocksToPlainText(toolMsg.content);
          if (idx === orderedIds.length - 1 && trailingNotes.length > 0) {
            text = `${text}\n\n<dsh_context_note>\n${trailingNotes.join('\n\n')}\n</dsh_context_note>`;
          }
          return {
            toolCallId: id,
            text,
            isError: toolMsg.isError,
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
  if (effectiveToolsFp && effectiveToolsFp !== opts.currentToolsFingerprint) {
    return { kind: 'cold-full-replay', reason: 'tool-declarations-changed' };
  }

  const tailMessages = messages.slice(anchorIdx + 1);
  if (tailMessages.length === 0) {
    return { kind: 'cold-full-replay', reason: 'empty-incremental-tail' };
  }

  // Detect history compaction or mutation (e.g. billion-context, DSH compaction, manual edits)
  // before anchorIdx without treating developer/system Context OS updates as history mutations.
  const expectedPrefixDigest = isWarmSession
    ? (opts.lastHistoryPrefixDigest ?? anchorReplay?.historyPrefixDigest)
    : anchorReplay?.historyPrefixDigest;
  if (
    hasMutatedHistoryPrefix(
      messages,
      anchorIdx,
      expectedPrefixDigest,
      anchorReplay?.historyPrefixDigest,
    )
  ) {
    return { kind: 'cold-full-replay', reason: 'history-prefix-mutated-or-compacted' };
  }

  // Detect any Context OS `surfaceOp: 'replace'` edits prior to anchorIdx without killing the session
  const updatedContextBlocks: string[] = [];
  const baselineDigests = isWarmSession
    ? (opts.lastContextDigests ?? anchorReplay?.contextDigests)
    : anchorReplay?.contextDigests;
  if (opts.forwardContextUpdates && baselineDigests) {
    let ordinal = 0;
    for (let i = 0; i <= anchorIdx; i++) {
      const msg = messages[i]!;
      if (msg.role !== 'developer' && msg.role !== 'user') continue;
      const text = blocksToPlainText(msg.content).trim();
      if (!text || isEphemeralRuntimeNotice(text)) continue;
      const key = isDurableMessage(msg) ? `${msg.role}:${msg.id}` : `${msg.role}#${ordinal++}`;
      const prevHash = baselineDigests[key];
      const nextHash = sha1Short(text);
      if (prevHash !== nextHash && msg.role === 'developer') {
        updatedContextBlocks.push(text);
      }
    }
  }

  if (isWarmSession) {
    return {
      kind: 'warm-incremental',
      tailMessages,
      updatedContextBlocks,
    };
  }

  // Forks can carry a parent's replay envelope. Rebuild from their own DSH history.
  // A suspended tool turn also needs replay: its HTTP request died with the transport.
  if (opts.allowResumeFromReplay && anchorReplay?.acpSessionId &&
      anchorReplay.dshSessionId === opts.dshSessionId &&
      !messages[anchorIdx]!.content.some(block => block.type === 'tool-call')) {
    return {
      kind: 'resume-incremental',
      resumeAcpSessionId: anchorReplay.acpSessionId,
      tailMessages,
      updatedContextBlocks,
    };
  }

  return { kind: 'cold-full-replay', reason: 'worker-restarted-without-resume' };
}
