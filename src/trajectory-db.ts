import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import type {
  ContentBlock,
  RequestMessage,
  ToolResultMessage,
  ToolSchema,
} from '@deepseek-ai/dsh-llm';
import { checkAgyStateDirectory } from './binary.js';
import { encodeMcpToolName, MCP_SERVER_NAME } from './mcp-bridge.js';
import {
  blocksToPlainText,
  cleanSystemPromptText,
  formatHarnessGuidelines,
  isAmbientContextMessage,
} from './session-sync.js';

/**
 * Official Gemini validator-bypass sentinel for replaying completed historical
 * tool calls / model turns that originated from non-Gemini providers (Claude,
 * DeepSeek, OpenAI) or compacted/aborted turns lacking a Google-issued binary
 * `thoughtSignature`.
 */
export const GEMINI_DUMMY_THOUGHT_SIGNATURE = 'skip_thought_signature_validator';

export const AGY_STEP_TYPE_USER_INPUT = 14;
export const AGY_STEP_TYPE_PLANNER_RESPONSE = 15;
export const AGY_STEP_TYPE_MCP_TOOL = 132;
const AGY_STEP_STATUS_DONE = 3;
const AGY_SOURCE_MODEL = 2;
const AGY_SOURCE_USER = 4;

export interface ConvertedAgyToolCall {
  callId: string;
  dshToolName: string;
  mcpToolName: string;
  qualifiedAcpName: string;
  argumentsJson: string;
  toolSummary: string;
  outputText: string;
  isError: boolean;
}

export type ConvertedAgyStep =
  | {
      kind: 'user-input';
      text: string;
    }
  | {
      kind: 'planner-text';
      text: string;
    }
  | {
      kind: 'planner-tool-batch';
      calls: ConvertedAgyToolCall[];
    };

function systemBlocksText(msg: RequestMessage): string {
  return msg.content
    .filter((b): b is Extract<ContentBlock, { type: 'text' }> => b.type === 'text')
    .map((b) => b.text)
    .join('\n')
    .trim();
}

function normalizeToolCallArguments(rawArgs: string, fallbackSummary: string): {
  argumentsJson: string;
  toolSummary: string;
} {
  let obj: Record<string, unknown> | undefined;
  try {
    const parsed: unknown = JSON.parse(rawArgs);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      obj = { ...(parsed as Record<string, unknown>) };
    }
  } catch {
    obj = undefined;
  }

  if (!obj) {
    const wrapped = {
      ...(rawArgs ? { raw: rawArgs } : {}),
      toolSummary: fallbackSummary,
    };
    return {
      argumentsJson: JSON.stringify(wrapped),
      toolSummary: fallbackSummary,
    };
  }

  const summary =
    typeof obj.toolSummary === 'string' && obj.toolSummary.trim()
      ? obj.toolSummary.trim()
      : fallbackSummary;
  if (typeof obj.toolSummary !== 'string' || !obj.toolSummary.trim()) {
    obj.toolSummary = summary;
  }
  return {
    argumentsJson: JSON.stringify(obj),
    toolSummary: summary,
  };
}

function visibleAssistantBlocksText(blocks: readonly ContentBlock[]): string {
  const nonReasoningAndNonTool = blocks.filter(
    (b) => b.type !== 'reasoning' && b.type !== 'tool-call',
  );
  return blocksToPlainText(nonReasoningAndNonTool).trim();
}

/**
 * Resolve the SQLite database path where `agy` stores a session trajectory
 * (`~/.gemini/antigravity-cli/conversations/<sessionId>.db`, or `$AGY_STATE_DIR` in tests).
 */
export function resolveAgyConversationDbPath(
  sessionId: string,
  env: NodeJS.ProcessEnv = process.env,
): string {
  const overrideDir = env.AGY_STATE_DIR?.trim();
  const stateDir = overrideDir ? path.resolve(overrideDir) : checkAgyStateDirectory().stateDir;
  return path.join(stateDir, 'conversations', `${sessionId}.db`);
}

/**
 * Split a full DSH `RequestMessage[]` list into:
 * - `historyMessages`: messages up through the last completed `assistant` / `tool` message
 *   (including in-place `system` messages so guidelines updates stay ordered), and
 * - `tailMessages`: trailing `user` / `developer` / `system` inputs for the active `session/prompt`.
 */
export function splitMessagesForTrajectorySync(messages: readonly RequestMessage[]): {
  hasPriorTurns: boolean;
  historyMessages: RequestMessage[];
  tailMessages: RequestMessage[];
} {
  let splitIdx = messages.length;
  while (splitIdx > 0) {
    const role = messages[splitIdx - 1]!.role;
    if (role === 'assistant' || role === 'tool') break;
    splitIdx--;
  }
  return {
    hasPriorTurns: splitIdx > 0,
    historyMessages: messages.slice(0, splitIdx),
    tailMessages: messages.slice(splitIdx),
  };
}

/**
 * Convert historical DSH messages (from any provider — Claude, DeepSeek, OpenAI, or compacted/forked
 * DSH history) into Gemini-valid `agy` trajectory steps:
 * - Strips foreign `reasoning` blocks and attaches `GEMINI_DUMMY_THOUGHT_SIGNATURE` to completed
 *   assistant/tool steps so Cloud Code's validator accepts cross-model history.
 * - Skips orphaned/aborted `tool-call` blocks that have no matching `ToolResultMessage`.
 * - Splits mixed `text + tool-call` assistant messages into separate text and tool-batch steps
 *   so unsigned text and signed `functionCall` parts never mix.
 * - Coalesces consecutive `developer` + ambient + `user` messages into single `USER_INPUT` turns
 *   and guarantees the trajectory starts with a `USER_INPUT` step carrying `<workspace_guidelines>`.
 */
export function convertDshMessagesToAgySteps(opts: {
  historyMessages: readonly RequestMessage[];
  explicitSystem?: string;
  tools?: readonly ToolSchema[];
  bridgeToolNames?: readonly string[] | boolean;
}): ConvertedAgyStep[] {
  const { historyMessages } = opts;
  if (historyMessages.length === 0) return [];

  // Build deterministic DSH -> MCP tool name mapping matching McpSessionEndpoint
  const usedMcpNames = new Set<string>();
  const dshToMcp = new Map<string, string>();
  for (const tool of opts.tools ?? []) {
    if (tool.deferLoading) continue;
    dshToMcp.set(tool.name, encodeMcpToolName(tool.name, usedMcpNames));
  }
  const resolveMcpName = (dshName: string): string => {
    const existing = dshToMcp.get(dshName);
    if (existing) return existing;
    const encoded = encodeMcpToolName(dshName, usedMcpNames);
    dshToMcp.set(dshName, encoded);
    return encoded;
  };

  // Pre-scan all tool results in history so we pair 1-to-1 and drop orphaned tool calls
  const toolResultByCallId = new Map<string, ToolResultMessage>();
  for (const msg of historyMessages) {
    if (msg.role === 'tool') {
      const tr = msg as ToolResultMessage;
      if (tr.toolCallId) {
        toolResultByCallId.set(tr.toolCallId, tr);
      }
    }
  }

  // Extract initial system prompt prior to the first non-system message
  let firstNonSystemIdx = 0;
  let initialInHistorySystem: string | undefined;
  while (
    firstNonSystemIdx < historyMessages.length &&
    historyMessages[firstNonSystemIdx]!.role === 'system'
  ) {
    initialInHistorySystem = systemBlocksText(historyMessages[firstNonSystemIdx]!);
    firstNonSystemIdx++;
  }

  const initialCombinedSystem = [opts.explicitSystem?.trim(), initialInHistorySystem]
    .filter(Boolean)
    .join('\n\n') || undefined;
  let effectiveCleanedSystem = cleanSystemPromptText(initialCombinedSystem);

  const initialGuidelines = formatHarnessGuidelines(
    initialCombinedSystem,
    opts.bridgeToolNames ?? Boolean(opts.tools && opts.tools.some((t) => !t.deferLoading)),
  );
  let guidelinesInjected = false;

  const steps: ConvertedAgyStep[] = [];
  const pairedToolCallIds = new Set<string>();

  let pendingDevBlocks: string[] = [];
  let pendingGuidelinesUpdates: string[] = [];
  let pendingAmbientBlocks: string[] = [];
  let pendingPrimaryUsers: string[] = [];

  const flushUserStep = (forcePlaceholder = false): void => {
    const sections: string[] = [];
    if (!guidelinesInjected && initialGuidelines) {
      sections.push(initialGuidelines);
      guidelinesInjected = true;
    }
    for (const dev of pendingDevBlocks) {
      sections.push(`<developer_context>\n${dev}\n</developer_context>`);
    }
    for (const sysUpd of pendingGuidelinesUpdates) {
      sections.push(`<workspace_guidelines_update>\n${sysUpd}\n</workspace_guidelines_update>`);
    }
    if (pendingAmbientBlocks.length > 0) {
      sections.push(`<runtime_context>\n${pendingAmbientBlocks.join('\n\n')}\n</runtime_context>`);
    }
    if (pendingPrimaryUsers.length === 1) {
      sections.push(pendingPrimaryUsers[0]!);
    } else if (pendingPrimaryUsers.length > 1) {
      sections.push(
        pendingPrimaryUsers.map((u) => `<user_message>\n${u}\n</user_message>`).join('\n'),
      );
    }

    pendingDevBlocks = [];
    pendingGuidelinesUpdates = [];
    pendingAmbientBlocks = [];
    pendingPrimaryUsers = [];

    if (sections.length === 0) {
      if (forcePlaceholder) {
        steps.push({ kind: 'user-input', text: 'Continue.' });
      }
      return;
    }

    // If only guidelines were present without user text, add a minimal prompt anchor
    if (
      sections.length === 1 &&
      sections[0] === initialGuidelines &&
      forcePlaceholder
    ) {
      sections.push('Continue.');
    }

    steps.push({ kind: 'user-input', text: sections.join('\n\n') });
  };

  for (let i = firstNonSystemIdx; i < historyMessages.length; i++) {
    const msg = historyMessages[i]!;

    if (msg.role === 'system') {
      const nextCleaned = cleanSystemPromptText(systemBlocksText(msg));
      if (nextCleaned !== effectiveCleanedSystem) {
        effectiveCleanedSystem = nextCleaned;
        if (nextCleaned) pendingGuidelinesUpdates.push(nextCleaned);
      }
      continue;
    }

    if (msg.role === 'developer') {
      const devText = blocksToPlainText(msg.content).trim();
      if (devText) pendingDevBlocks.push(devText);
      continue;
    }

    if (msg.role === 'user') {
      const plain = blocksToPlainText(msg.content).trim();
      if (!plain) continue;
      if (isAmbientContextMessage(msg)) {
        pendingAmbientBlocks.push(plain);
      } else {
        pendingPrimaryUsers.push(plain);
      }
      continue;
    }

    if (msg.role === 'tool') {
      const tr = msg as ToolResultMessage;
      if (pairedToolCallIds.has(tr.toolCallId)) {
        // Already emitted in exact batch order right after its owning planner-tool-batch step
        continue;
      }
      // Stray tool result without a preceding assistant tool-call (e.g. partial history slice):
      // fold as context note rather than emitting an unmatched Gemini functionResponse.
      const strayText = blocksToPlainText(tr.content).trim();
      if (strayText) {
        pendingDevBlocks.push(
          `[Historical tool output (${tr.toolCallId}${tr.isError ? ', error' : ''})]\n${strayText}`,
        );
      }
      continue;
    }

    if (msg.role === 'assistant') {
      const visibleText = visibleAssistantBlocksText(msg.content);
      const completedCalls: ConvertedAgyToolCall[] = [];

      for (const block of msg.content) {
        if (block.type !== 'tool-call') continue;
        const resultMsg = toolResultByCallId.get(block.id);
        if (!resultMsg) {
          // Orphaned/aborted tool call with no matching tool_result: skip cleanly
          continue;
        }
        pairedToolCallIds.add(block.id);
        const mcpToolName = resolveMcpName(block.name);
        const { argumentsJson, toolSummary } = normalizeToolCallArguments(
          block.arguments,
          block.name,
        );
        const rawOut = blocksToPlainText(resultMsg.content);
        const outputText = resultMsg.isError
          ? `[Tool execution error]\n${rawOut}`
          : rawOut;
        completedCalls.push({
          callId: block.id,
          dshToolName: block.name,
          mcpToolName,
          qualifiedAcpName: `mcp_${MCP_SERVER_NAME}_${mcpToolName}`,
          argumentsJson,
          toolSummary,
          outputText,
          isError: Boolean(resultMsg.isError),
        });
      }

      if (!visibleText && completedCalls.length === 0) {
        continue;
      }

      // Ensure trajectory always begins with a USER_INPUT step and flush any queued user context
      flushUserStep(steps.length === 0);

      // Split mixed text + tool-call turns so unsigned text and signed functionCalls never mix
      if (visibleText) {
        steps.push({ kind: 'planner-text', text: visibleText });
      }
      if (completedCalls.length > 0) {
        steps.push({ kind: 'planner-tool-batch', calls: completedCalls });
      }
    }
  }

  // Flush any trailing buffered user/developer notes in historyMessages
  flushUserStep(false);

  return steps;
}

// Minimal Protobuf wire encoders (varint + length-delimited bytes/strings)
function encodeVarint(value: number | bigint): Buffer {
  let v = typeof value === 'bigint' ? value : BigInt(Math.max(0, Math.trunc(value)));
  const out: number[] = [];
  while (v >= 0x80n) {
    out.push(Number((v & 0x7fn) | 0x80n));
    v >>= 7n;
  }
  out.push(Number(v));
  return Buffer.from(out);
}

function pbVarint(fieldNumber: number, value: number | bigint): Buffer {
  return Buffer.concat([encodeVarint((fieldNumber << 3) | 0), encodeVarint(value)]);
}

function pbBytes(fieldNumber: number, payload: Buffer | string): Buffer {
  const buf = typeof payload === 'string' ? Buffer.from(payload, 'utf8') : payload;
  return Buffer.concat([encodeVarint((fieldNumber << 3) | 2), encodeVarint(buf.byteLength), buf]);
}

function encodeTimestamp(epochSeconds: number, nanos = 1_000_000): Buffer {
  return Buffer.concat([pbVarint(1, epochSeconds), pbVarint(2, nanos)]);
}

function encodeStepMetadata(opts: {
  trajectoryId: string;
  sessionId: string;
  stepIndex: number;
  source: number;
  executionId: string;
  epochSeconds: number;
  toolCallMsg?: Buffer;
}): Buffer {
  const stepRef = Buffer.concat([
    pbBytes(1, opts.trajectoryId),
    pbVarint(2, opts.stepIndex),
    pbBytes(4, opts.sessionId),
  ]);
  return Buffer.concat([
    pbBytes(1, encodeTimestamp(opts.epochSeconds)),
    pbVarint(3, opts.source),
    ...(opts.toolCallMsg ? [pbBytes(4, opts.toolCallMsg)] : []),
    pbBytes(12, opts.executionId),
    pbBytes(20, stepRef),
    pbBytes(35, crypto.randomUUID().replace(/-/g, '')),
  ]);
}

/**
 * Hydrate an `agy`-created SQLite session database (`<stateDir>/conversations/<sessionId>.db`)
 * with native multi-step trajectory records converted from DSH's `RequestMessage[]` history.
 */
export function hydrateAgyConversationDb(opts: {
  dbPath: string;
  sessionId: string;
  steps: readonly ConvertedAgyStep[];
}): { insertedRows: number } {
  if (opts.steps.length === 0) return { insertedRows: 0 };
  if (!fs.existsSync(opts.dbPath)) {
    throw new Error(`agy session database does not exist: ${opts.dbPath}`);
  }

  const db = new DatabaseSync(opts.dbPath);
  try {
    const metaRow = db
      .prepare('SELECT trajectory_id FROM trajectory_meta LIMIT 1')
      .get() as { trajectory_id?: string } | undefined;
    const trajectoryId = metaRow?.trajectory_id || crypto.randomUUID();

    const sigMsg = pbBytes(2, pbBytes(1, GEMINI_DUMMY_THOUGHT_SIGNATURE));
    const baseEpoch = Math.max(1, Math.floor(Date.now() / 1000) - opts.steps.length * 2 - 5);

    const insertStmt = db.prepare(
      'INSERT INTO steps (idx, step_type, status, has_subtrajectory, metadata, step_payload, step_format) VALUES (?, ?, ?, 0, ?, ?, 0)',
    );

    db.exec('BEGIN IMMEDIATE');
    try {
      db.exec('DELETE FROM steps');
      let rowIdx = 0;
      let executionId = crypto.randomUUID();

      for (const step of opts.steps) {
        const epochSeconds = baseEpoch + rowIdx;

        if (step.kind === 'user-input') {
          executionId = crypto.randomUUID();
          const meta = encodeStepMetadata({
            trajectoryId,
            sessionId: opts.sessionId,
            stepIndex: rowIdx,
            source: AGY_SOURCE_USER,
            executionId,
            epochSeconds,
          });
          const userBody = Buffer.concat([
            pbBytes(2, step.text),
            pbBytes(3, pbBytes(1, step.text)),
          ]);
          const payload = Buffer.concat([
            pbVarint(1, AGY_STEP_TYPE_USER_INPUT),
            pbVarint(4, AGY_STEP_STATUS_DONE),
            pbBytes(5, meta),
            pbBytes(19, userBody),
          ]);
          insertStmt.run(rowIdx++, AGY_STEP_TYPE_USER_INPUT, AGY_STEP_STATUS_DONE, meta, payload);
          continue;
        }

        if (step.kind === 'planner-text') {
          const meta = encodeStepMetadata({
            trajectoryId,
            sessionId: opts.sessionId,
            stepIndex: rowIdx,
            source: AGY_SOURCE_MODEL,
            executionId,
            epochSeconds,
          });
          const botId = `bot-${crypto.randomUUID()}`;
          const plannerBody = Buffer.concat([
            pbBytes(1, step.text),
            pbBytes(6, botId),
            pbBytes(8, step.text),
            pbVarint(12, 2),
            pbBytes(14, sigMsg),
          ]);
          const payload = Buffer.concat([
            pbVarint(1, AGY_STEP_TYPE_PLANNER_RESPONSE),
            pbVarint(4, AGY_STEP_STATUS_DONE),
            pbBytes(5, meta),
            pbBytes(20, plannerBody),
          ]);
          insertStmt.run(
            rowIdx++,
            AGY_STEP_TYPE_PLANNER_RESPONSE,
            AGY_STEP_STATUS_DONE,
            meta,
            payload,
          );
          continue;
        }

        if (step.kind === 'planner-tool-batch') {
          const encodedCalls = step.calls.map((call, callPos) =>
            Buffer.concat([
              pbBytes(1, call.callId),
              pbBytes(2, call.qualifiedAcpName),
              pbBytes(3, call.argumentsJson),
              // Gemini attaches thoughtSignature to the first parallel functionCall in a turn
              ...(callPos === 0 ? [pbBytes(7, sigMsg)] : []),
            ]),
          );

          const plannerMeta = encodeStepMetadata({
            trajectoryId,
            sessionId: opts.sessionId,
            stepIndex: rowIdx,
            source: AGY_SOURCE_MODEL,
            executionId,
            epochSeconds,
          });
          const botId = `bot-${crypto.randomUUID()}`;
          const plannerBody = Buffer.concat([
            pbBytes(6, botId),
            ...encodedCalls.map((tcBuf) => pbBytes(7, tcBuf)),
            pbVarint(12, 2),
          ]);
          const plannerPayload = Buffer.concat([
            pbVarint(1, AGY_STEP_TYPE_PLANNER_RESPONSE),
            pbVarint(4, AGY_STEP_STATUS_DONE),
            pbBytes(5, plannerMeta),
            pbBytes(20, plannerBody),
          ]);
          insertStmt.run(
            rowIdx++,
            AGY_STEP_TYPE_PLANNER_RESPONSE,
            AGY_STEP_STATUS_DONE,
            plannerMeta,
            plannerPayload,
          );

          for (let cIdx = 0; cIdx < step.calls.length; cIdx++) {
            const call = step.calls[cIdx]!;
            const tcBuf = encodedCalls[cIdx]!;
            const toolMeta = encodeStepMetadata({
              trajectoryId,
              sessionId: opts.sessionId,
              stepIndex: rowIdx,
              source: AGY_SOURCE_MODEL,
              executionId,
              epochSeconds: baseEpoch + rowIdx,
              toolCallMsg: tcBuf,
            });
            const mcpIdent = `${MCP_SERVER_NAME}/${call.mcpToolName}`;
            const mcpIdMsg = Buffer.concat([pbBytes(1, 'mcp'), pbBytes(2, mcpIdent)]);
            const field133 = pbBytes(
              2,
              Buffer.concat([
                pbBytes(1, mcpIdMsg),
                pbVarint(2, 1),
                pbBytes(5, 'mcp'),
                pbBytes(6, mcpIdent),
              ]),
            );
            const field140 = Buffer.concat([
              pbBytes(1, Buffer.concat([pbBytes(1, 'toolSummary'), pbBytes(2, call.toolSummary)])),
              pbBytes(2, pbBytes(1, call.outputText)),
            ]);
            const toolPayload = Buffer.concat([
              pbVarint(1, AGY_STEP_TYPE_MCP_TOOL),
              pbVarint(4, AGY_STEP_STATUS_DONE),
              pbBytes(5, toolMeta),
              pbBytes(133, field133),
              pbBytes(140, field140),
            ]);
            insertStmt.run(
              rowIdx++,
              AGY_STEP_TYPE_MCP_TOOL,
              AGY_STEP_STATUS_DONE,
              toolMeta,
              toolPayload,
            );
          }
        }
      }

      db.exec('COMMIT');
      return { insertedRows: rowIdx };
    } catch (err) {
      try {
        db.exec('ROLLBACK');
      } catch {
        // Ignore rollback failure
      }
      throw err;
    }
  } finally {
    db.close();
  }
}
