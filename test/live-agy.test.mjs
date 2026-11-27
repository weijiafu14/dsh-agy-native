import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { AgyAcpAdapter, resolveAgyAcpConfig } from '../dist/core.js';

// 1x1 solid red PNG
const RED_1X1_PNG_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';

async function collectChunks(iterable) {
  const chunks = [];
  for await (const chunk of iterable) {
    chunks.push(chunk);
  }
  return chunks;
}

test(
  'Live official agy E2E: eager tool call + warm session continuation + multimodal image',
  { skip: process.env.DSH_AGY_LIVE_TEST !== '1' },
  async () => {
    const tmpCache = fs.mkdtempSync(path.join(os.tmpdir(), 'agy-live-e2e-'));
    const cfg = resolveAgyAcpConfig({
      poolSize: 1,
      batchWindowMs: 25,
      cacheDir: tmpCache,
    });

    const adapter = new AgyAcpAdapter({
      getConfig: () => cfg,
      loadImage: async () => ({
        base64: RED_1X1_PNG_BASE64,
        mimeType: 'image/png',
      }),
    });

    try {
      const models = await adapter.listModels('agy-acp');
      assert.ok(models.length > 0, 'Expected real models discovered from agy');

      const tools = [
        {
          name: 'lookup_magic_code',
          description: 'Returns the secret verification code for a given key.',
          parameters: {
            type: 'object',
            properties: { key: { type: 'string' } },
            required: ['key'],
            additionalProperties: false,
          },
        },
      ];

      const messages = [
        {
          id: 'u-live-1',
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Call lookup_magic_code with key="jade" right now and reply with the exact code returned.',
            },
            {
              type: 'image',
              attachment: {
                attachmentId: 'red-dot',
                mediaType: 'image/png',
                bytes: 68,
                width: 1,
                height: 1,
              },
            },
          ],
          source: { kind: 'user' },
        },
      ];

      let finishReason = 'tool-calls';
      let stepGuard = 0;
      let finalText = '';
      let firstAcpSessionId;

      while (finishReason === 'tool-calls' && stepGuard++ < 4) {
        const chunks = await collectChunks(
          adapter.stream({
            provider: 'agy-acp',
            model: 'gemini-3.8-flash',
            reasoningEffort: 'low',
            sessionId: 'live-dsh-session-1',
            messages,
            tools,
          }),
        );

        const finish = chunks.find((c) => c.type === 'finish');
        assert.ok(finish);
        finishReason = finish.reason.kind;
        firstAcpSessionId = finish.replayState.response.acpSessionId;

        const blocks = chunks.filter((c) => c.type === 'block-end').map((c) => c.block);
        messages.push({
          id: `asst-live-${stepGuard}`,
          role: 'assistant',
          content: blocks,
          source: {
            provider: 'agy-acp',
            model: 'gemini-3.8-flash',
            replayState: finish.replayState,
          },
        });

        if (finishReason === 'tool-calls') {
          const toolCalls = blocks.filter((b) => b.type === 'tool-call');
          assert.ok(toolCalls.length >= 1);
          for (const tc of toolCalls) {
            const parsedArgs = JSON.parse(tc.arguments);
            assert.equal(parsedArgs.toolSummary, undefined, 'toolSummary must be stripped');
            assert.equal(parsedArgs.key, 'jade');
            messages.push({
              id: `tool-live-${tc.id}`,
              role: 'tool',
              toolCallId: tc.id,
              content: [{ type: 'text', text: 'MAGIC-9482-OK' }],
              source: { kind: 'tool' },
            });
          }
        } else {
          finalText = blocks
            .filter((b) => b.type === 'text')
            .map((b) => b.text)
            .join('');
        }
      }

      assert.equal(finishReason, 'stop');
      assert.match(finalText, /MAGIC-9482-OK/);

      // Turn 2: warm incremental follow-up on the SAME ACP session!
      messages.push({
        id: 'u-live-2',
        role: 'user',
        content: [{ type: 'text', text: 'Repeat just the 4 digits from that code.' }],
        source: { kind: 'user' },
      });

      const turn2Chunks = await collectChunks(
        adapter.stream({
          provider: 'agy-acp',
          model: 'gemini-3.8-flash',
          reasoningEffort: 'low',
          sessionId: 'live-dsh-session-1',
          messages,
          tools,
        }),
      );
      const turn2Finish = turn2Chunks.find((c) => c.type === 'finish');
      assert.equal(turn2Finish.reason.kind, 'stop');
      assert.equal(
        turn2Finish.replayState.response.acpSessionId,
        firstAcpSessionId,
        'Expected Turn 2 to reuse the exact same warm ACP session',
      );
      const turn2Text = turn2Chunks
        .filter((c) => c.type === 'block-end' && c.block.type === 'text')
        .map((c) => c.block.text)
        .join('');
      assert.match(turn2Text, /9482/);

      // A new role on this same pool slot switches from direct tools to PTC.
      const ptc = await collectChunks(adapter.stream({
        provider: 'agy-acp',
        model: 'gemini-3.8-flash',
        sessionId: 'live-dsh-session-2',
        messages: [{ role: 'user', content: [{ type: 'text', text: 'Look up the verification code for jade using the available tool.' }] }],
        tools: [{
          name: 'run_code',
          description: 'Run a JavaScript program. The available SDK method is tools.lookup_magic_code({key: string}).',
          parameters: {
            type: 'object',
            properties: { code: { type: 'string' } },
            required: ['code'],
          },
        }],
      }));
      const call = ptc.find(c => c.type === 'block-end' && c.block.type === 'tool-call')?.block;
      assert.equal(call?.name, 'run_code');
      assert.match(JSON.parse(call.arguments).code, /lookup_magic_code/);

      // Cross-model (>240KB non-Gemini history with foreign reasoning + orphaned call + parallel tool calls)
      // converts into native agy SQLite trajectory via session/load with zero 192KB prompt truncation!
      const hugePadding = 'BUILD_ARTIFACT_LINE_0123456789ABCDEF\n'.repeat(6_800); // ~245 KB > 192 KB
      const crossModelMessages = [
        {
          id: 'cm-u1',
          role: 'user',
          content: [{ type: 'text', text: 'Read the vault key and build log.' }],
          source: { kind: 'user' },
        },
        {
          id: 'cm-a1',
          role: 'assistant',
          content: [
            { type: 'reasoning', text: 'DeepSeek R1 reasoning that has no Google signature...' },
            { type: 'text', text: 'Reading vault and build log in parallel.' },
            { type: 'tool-call', id: 'ds_call_1', name: 'lookup_magic_code', arguments: '{"key":"vault"}' },
            { type: 'tool-call', id: 'ds_call_2', name: 'lookup_magic_code', arguments: '{"key":"log"}' },
            { type: 'tool-call', id: 'ds_call_orphan', name: 'lookup_magic_code', arguments: '{"key":"aborted"}' },
          ],
          source: { provider: 'deepseek', model: 'deepseek-v4' },
        },
        {
          id: 'cm-t1',
          role: 'tool',
          toolCallId: 'ds_call_1',
          content: [{ type: 'text', text: 'VAULT-SAPPHIRE-6104' }],
          source: { kind: 'tool' },
        },
        {
          id: 'cm-t2',
          role: 'tool',
          toolCallId: 'ds_call_2',
          content: [{ type: 'text', text: hugePadding }],
          source: { kind: 'tool' },
        },
        {
          id: 'cm-a2',
          role: 'assistant',
          content: [{ type: 'text', text: 'Finished reading vault and 245KB build log.' }],
          source: { provider: 'deepseek', model: 'deepseek-v4' },
        },
        {
          id: 'cm-u2',
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Without calling any tools, reply with BOTH the exact VAULT-... code from Turn 1 AND the token TAIL-7731.',
            },
          ],
          source: { kind: 'user' },
        },
      ];

      const cmChunks = await collectChunks(
        adapter.stream({
          provider: 'agy-acp',
          model: 'gemini-3.8-flash',
          reasoningEffort: 'low',
          sessionId: 'live-cross-model-session',
          messages: crossModelMessages,
          tools,
        }),
      );
      const cmFinish = cmChunks.find((c) => c.type === 'finish');
      assert.equal(cmFinish?.reason.kind, 'stop');
      const cmText = cmChunks
        .filter((c) => c.type === 'block-end' && c.block.type === 'text')
        .map((c) => c.block.text)
        .join('');
      assert.match(cmText, /VAULT-SAPPHIRE-6104/, 'Must recall Turn 1 tool output from hydrated SQLite trajectory');
      assert.match(cmText, /TAIL-7731/, 'Current input after >240KB history must never be truncated');
    } finally {
      await adapter.dispose();
      fs.rmSync(tmpCache, { recursive: true, force: true });
    }
  },
);
