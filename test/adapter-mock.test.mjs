import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { AgyAcpAdapter, resolveAgyAcpConfig } from '../dist/core.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FAKE_AGY_PATH = path.join(__dirname, 'fixtures', 'fake-agy.mjs');
fs.chmodSync(FAKE_AGY_PATH, 0o755);

async function collectChunks(iterable) {
  const chunks = [];
  for await (const chunk of iterable) {
    chunks.push(chunk);
  }
  return chunks;
}

test('AgyAcpAdapter streams reasoning, parallel tool-calls, warm continuation, images, and isolates aux calls', async () => {
  const tmpCache = fs.mkdtempSync(path.join(os.tmpdir(), 'agy-adapter-test-'));
  const cfg = resolveAgyAcpConfig({
    agyPath: FAKE_AGY_PATH,
    poolSize: 2,
    batchWindowMs: 30,
    cacheDir: tmpCache,
  });

  const adapter = new AgyAcpAdapter({
    getConfig: () => cfg,
    loadImage: async (ref) => ({
      base64: Buffer.from(`PIXELS:${ref.attachmentId}`).toString('base64'),
      mimeType: ref.mediaType,
    }),
  });

  try {
    const models = await adapter.listModels('agy-acp');
    assert.equal(models.length, 2);
    assert.equal(models[0].id, 'gemini-4-carbon');
    assert.equal(models[1].id, 'gemini-4-hydrogen');

    const resolvedHydrogen = await adapter.resolveModel('agy-acp', 'gemini-4-hydrogen');
    assert.deepEqual(
      resolvedHydrogen.reasoning?.efforts.map((e) => e.id),
      ['high', 'xhigh'],
    );

    const tools = [
      { name: 'get_left', description: 'Left', parameters: { type: 'object', properties: { q: { type: 'string' } } } },
      { name: 'get_right', description: 'Right', parameters: { type: 'object', properties: { q: { type: 'string' } } } },
    ];

    // Step 1: User triggers parallel tool calls
    const user1 = {
      id: 'u1',
      role: 'user',
      content: [{ type: 'text', text: 'Please CALL_PARALLEL_TOOLS now' }],
      source: { kind: 'user' },
    };

    const step1Chunks = await collectChunks(
      adapter.stream({
        provider: 'agy-acp',
        model: 'gemini-4-carbon',
        reasoningEffort: 'high',
        sessionId: 'dsh-sess-1',
        messages: [user1],
        tools,
      }),
    );

    const step1Finish = step1Chunks.find((c) => c.type === 'finish');
    assert.equal(step1Finish?.reason.kind, 'tool-calls');

    const toolBlocks = step1Chunks
      .filter((c) => c.type === 'block-end' && c.block.type === 'tool-call')
      .map((c) => c.block);
    assert.equal(toolBlocks.length, 2, 'Both parallel tools coalesced into single DSH turn');
    assert.equal(toolBlocks[0].arguments, '{"q":"L"}', 'toolSummary stripped cleanly');
    assert.equal(toolBlocks[1].arguments, '{"q":"R"}', 'toolSummary stripped cleanly');

    const reasoningEnd = step1Chunks.find(
      (c) => c.type === 'block-end' && c.block.type === 'reasoning',
    );
    assert.equal(reasoningEnd?.block.text, 'Thinking step...');

    // Step 2: DSH returns both ToolResultMessages -> warm worker resolves MCP calls & finishes
    const asstToolMsg = {
      id: 'a1',
      role: 'assistant',
      content: toolBlocks,
      source: {
        provider: 'agy-acp',
        model: 'gemini-4-carbon',
        replayState: step1Finish.replayState,
      },
    };
    const toolRes1 = {
      id: 't1',
      role: 'tool',
      toolCallId: toolBlocks[0].id,
      content: [{ type: 'text', text: 'ALPHA' }],
      source: { kind: 'tool' },
    };
    const toolRes2 = {
      id: 't2',
      role: 'tool',
      toolCallId: toolBlocks[1].id,
      content: [{ type: 'text', text: 'BETA' }],
      source: { kind: 'tool' },
    };

    const step2Chunks = await collectChunks(
      adapter.stream({
        provider: 'agy-acp',
        model: 'gemini-4-carbon',
        reasoningEffort: 'high',
        sessionId: 'dsh-sess-1',
        messages: [user1, asstToolMsg, toolRes1, toolRes2],
        tools,
      }),
    );

    const step2Text = step2Chunks.find((c) => c.type === 'block-end' && c.block.type === 'text');
    assert.equal(step2Text?.block.text, 'Combined=ALPHA+BETA');

    const step1Usage = step1Chunks.find((c) => c.type === 'usage');
    const step2Usage = step2Chunks.find((c) => c.type === 'usage');
    assert.ok(step1Usage, 'Intermediate tool step emits single-step usage from usage_update');
    assert.ok(step2Usage, 'Final step emits reconciled remainder usage');
    assert.equal(step1Usage.usage.totalTokens, 512);
    assert.equal(step1Usage.usage.inputTokens + step2Usage.usage.inputTokens, 1000);
    assert.equal(step1Usage.usage.outputTokens + step2Usage.usage.outputTokens, 50);
    assert.equal((step1Usage.usage.cacheReadTokens ?? 0) + (step2Usage.usage.cacheReadTokens ?? 0), 400);
    assert.equal(step1Usage.usage.totalTokens + step2Usage.usage.totalTokens, 1450);
    assert.equal(step2Usage.usage.reasoningTokens, 15);
    assert.ok(
      step2Usage.usage.inputTokens + (step2Usage.usage.cacheReadTokens ?? 0) < 1000,
      'Final step prompt pressure reflects step 2 alone rather than step1+step2 sum',
    );

    const step2Finish = step2Chunks.find((c) => c.type === 'finish');
    assert.equal(step2Finish?.reason.kind, 'stop');
    const firstSessionId = step2Finish.replayState.response.acpSessionId;

    // Auxiliary call (e.g. session-title) must NOT disturb `firstSessionId` or inject MCP guidelines!
    const auxChunks = await collectChunks(
      adapter.stream({
        provider: 'agy-acp',
        model: 'gemini-4-carbon',
        purpose: 'session-title',
        messages: [{ role: 'user', content: [{ type: 'text', text: 'Generate title ECHO_PROMPT' }] }],
      }),
    );
    const auxFinish = auxChunks.find((c) => c.type === 'finish');
    assert.notEqual(auxFinish.replayState.response.acpSessionId, firstSessionId);
    const auxText = auxChunks.find((c) => c.type === 'block-end' && c.block.type === 'text');
    assert.ok(auxText);
    assert.doesNotMatch(auxText.block.text, /<workspace_guidelines>/);

    // Turn 2 on `dsh-sess-1` with multimodal ImageBlock reuses warm `firstSessionId` (Turn=2)!
    const asstFinal1 = {
      id: 'a2',
      role: 'assistant',
      content: [step2Text.block],
      source: {
        provider: 'agy-acp',
        model: 'gemini-4-carbon',
        replayState: step2Finish.replayState,
      },
    };
    const user2WithImage = {
      id: 'u2',
      role: 'user',
      content: [
        { type: 'text', text: 'Inspect this screenshot' },
        {
          type: 'image',
          attachment: {
            attachmentId: 'img-99',
            mediaType: 'image/png',
            bytes: 128,
            width: 100,
            height: 80,
          },
        },
      ],
      source: { kind: 'user' },
    };

    const turn2Chunks = await collectChunks(
      adapter.stream({
        provider: 'agy-acp',
        model: 'gemini-4-hydrogen',
        reasoningEffort: 'xhigh',
        sessionId: 'dsh-sess-1',
        messages: [user1, asstToolMsg, toolRes1, toolRes2, asstFinal1, user2WithImage],
        tools,
      }),
    );

    const turn2Text = turn2Chunks.find((c) => c.type === 'block-end' && c.block.type === 'text');
    assert.match(
      turn2Text.block.text,
      new RegExp(`Session=${firstSessionId} Turn=2 Model=gemini-4-hydrogen-xhigh \\[img:image/png:`),
    );

    // 4. Verify upstream 429 circuit breaker trips immediately on attempt 1, cancels agy, and throws non-retryable QUOTA error
    const leakedText = [];
    await assert.rejects(
      async () => {
        for await (const chunk of adapter.stream({
          provider: 'agy-acp',
          model: 'gemini-4-carbon',
          system: 'You are an AI agent powered by DeepSeek Harness.\nFollow workspace rules.',
          messages: [
            {
              id: 'u-429',
              role: 'user',
              content: [{ type: 'text', text: 'SIMULATE_UPSTREAM_429' }],
              source: { kind: 'user' },
            },
          ],
        })) {
          if (chunk.type === 'text-delta') leakedText.push(chunk.text);
        }
      },
      (err) => {
        assert.equal(err.code, 'QUOTA');
        assert.match(err.message, /circuit breaker tripped on attempt 1/);
        return true;
      },
    );
    assert.deepEqual(leakedText, [], 'In-band 429 banner must never leak as assistant text');
  } finally {
    await adapter.dispose();
    fs.rmSync(tmpCache, { recursive: true, force: true });
  }
});

test('AgyAcpAdapter resets ACP session on prefix compaction (between turns and mid-turn) without regressing Context OS warm reuse', async () => {
  const tmpCache = fs.mkdtempSync(path.join(os.tmpdir(), 'agy-compaction-test-'));
  const cfg = resolveAgyAcpConfig({
    agyPath: FAKE_AGY_PATH,
    poolSize: 1,
    batchWindowMs: 25,
    cacheDir: tmpCache,
  });
  const adapter = new AgyAcpAdapter({ getConfig: () => cfg });

  const tools = [
    { name: 'get_left', description: 'Left', parameters: { type: 'object', properties: { q: { type: 'string' } } } },
    { name: 'get_right', description: 'Right', parameters: { type: 'object', properties: { q: { type: 'string' } } } },
  ];

  try {
    await adapter.listModels('agy-acp');

    const devV1 = {
      id: 'dev-ctx',
      role: 'developer',
      content: [{ type: 'text', text: 'CTX_OS_STATE_V1' }],
      source: { kind: 'developer' },
    };
    const u1 = {
      id: 'u1',
      role: 'user',
      content: [{ type: 'text', text: 'DROPPED_SECRET_TURN_1' }],
      source: { kind: 'user' },
    };

    // Turn 1: cold start
    const t1Chunks = await collectChunks(
      adapter.stream({
        provider: 'agy-acp',
        model: 'gemini-4-carbon',
        sessionId: 'dsh-compact-1',
        messages: [devV1, u1],
        tools,
      }),
    );
    const t1Text = t1Chunks.find((c) => c.type === 'block-end' && c.block.type === 'text');
    const t1Finish = t1Chunks.find((c) => c.type === 'finish');
    const sess1 = t1Finish.replayState.response.acpSessionId;
    assert.ok(t1Finish.replayState.response.historyPrefixDigest, 'Turn 1 must persist historyPrefixDigest');

    const a1 = {
      id: 'a1',
      role: 'assistant',
      content: [t1Text.block],
      source: { provider: 'agy-acp', model: 'gemini-4-carbon', replayState: t1Finish.replayState },
    };

    // Turn 2: Context OS mutates developer surface only -> must stay on warm sess1 (Turn=2) and forward update
    const devV2 = {
      id: 'dev-ctx',
      role: 'developer',
      content: [{ type: 'text', text: 'CTX_OS_STATE_V2_UPDATED' }],
      source: { kind: 'developer' },
    };
    const u2 = {
      id: 'u2',
      role: 'user',
      content: [{ type: 'text', text: 'ECHO_PROMPT Turn 2 check' }],
      source: { kind: 'user' },
    };
    const t2Chunks = await collectChunks(
      adapter.stream({
        provider: 'agy-acp',
        model: 'gemini-4-carbon',
        sessionId: 'dsh-compact-1',
        messages: [devV2, u1, a1, u2],
        tools,
      }),
    );
    const t2Text = t2Chunks.find((c) => c.type === 'block-end' && c.block.type === 'text').block.text;
    const t2Finish = t2Chunks.find((c) => c.type === 'finish');
    assert.equal(t2Finish.replayState.response.acpSessionId, sess1);
    assert.match(t2Text, new RegExp(`Session=${sess1} Turn=2`));
    assert.match(t2Text, /CTX_OS_STATE_V2_UPDATED/);
    assert.doesNotMatch(t2Text, /<conversation_history>/);

    const a2 = {
      id: 'a2',
      role: 'assistant',
      content: [{ type: 'text', text: t2Text }],
      source: { provider: 'agy-acp', model: 'gemini-4-carbon', replayState: t2Finish.replayState },
    };

    // Turn 3: billion-context replaces [u1, a1] with a summary node while keeping a2 intact -> must reset to fresh ACP session (Turn=1)
    const summaryMsg = {
      id: 'sum-1',
      role: 'user',
      content: [{ type: 'text', text: 'COMPACTED_SUMMARY_MARKER' }],
      source: { kind: 'user' },
    };
    const u3 = {
      id: 'u3',
      role: 'user',
      content: [{ type: 'text', text: 'ECHO_PROMPT Turn 3 after compaction' }],
      source: { kind: 'user' },
    };
    const t3Chunks = await collectChunks(
      adapter.stream({
        provider: 'agy-acp',
        model: 'gemini-4-carbon',
        sessionId: 'dsh-compact-1',
        messages: [devV2, summaryMsg, u2, a2, u3],
        tools,
      }),
    );
    const t3Text = t3Chunks.find((c) => c.type === 'block-end' && c.block.type === 'text').block.text;
    const t3Finish = t3Chunks.find((c) => c.type === 'finish');
    const sess2 = t3Finish.replayState.response.acpSessionId;
    assert.notEqual(sess2, sess1, 'Prefix compaction must force a fresh ACP session');
    assert.match(t3Text, new RegExp(`Session=${sess2} Turn=1`));
    assert.match(t3Text, /<conversation_history>/);
    assert.match(t3Text, /COMPACTED_SUMMARY_MARKER/);
    assert.doesNotMatch(t3Text, /DROPPED_SECRET_TURN_1/);

    const a3 = {
      id: 'a3',
      role: 'assistant',
      content: [{ type: 'text', text: t3Text }],
      source: { provider: 'agy-acp', model: 'gemini-4-carbon', replayState: t3Finish.replayState },
    };

    // Turn 4: subsequent turn resumes normal warm incremental reuse on sess2 (Turn=2)
    const u4 = {
      id: 'u4',
      role: 'user',
      content: [{ type: 'text', text: 'CALL_PARALLEL_TOOLS trigger mid-turn' }],
      source: { kind: 'user' },
    };
    const t4ToolChunks = await collectChunks(
      adapter.stream({
        provider: 'agy-acp',
        model: 'gemini-4-carbon',
        sessionId: 'dsh-compact-1',
        messages: [devV2, summaryMsg, u2, a2, u3, a3, u4],
        tools,
      }),
    );
    const t4ToolFinish = t4ToolChunks.find((c) => c.type === 'finish');
    assert.equal(t4ToolFinish?.reason.kind, 'tool-calls');
    assert.equal(t4ToolFinish.replayState.response.acpSessionId, sess2, 'Turn 4 starts warm on sess2');
    const toolBlocks = t4ToolChunks
      .filter((c) => c.type === 'block-end' && c.block.type === 'tool-call')
      .map((c) => c.block);
    assert.equal(toolBlocks.length, 2);

    const a4Tools = {
      id: 'a4-tools',
      role: 'assistant',
      content: toolBlocks,
      source: { provider: 'agy-acp', model: 'gemini-4-carbon', replayState: t4ToolFinish.replayState },
    };
    const tr1 = {
      id: 'tr-1',
      role: 'tool',
      toolCallId: toolBlocks[0].id,
      content: [{ type: 'text', text: 'ECHO_PROMPT LEFT_OUT' }],
      source: { kind: 'tool' },
    };
    const tr2 = {
      id: 'tr-2',
      role: 'tool',
      toolCallId: toolBlocks[1].id,
      content: [{ type: 'text', text: 'RIGHT_OUT' }],
      source: { kind: 'tool' },
    };

    // Mid-turn compaction replaces [summaryMsg, u2, a2, u3, a3] before a4Tools while pending MCP calls are open
    const midTurnSummary = {
      id: 'sum-mid',
      role: 'user',
      content: [{ type: 'text', text: 'MID_TURN_COMPACTED_SUMMARY' }],
      source: { kind: 'user' },
    };
    const t4AfterMidCompact = await collectChunks(
      adapter.stream({
        provider: 'agy-acp',
        model: 'gemini-4-carbon',
        sessionId: 'dsh-compact-1',
        messages: [devV2, midTurnSummary, u4, a4Tools, tr1, tr2],
        tools,
      }),
    );
    const t4FinalText = t4AfterMidCompact.find((c) => c.type === 'block-end' && c.block.type === 'text').block.text;
    const t4FinalFinish = t4AfterMidCompact.find((c) => c.type === 'finish');
    const sess3 = t4FinalFinish.replayState.response.acpSessionId;
    assert.notEqual(sess3, sess2, 'Mid-turn compaction must abort stale ACP session and open a fresh one');
    assert.match(t4FinalText, new RegExp(`Session=${sess3} Turn=1`));
    assert.match(t4FinalText, /MID_TURN_COMPACTED_SUMMARY/);
    assert.match(t4FinalText, /LEFT_OUT/);
    assert.match(t4FinalText, /RIGHT_OUT/);
    assert.doesNotMatch(t4FinalText, /COMPACTED_SUMMARY_MARKER/);

    // Verify resume-incremental across worker restart: intact prefix resumes sess3 via session/load,
    // whereas compacted prefix before anchor forces a fresh ACP session instead of resuming stale context.
    const a4Final = {
      id: 'a4-final',
      role: 'assistant',
      content: [{ type: 'text', text: t4FinalText }],
      source: { provider: 'agy-acp', model: 'gemini-4-carbon', replayState: t4FinalFinish.replayState },
    };
    const u5 = {
      id: 'u5',
      role: 'user',
      content: [{ type: 'text', text: 'ECHO_PROMPT Turn 5 after restart' }],
      source: { kind: 'user' },
    };

    const resumedAdapter = new AgyAcpAdapter({
      getConfig: () => resolveAgyAcpConfig({ ...cfg, resumeSessions: true }),
    });
    try {
      await resumedAdapter.listModels('agy-acp');

      // Intact history -> resumes sess3 via session/load (Turn=2, no <conversation_history>)
      const intactResumeChunks = await collectChunks(
        resumedAdapter.stream({
          provider: 'agy-acp',
          model: 'gemini-4-carbon',
          sessionId: 'dsh-compact-1',
          messages: [devV2, midTurnSummary, u4, a4Tools, tr1, tr2, a4Final, u5],
          tools,
        }),
      );
      const intactResumeText = intactResumeChunks.find((c) => c.type === 'block-end' && c.block.type === 'text').block.text;
      const intactResumeFinish = intactResumeChunks.find((c) => c.type === 'finish');
      assert.equal(intactResumeFinish.replayState.response.acpSessionId, sess3);
      assert.match(intactResumeText, new RegExp(`Session=${sess3} Turn=2`));
      assert.doesNotMatch(intactResumeText, /<conversation_history>/);

      // Simulate restart again, now with compacted prefix before a4Final -> refuses session/load, opens fresh session
      await resumedAdapter.dispose();
      const postRestartSummary = {
        id: 'sum-restart',
        role: 'user',
        content: [{ type: 'text', text: 'POST_RESTART_COMPACTED_SUMMARY' }],
        source: { kind: 'user' },
      };
      const compactedResumeChunks = await collectChunks(
        resumedAdapter.stream({
          provider: 'agy-acp',
          model: 'gemini-4-carbon',
          sessionId: 'dsh-compact-1',
          messages: [devV2, postRestartSummary, a4Final, u5],
          tools,
        }),
      );
      const compactedResumeText = compactedResumeChunks.find((c) => c.type === 'block-end' && c.block.type === 'text').block.text;
      const compactedResumeFinish = compactedResumeChunks.find((c) => c.type === 'finish');
      const sessAfterCompactedResume = compactedResumeFinish.replayState.response.acpSessionId;
      assert.notEqual(sessAfterCompactedResume, sess3, 'Compacted history must not resume stale ACP session');
      assert.match(compactedResumeText, new RegExp(`Session=${sessAfterCompactedResume} Turn=1`));
      assert.match(compactedResumeText, /<conversation_history>/);
      assert.match(compactedResumeText, /POST_RESTART_COMPACTED_SUMMARY/);
    } finally {
      await resumedAdapter.dispose();
    }
  } finally {
    await adapter.dispose();
    fs.rmSync(tmpCache, { recursive: true, force: true });
  }
});

test('AgyAcpAdapter decomposes 6-step tool-loop usage so context occupancy stays ~18% instead of spiking to 100%', async () => {
  const tmpCache = fs.mkdtempSync(path.join(os.tmpdir(), 'agy-usage-test-'));
  const cfg = resolveAgyAcpConfig({
    agyPath: FAKE_AGY_PATH,
    poolSize: 1,
    batchWindowMs: 10,
    cacheDir: tmpCache,
  });
  const adapter = new AgyAcpAdapter({ getConfig: () => cfg });

  try {
    const tools = [
      { name: 'get_left', description: 'Left', parameters: { type: 'object', properties: { q: { type: 'string' } } } },
    ];
    const messages = [
      {
        id: 'u-loop',
        role: 'user',
        content: [{ type: 'text', text: 'Run CALL_SIX_STEP_LOOP' }],
        source: { kind: 'user' },
      },
    ];

    const stepUsages = [];
    let finishKind = 'tool-calls';
    let step = 0;

    while (finishKind === 'tool-calls' && step++ < 10) {
      const chunks = await collectChunks(
        adapter.stream({
          provider: 'agy-acp',
          model: 'gemini-4-carbon',
          sessionId: 'dsh-usage-loop',
          messages,
          tools,
        }),
      );
      const usageChunk = chunks.find((c) => c.type === 'usage');
      assert.ok(usageChunk, `Expected usage chunk on step ${step}`);
      stepUsages.push(usageChunk.usage);

      const finish = chunks.find((c) => c.type === 'finish');
      finishKind = finish.reason.kind;
      const blocks = chunks.filter((c) => c.type === 'block-end').map((c) => c.block);
      messages.push({
        id: `a-loop-${step}`,
        role: 'assistant',
        content: blocks,
        source: { provider: 'agy-acp', model: 'gemini-4-carbon', replayState: finish.replayState },
      });
      if (finishKind === 'tool-calls') {
        for (const tc of blocks.filter((b) => b.type === 'tool-call')) {
          messages.push({
            id: `t-loop-${step}`,
            role: 'tool',
            toolCallId: tc.id,
            content: [{ type: 'text', text: `ok-${step}` }],
            source: { kind: 'tool' },
          });
        }
      }
    }

    assert.equal(finishKind, 'stop');
    assert.equal(stepUsages.length, 6);

    // Every single step (including step 6) keeps prompt pressure ~180k-250k (< 25% of 1M), never 1.12M (100%)
    for (const u of stepUsages) {
      const pressure = u.inputTokens + (u.cacheReadTokens ?? 0) + (u.cacheWriteTokens ?? 0);
      assert.ok(pressure > 150_000 && pressure < 250_000, `Unexpected single-step pressure: ${pressure}`);
    }

    // Total prompt tokens across all 6 steps equals ACP's exact cumulative prompt total (19,200 + 1,099,200 = 1,118,400)
    const sumPrompt = stepUsages.reduce(
      (acc, u) => acc + u.inputTokens + (u.cacheReadTokens ?? 0) + (u.cacheWriteTokens ?? 0),
      0,
    );
    assert.equal(sumPrompt, 1_118_400);
  } finally {
    await adapter.dispose();
    fs.rmSync(tmpCache, { recursive: true, force: true });
  }
});

const user = text => ({ role: 'user', content: [{ type: 'text', text }] });
function assistant(chunks) {
  return {
    id: 'reply', role: 'assistant',
    content: chunks.filter(c => c.type === 'block-end').map(c => c.block),
    source: { replayState: chunks.find(c => c.type === 'finish').replayState },
  };
}

function muxFixture(t, idleTtlMs = 0) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agy-mux-'));
  const cfg = { ...resolveAgyAcpConfig({ agyPath: FAKE_AGY_PATH, cacheDir: dir, workingDirectory: dir }), idleTtlMs };
  const adapter = new AgyAcpAdapter({ getConfig: () => cfg });
  t.after(async () => {
    await adapter.dispose();
    fs.rmSync(dir, { recursive: true, force: true });
  });
  return (sessionId, messages, tools = []) => collectChunks(adapter.stream({
    provider: 'agy-acp', model: 'gemini-4-carbon', sessionId, messages, tools,
  }));
}

test('eight interleaved sessions share one ACP child and isolate permissions, MCP routes, streams and results', { timeout: 12000 }, async t => {
  const turn = muxFixture(t);
  const members = Array.from({ length: 8 }, (_, i) => ({
    session: 'team-' + i,
    tools: ['run_code', 'member_' + i].map(name => ({ name, description: name, parameters: { type: 'object' } })),
    messages: [user('CALL_PTC_MEMBER_' + i)],
  }));
  const calls = await Promise.all(members.map(async m => assistant(await turn(m.session, m.messages, m.tools))));
  const routes = new Set(), pids = new Set(), sessions = new Set();
  await Promise.all(members.map(async (m, i) => {
    const reply = calls[i];
    sessions.add(reply.source.replayState.response.acpSessionId);
    const [call] = reply.content.filter(b => b.type === 'tool-call');
    assert.equal(call.name, 'run_code');
    const args = JSON.parse(call.arguments);
    assert.equal(args.member, String(i));
    assert.equal(args.own, 'allow_once');
    assert.equal(args.foreign, 'reject_once');
    assert.equal(args.unknown, 'reject_once');
    routes.add(args.route);
    pids.add(args.pid);
    const result = { role: 'tool', toolCallId: call.id, content: [{ type: 'text', text: 'reply-' + i }] };
    const final = assistant(await turn(m.session, [...m.messages, reply, result], m.tools));
    assert.deepEqual(final.content, [{ type: 'text', text: `member=${i} result=reply-${i} active=8` }]);
    assert.equal(final.source.replayState.response.acpSessionId, reply.source.replayState.response.acpSessionId);
  }));
  assert.equal(routes.size, 8);
  assert.equal(sessions.size, 8);
  assert.equal(pids.size, 1);
});

test('idle close releases just the idle session while a sibling MCP call waits for DSH', { timeout: 8000 }, async t => {
  const turn = muxFixture(t, 60);
  const tools = ['run_code', 'member_0'].map(name => ({ name, parameters: { type: 'object' } }));
  const start = user('CALL_PTC_MEMBER_0');
  const reply = assistant(await turn('waiting', [start], tools));
  await turn('idle', [user('hello')]);
  await new Promise(resolve => setTimeout(resolve, 240));
  const call = reply.content.find(b => b.type === 'tool-call');
  const final = assistant(await turn('waiting', [start, reply, {
    role: 'tool', toolCallId: call.id, content: [{ type: 'text', text: 'survived' }],
  }], tools));
  assert.equal(final.content[0].text, 'member=0 result=survived active=1');
});

test('process loss restores saved sessions, cold-replays unavailable sessions, and isolates forks', { timeout: 9000 }, async t => {
  const turn = muxFixture(t);
  const first = user('SAVE_FOR_LOAD');
  const saved = assistant(await turn('saved', [first]));
  const other = assistant(await turn('other', [user('hello')]));
  await assert.rejects(turn('crash', [user('EXIT_FIXTURE_PROCESS')]), /exited unexpectedly/);
  const [resumed, rebuilt, fork] = await Promise.all([
    turn('saved', [first, saved, user('continue')]),
    turn('other', [user('hello'), other, user('member report A'), user('member report B'), user('INSPECT_CURRENT_INPUT')]),
    turn('fork', [first, saved, user('forked')]),
  ].map(p => p.then(assistant)));
  const id = msg => msg.source.replayState.response.acpSessionId;
  assert.equal(id(resumed), id(saved));
  assert.notEqual(id(rebuilt), id(other));
  assert.match(rebuilt.content.at(-1).text, /member report A[\s\S]*member report B/);
  assert.doesNotMatch(rebuilt.content.at(-1).text, /hello/);
  assert.notEqual(id(fork), id(saved));
  assert.match(resumed.content.at(-1).text, /Turn=2/);
  assert.ok(resumed.content.every(b => !b.text.includes('historical load event')));
});
