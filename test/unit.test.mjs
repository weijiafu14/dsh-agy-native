import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import {
  AGY_PRE_CHECKPOINT_CONTEXT_CAP,
  AGY_STEP_TYPE_MCP_TOOL,
  AGY_STEP_TYPE_PLANNER_RESPONSE,
  AGY_STEP_TYPE_USER_INPUT,
  AgyModelCatalog,
  DEFAULT_MAX_CONTEXT_WINDOW,
  GEMINI_DUMMY_THOUGHT_SIGNATURE,
  LoopbackMcpBridge,
  McpSessionEndpoint,
  buildCatalogEntries,
  buildFullReplayPromptParts,
  buildIncrementalPromptParts,
  buildMcpToolResultContent,
  clampEffectiveContextWindow,
  classifyInBandAgyApiError,
  computeHistoryPrefixDigest,
  computeToolsFingerprint,
  convertDshMessagesToAgySteps,
  createInFlightTurnUsageState,
  createSanitizedAgyEnv,
  encodeMcpToolName,
  estimateIntermediateStepUsage,
  formatHarnessGuidelines,
  hydrateAgyConversationDb,
  isAmbientContextMessage,
  planSessionSync,
  reconcileFinalStepUsage,
  resolveAgyAcpConfig,
  resolveAgyBinary,
  sanitizeMcpToolArguments,
  splitMessagesForTrajectorySync,
  supportsProcessPerSession,
  truncateToolOutputUtf8,
} from '../dist/core.js';

test('createSanitizedAgyEnv strips ambient API keys to preserve subscription OAuth', () => {
  const dirty = {
    PATH: '/usr/bin',
    HOME: '/Users/demo',
    GEMINI_API_KEY: 'AIzaSyFakeKey',
    GOOGLE_API_KEY: 'AIzaSyFakeKey2',
    AGY_ADC_AUTH: '1',
    DSH_AGY_LOCAL_KEY: 'secret',
  };
  const clean = createSanitizedAgyEnv(dirty);
  assert.equal(clean.PATH, '/usr/bin');
  assert.equal(clean.GEMINI_API_KEY, undefined);
  assert.equal(clean.GOOGLE_API_KEY, undefined);
  assert.equal(clean.AGY_ADC_AUTH, undefined);
  assert.equal(clean.DSH_AGY_LOCAL_KEY, undefined);
});

test('resolveAgyBinary validates explicit path and rejects missing binary', () => {
  assert.throws(
    () => resolveAgyBinary('/nonexistent/path/to/agy'),
    /not found or is not executable/,
  );
});

test('AgyModelCatalog groups families, supports Hydrogen xhigh, and persists cache', () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'agy-cat-test-'));
  try {
    const rawModels = [
      { modelId: 'gemini-4-carbon-medium', name: 'Gemini 4 Carbon (Medium)' },
      { modelId: 'gemini-4-carbon-high', name: 'Gemini 4 Carbon (High)' },
      { modelId: 'gemini-4-carbon-low', name: 'Gemini 4 Carbon (Low)' },
      { modelId: 'gemini-4-hydrogen-high', name: 'Gemini 4 Hydrogen (High)' },
      { modelId: 'gemini-4-hydrogen-xhigh', name: 'Gemini 4 Hydrogen (XHigh)' },
      { modelId: 'gemini-4-hydrogen-max', name: 'Gemini 4 Hydrogen (Max)' },
      { modelId: 'gemini-4-hydrogen-extra', name: 'Gemini 4 Hydrogen (Extra)' },
      { modelId: 'gemini-4-hydrogen-medium', name: 'Gemini 4 Hydrogen (Medium)' },
      { modelId: 'gemini-4-hydrogen-low', name: 'Gemini 4 Hydrogen (Low)' },
      { modelId: 'gemini-pro-orchestrator', name: 'Orchestrator (Carbon)' },
      { modelId: 'gpt-oss-120b-medium', name: 'GPT-OSS 120B (Medium)' },
    ];

    const entries = buildCatalogEntries(rawModels);
    assert.equal(entries.length, 4);

    const hydrogen = entries.find((e) => e.id === 'gemini-4-hydrogen');
    assert.ok(hydrogen);
    assert.equal(hydrogen.name, 'Gemini 4 Hydrogen');
    assert.deepEqual([...hydrogen.effortToRawModelId.keys()], ['low', 'medium', 'high', 'extra', 'xhigh', 'max']);
    assert.equal(hydrogen.defaultEffort, 'high');

    const catalog = new AgyModelCatalog(tmpDir);
    catalog.update(rawModels, { name: 'Google AI Ultra', id: 'ultra' });

    assert.equal(catalog.resolveRawModelId('gemini-4-hydrogen', 'xhigh'), 'gemini-4-hydrogen-xhigh');
    assert.equal(catalog.resolveRawModelId('gemini-4-hydrogen', 'max'), 'gemini-4-hydrogen-max');
    assert.equal(catalog.resolveRawModelId('gemini-4-hydrogen', 'extra'), 'gemini-4-hydrogen-extra');
    assert.equal(catalog.resolveRawModelId('gemini-4-carbon'), 'gemini-4-carbon-medium');
    assert.equal(catalog.resolveRawModelId('gemini-4-carbon-high'), 'gemini-4-carbon-high');
    assert.equal(catalog.resolveRawModelId('gemini-pro-orchestrator'), 'gemini-pro-orchestrator');
    assert.equal(catalog.resolveRawModelId('gpt-oss-120b-medium'), 'gpt-oss-120b-medium');

    const resolvedCarbon = catalog.toResolvedModelInfo('agy-acp', 'gemini-4-carbon');
    assert.equal(resolvedCarbon.systemPromptUpdate, 'in-history');
    assert.equal(resolveAgyAcpConfig({}).maxToolOutputBytes, 0);
    assert.equal(truncateToolOutputUtf8('x'.repeat(250_000), 0).length, 250_000);

    const reloaded = new AgyModelCatalog(tmpDir);
    assert.equal(reloaded.tier?.name, 'Google AI Ultra');
    const infos = reloaded.toLlmModelInfos('agy-acp');
    assert.equal(infos.length, 4);
    assert.deepEqual(infos[0].inputModalities, ['text', 'image']);
    assert.equal(infos[0].badge?.text, 'Ultra');
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

test('sanitizeMcpToolArguments strips undeclared agy toolSummary but keeps declared one', () => {
  const strictSchema = {
    name: 'workspace_read',
    description: 'Read file',
    parameters: {
      type: 'object',
      properties: { file_path: { type: 'string' } },
      additionalProperties: false,
    },
  };
  assert.deepEqual(
    sanitizeMcpToolArguments(
      { file_path: '/tmp/a.txt', toolSummary: 'Read /tmp/a.txt' },
      strictSchema,
    ),
    { file_path: '/tmp/a.txt' },
  );

  const schemaWithSummary = {
    name: 'custom_summary_tool',
    description: 'Has toolSummary',
    parameters: {
      type: 'object',
      properties: { file_path: { type: 'string' }, toolSummary: { type: 'string' } },
    },
  };
  assert.deepEqual(
    sanitizeMcpToolArguments(
      { file_path: '/tmp/a.txt', toolSummary: 'Keep me' },
      schemaWithSummary,
    ),
    { file_path: '/tmp/a.txt', toolSummary: 'Keep me' },
  );
});

test('truncateToolOutputUtf8 never splits multi-byte CJK characters', () => {
  const input = '你好世界🚀'.repeat(200);
  const truncated = truncateToolOutputUtf8(input, 300);
  assert.ok(Buffer.byteLength(truncated, 'utf8') <= 300);
  assert.ok(!truncated.includes('\uFFFD'), 'Should not contain UTF-8 replacement char');
  assert.match(truncated, /Output truncated at 300 bytes/);
});

test('LoopbackMcpBridge enforces token auth and coalesces parallel tool calls into one batch', async () => {
  const bridge = new LoopbackMcpBridge();
  try {
    const endpoint = await bridge.createSessionEndpoint({
      maxToolOutputBytes: 16384,
      batchWindowMs: 30,
    });
    endpoint.setTools([
      {
        name: 'read_a',
        description: 'Read A',
        parameters: { type: 'object', properties: { id: { type: 'string' } } },
      },
      {
        name: 'read_b',
        description: 'Read B',
        parameters: { type: 'object', properties: { id: { type: 'string' } } },
      },
    ]);

    const cfg = endpoint.toAcpMcpServerConfig();

    // Unauthorized request must fail with 401
    const unauth = await fetch(cfg.url, {
      method: 'POST',
      body: JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'ping' }),
    });
    assert.equal(unauth.status, 401);

    const sibling = await bridge.createSessionEndpoint({ maxToolOutputBytes: 16384, batchWindowMs: 0 });
    const wrongRoute = await fetch(sibling.toAcpMcpServerConfig().url, {
      method: 'POST', headers: Object.fromEntries(cfg.headers.map(h => [h.name, h.value])),
      body: JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'ping' }),
    });
    assert.equal(wrongRoute.status, 401);
    sibling.dispose();

    // Send two concurrent tools/call requests within batchWindowMs
    const batches = [];
    endpoint.onNextBatch((batch) => batches.push(batch));

    const headers = Object.fromEntries(cfg.headers.map((h) => [h.name, h.value]));
    const callA = fetch(cfg.url, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        jsonrpc: '2.0',
        id: 10,
        method: 'tools/call',
        params: { name: 'read_a', arguments: { id: '1', toolSummary: 'summary a' } },
      }),
    });
    const callB = fetch(cfg.url, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        jsonrpc: '2.0',
        id: 11,
        method: 'tools/call',
        params: { name: 'read_b', arguments: { id: '2', toolSummary: 'summary b' } },
      }),
    });

    await new Promise((r) => setTimeout(r, 60));
    assert.equal(batches.length, 1, 'Expected both calls coalesced into a single batch');
    assert.equal(batches[0].length, 2);
    const itemA = batches[0].find((c) => c.toolName === 'read_a');
    const itemB = batches[0].find((c) => c.toolName === 'read_b');
    assert.equal(itemA?.argumentsJson, '{"id":"1"}');
    assert.equal(itemB?.argumentsJson, '{"id":"2"}');

    endpoint.resolveToolCall(itemA.toolCallId, { text: 'RES_A' });
    endpoint.resolveToolCall(itemB.toolCallId, { text: 'RES_B', isError: true });

    const [resA, resB] = await Promise.all([callA.then((r) => r.json()), callB.then((r) => r.json())]);
    assert.equal(resA.result.content[0].text, 'RES_A');
    assert.equal(resB.result.content[0].text, 'RES_B');
    assert.equal(resB.result.isError, true);
  } finally {
    await bridge.close();
  }
});

test('planSessionSync preserves warm session on tail appends, resolves pending tools with separate context blocks, and rebuilds on prefix rewrites', async () => {
  const toolsFp = computeToolsFingerprint([{ name: 'bash', description: 'Run', parameters: {} }]);

  const initialMessages = [
    {
      id: 'sys-1',
      role: 'system',
      content: [{ type: 'text', text: 'You are a coding agent powered by the gemini-4-carbon model.\nFollow strict repo guidelines.' }],
      source: { kind: 'system-prompt' },
    },
    {
      id: 'dev-surface-1',
      role: 'developer',
      content: [{ type: 'text', text: 'Todo status: step 1 pending' }],
      source: { kind: 'developer' },
    },
    {
      id: 'user-1',
      role: 'user',
      content: [{ type: 'text', text: 'Inspect project' }],
      source: { kind: 'user' },
    },
  ];
  const initialPrefixDigest = computeHistoryPrefixDigest(initialMessages, initialMessages.length);

  // Assistant emitted a tool call; while tool ran, a background job finished (`tool-jobs`) AND clock snapshot arrived
  const midTurnMessages = [
    ...initialMessages,
    {
      id: 'asst-1',
      role: 'assistant',
      content: [{ type: 'tool-call', id: 'call_1', name: 'bash', arguments: '{"cmd":"ls"}' }],
      source: {
        provider: 'agy-acp',
        model: 'gemini-4-carbon',
        replayState: {
          response: {
            version: 1,
            acpSessionId: 'sess-live-1',
            rawModelId: 'gemini-4-carbon-medium',
            syncedMessageCount: 4,
            toolsFingerprint: toolsFp,
            historyPrefixDigest: initialPrefixDigest,
          },
        },
      },
    },
    {
      id: 'tool-res-1',
      role: 'tool',
      toolCallId: 'call_1',
      content: [
        { type: 'text', text: 'README.md\nsrc/' },
        {
          type: 'image',
          attachment: { attachmentId: 'preview.png', mediaType: 'image/png', bytes: 32, width: 16, height: 16 },
        },
      ],
      source: { kind: 'tool' },
    },
    {
      id: 'job-done-1',
      role: 'user',
      content: [{ type: 'text', text: 'background job bg-1 finished [status: exited(0)]. Output: OK' }],
      source: { kind: 'tool-jobs', form: 'notice' },
    },
  ];

  const midTurnDecision = planSessionSync({
    messages: midTurnMessages,
    activeAcpSessionId: 'sess-live-1',
    lastToolsFingerprint: toolsFp,
    currentToolsFingerprint: toolsFp,
    lastHistoryPrefixDigest: initialPrefixDigest,
    pendingToolCallIds: new Set(['call_1', 'call_2_still_running']),
    allowResumeFromReplay: true,
  });

  // Partial resolution works cleanly without resetting warm session!
  assert.equal(midTurnDecision.kind, 'resolve-pending-tools');
  assert.equal(midTurnDecision.resolutions.length, 1);
  assert.equal(midTurnDecision.resolutions[0].toolCallId, 'call_1');
  assert.deepEqual(midTurnDecision.resolutions[0].trailingNotes, [
    'background job bg-1 finished [status: exited(0)]. Output: OK',
  ]);

  // Multimodal MCP content keeps tool text, tool image, and trailing notice in separate MCP blocks!
  const mcpItems = await buildMcpToolResultContent({
    content: midTurnDecision.resolutions[0].content,
    trailingNotes: midTurnDecision.resolutions[0].trailingNotes,
    loadImage: async (ref) => ({ base64: `B64_${ref.attachmentId}`, mimeType: ref.mediaType }),
  });
  assert.equal(mcpItems.length, 3);
  assert.deepEqual(mcpItems[0], { type: 'text', text: 'README.md\nsrc/' });
  assert.deepEqual(mcpItems[1], { type: 'image', data: 'B64_preview.png', mimeType: 'image/png' });
  assert.match(mcpItems[2].text, /<dsh_context_note>[\s\S]*bg-1 finished/);

  // Turn 2: Assistant finished; tail appends developer update + human message + clock snapshot -> warm-incremental
  const turn1History = [...midTurnMessages];
  const turn1PrefixDigest = computeHistoryPrefixDigest(turn1History, turn1History.length);
  const asstFinal1 = {
    id: 'asst-final-1',
    role: 'assistant',
    content: [{ type: 'text', text: 'Found README.md and src/.' }],
    source: {
      provider: 'agy-acp',
      model: 'gemini-4-carbon',
      replayState: {
        response: {
          version: 1,
          acpSessionId: 'sess-live-1',
          rawModelId: 'gemini-4-carbon-medium',
          syncedMessageCount: turn1History.length + 1,
          toolsFingerprint: toolsFp,
          historyPrefixDigest: turn1PrefixDigest,
        },
      },
    },
  };
  const turn2Messages = [
    ...turn1History,
    asstFinal1,
    {
      id: 'dev-tail-2',
      role: 'developer',
      content: [{ type: 'text', text: 'Todo status: step 1 COMPLETED, step 2 IN PROGRESS' }],
      source: { kind: 'developer' },
    },
    {
      id: 'user-2',
      role: 'user',
      content: [{ type: 'text', text: 'Now read README.md' }],
      source: { kind: 'user' },
    },
    {
      id: 'clock-2',
      role: 'user',
      content: [{ type: 'text', text: '2026-11-26T12:30:00+08:00' }],
      source: { kind: 'time-context', form: 'snapshot' },
    },
  ];

  const turn2Decision = planSessionSync({
    messages: turn2Messages,
    activeAcpSessionId: 'sess-live-1',
    lastToolsFingerprint: toolsFp,
    currentToolsFingerprint: toolsFp,
    lastHistoryPrefixDigest: turn1PrefixDigest,
    pendingToolCallIds: new Set(),
    allowResumeFromReplay: true,
  });

  assert.equal(turn2Decision.kind, 'warm-incremental');
  assert.equal(turn2Decision.tailMessages.length, 3);
  const incParts = await buildIncrementalPromptParts({
    tailMessages: turn2Decision.tailMessages,
    priorSystemPrompt: turn2Decision.priorSystemPrompt,
  });
  const incText = incParts.map((p) => p.text).join('\n');
  assert.ok(incText.indexOf('2026-11-26T12:30:00+08:00') < incText.indexOf('Now read README.md'));

  // Cold replay preserves structured tool calls, tool results, and current input
  const replayParts = await buildFullReplayPromptParts({
    messages: turn2Messages,
  });
  const combinedReplayText = replayParts.map((p) => (p.type === 'text' ? p.text : '')).join('\n');
  assert.match(combinedReplayText, /<tool_call id="call_1" name="bash">/);
  assert.match(combinedReplayText, /<tool_result id="call_1" status="ok">/);
  assert.match(combinedReplayText, /Now read README\.md/);
});

test('formatHarnessGuidelines & buildFullReplayPromptParts strip third-party persona headers and place runtime reminders before human request', async () => {
  const rawSystem = [
    'You are an AI agent powered by DeepSeek Harness.',
    '',
    'You are a coding agent powered by the gemini-3.8-flash model.',
    '',
    'Use the read tool — not shell commands like cat — to inspect text files.',
  ].join('\n');

  const guidelines = formatHarnessGuidelines(rawSystem, true);
  assert.ok(guidelines);
  assert.doesNotMatch(guidelines, /powered by DeepSeek Harness/i);
  assert.doesNotMatch(guidelines, /You are a coding agent/i);
  assert.match(guidelines, /mcp_dsh_<tool_name>/);
  assert.match(guidelines, /brain\//);
  assert.match(guidelines, /Use the read tool/);
  assert.equal(formatHarnessGuidelines(undefined, false), undefined);

  const parts = await buildFullReplayPromptParts({
    systemPrompt: rawSystem,
    hasTools: true,
    messages: [
      {
        id: 'human-1',
        role: 'user',
        content: [{ type: 'text', text: 'Reply with PONG_OK only.' }],
        source: { kind: 'user' },
      },
      {
        id: 'skills-1',
        role: 'user',
        content: [{ type: 'text', text: 'Available skills catalog...' }],
        source: { kind: 'skill-catalog', form: 'catalog' },
      },
      {
        id: 'ctx-1',
        role: 'user',
        content: [{ type: 'text', text: 'Working directory snapshot: /repo' }],
        source: { kind: 'runtime-context', form: 'snapshot' },
      },
    ],
  });

  const fullText = parts.map((p) => (p.type === 'text' ? p.text : '')).join('\n');
  const reminderPos = fullText.indexOf('Available skills catalog...');
  const humanPos = fullText.indexOf('Reply with PONG_OK only.');
  assert.ok(reminderPos > 0);
  assert.ok(humanPos > reminderPos, 'Human request must appear after trailing ambient snapshots');

  // Verify structured `isAmbientContextMessage` classification
  assert.equal(isAmbientContextMessage({ role: 'user', content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' } }), false);
  assert.equal(isAmbientContextMessage({ role: 'user', content: [{ type: 'text', text: 'job finished' }], source: { kind: 'tool-jobs', form: 'notice' } }), false);
  assert.equal(isAmbientContextMessage({ role: 'user', content: [{ type: 'text', text: 'clock' }], source: { kind: 'time-context', form: 'snapshot' } }), true);
  assert.equal(isAmbientContextMessage({ role: 'user', content: [{ type: 'text', text: 'bg' }], source: { kind: 'context-composition', slot: 'background' } }), true);
  assert.equal(isAmbientContextMessage({ role: 'user', content: [{ type: 'text', text: 'event' }], source: { kind: 'context-composition' } }), false);
});

test('workspace guidelines match the registered bridge names, including PTC-only and empty catalogs', async () => {
  const endpoint = new McpSessionEndpoint({
    token: 'fixture', url: 'http://localhost', maxToolOutputBytes: 1024, batchWindowMs: 0, onDispose() {},
  });
  const setTools = names => endpoint.setTools(names.map(name => ({
    name, description: name, parameters: { type: 'object' },
  })));
  const prompt = async () => (await buildFullReplayPromptParts({
    messages: [], bridgeToolNames: endpoint.bridgeToolNames,
  }))[0].text;

  setTools(['issue/search', 'issue_search']);
  const standard = await prompt();
  assert.equal(new Set(endpoint.bridgeToolNames).size, 2);
  for (const name of endpoint.bridgeToolNames) assert.ok(standard.includes(`mcp_dsh_${name}`));
  assert.match(standard, /brain\//);
  assert.doesNotMatch(standard, /mcp_dsh_read|mcp_dsh_run_code/);
  setTools(['run_code']);
  const ptc = await prompt();
  assert.match(ptc, /mcp_dsh_run_code/);
  assert.match(ptc, /exact DSH tool names in the supplied SDK/);
  assert.match(ptc, /ambient MCP servers/);
  assert.match(ptc, /brain\//);
  assert.doesNotMatch(ptc, /issue_search/);
  setTools([]);
  const empty = await prompt();
  assert.match(empty, /No DSH tools/);
  assert.match(empty, /brain\//);
  assert.doesNotMatch(empty, /mcp_dsh_\*/);
});

test('supportsProcessPerSession detects --process-per-session flag and falls back on older binaries', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agy-help-probe-'));
  const modernBin = path.join(dir, 'agy-modern');
  const legacyBin = path.join(dir, 'agy-legacy');
  fs.writeFileSync(
    modernBin,
    '#!/usr/bin/env node\nconsole.log("  --process-per-session  Serve each session from an isolated worker process");\n',
    { mode: 0o755 },
  );
  fs.writeFileSync(
    legacyBin,
    '#!/usr/bin/env node\nconsole.log("  --stdio  Use stdio JSON-RPC transport");\n',
    { mode: 0o755 },
  );
  try {
    assert.equal(await supportsProcessPerSession(modernBin), true);
    assert.equal(await supportsProcessPerSession(legacyBin), false);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('an existing literal MCP name is preserved when an encoded collision needs the same suffix', () => {
  const original = 'issue.search';
  const base = 'issue_search';
  const literal = encodeMcpToolName(original, new Set([base]));
  const used = new Set([base, literal]);
  const encoded = encodeMcpToolName(original, used);
  assert.equal(used.size, 3);
  assert.ok(used.has(encoded));
  assert.notEqual(encoded, literal);
});

for (const [preset, toolNames] of Object.entries({
  standard: ['read', 'mcp__sample-server__list_items'],
  ptc: ['run_code'],
  minimal: ['read', 'bash'],
  cordis: ['cordis', 'read', 'plugin_manager'],
  empty: [],
  custom: ['evaluate.workflow', 'evaluate_workflow', 'mcp__x-y__fetch', 'a'.repeat(70) + 'x', 'a'.repeat(70) + 'y'],
})) {
  test(`${preset} routes exactly the tools projected by the request`, () => {
    const endpoint = new McpSessionEndpoint({
      token: 'fixture', url: 'http://localhost', maxToolOutputBytes: 1024, batchWindowMs: 0, onDispose() {},
    });
    endpoint.setTools(toolNames.map(name => ({ name, description: name, parameters: { type: 'object' } })));
    const names = endpoint.bridgeToolNames;
    assert.equal(new Set(names).size, toolNames.length);
    const guidelines = formatHarnessGuidelines(undefined, names);
    const advertised = guidelines.match(/Direct calls available in this request: (.*?)\./);
    assert.deepEqual(advertised ? advertised[1].split(', ') : [], names.map(name => `mcp_dsh_${name}`));
    for (const name of names) {
      assert.match(name, /^[a-zA-Z0-9_-]{1,64}$/);
      assert.equal(endpoint.isAllowedPermissionRequest(`mcp_dsh_${name}`), true);
      assert.equal(endpoint.isAllowedPermissionRequest('call_mcp_tool', { ServerName: 'dsh', ToolName: name }), true);
      assert.equal(endpoint.isAllowedPermissionRequest('call_mcp_tool', { ServerName: 'external', ToolName: name }), false);
    }
    endpoint.setTools([]);
    for (const name of names) assert.equal(endpoint.isAllowedPermissionRequest(`mcp_dsh_${name}`), false);
  });
}

test('ACP allows active eager and dsh dispatcher calls and rejects other requests', () => {
  const endpoint = new McpSessionEndpoint({
    token: 'fixture', url: 'http://localhost', maxToolOutputBytes: 1024, batchWindowMs: 0, onDispose() {},
  });
  endpoint.setTools([{ name: 'run_code', parameters: { type: 'object' }, description: 'PTC' }]);
  assert.equal(endpoint.isAllowedPermissionRequest('mcp_dsh_run_code', { toolSummary: 'Run program' }), true);
  assert.equal(endpoint.isAllowedPermissionRequest('call_mcp_tool', { ServerName: 'dsh', ToolName: 'run_code' }), true);
  assert.equal(endpoint.isAllowedPermissionRequest('call_mcp_tool', { ServerName: 'external', ToolName: 'run_code' }), false);
  assert.equal(endpoint.isAllowedPermissionRequest('run_code'), false);
  assert.equal(endpoint.isAllowedPermissionRequest('mcp_dsh_unknown'), false);
  endpoint.setTools([]);
  assert.equal(endpoint.isAllowedPermissionRequest('mcp_dsh_run_code'), false);
});

test('classifyInBandAgyApiError detects standalone agy 429/403 retry banners without false positives in prose', () => {
  const err429 = classifyInBandAgyApiError(
    '\n⚠ API error (attempt 1): RESOURCE_EXHAUSTED (code 429): Resource has been exhausted (e.g. check quota).\n',
  );
  assert.ok(err429);
  assert.equal(err429.code, 'QUOTA');

  const errAuth = classifyInBandAgyApiError(
    '\n⚠ API error (attempt 1): UNAUTHENTICATED (code 401): Token expired.\n',
  );
  assert.ok(errAuth);
  assert.equal(errAuth.code, 'INVALID_CREDENTIAL');

  // Prose discussion mentioning the warning in the middle of text must NOT trigger
  const prose = classifyInBandAgyApiError(
    'Earlier we saw `⚠ API error (attempt 1): RESOURCE_EXHAUSTED (code 429): ...` in the logs.',
  );
  assert.equal(prose, undefined);
});

test('computeHistoryPrefixDigest strips model persona headers while detecting any genuine prefix mutation', () => {
  const base = [
    {
      id: 'sys-0',
      role: 'system',
      content: [{ type: 'text', text: 'You are an AI agent powered by DeepSeek Harness.\n\nYou are a coding agent powered by the gemini-4-carbon model.\n\nSystem v1' }],
      source: { kind: 'system-prompt' },
    },
    { id: 'dev-1', role: 'developer', content: [{ type: 'text', text: 'Task: step 1' }], source: { kind: 'developer' } },
    { role: 'user', content: [{ type: 'text', text: 'Hello repo' }] },
    {
      id: 'a-1',
      role: 'assistant',
      content: [{ type: 'tool-call', id: 'tc-1', name: 'read', arguments: '{"file":"a.ts"}' }],
      source: { provider: 'agy-acp', model: 'gemini-4-carbon' },
    },
    {
      id: 't-1',
      role: 'tool',
      toolCallId: 'tc-1',
      content: [{ type: 'text', text: 'export const x = 1;' }],
      source: { kind: 'tool' },
    },
  ];

  // Switching models rewrites the third-party model-name persona header, or promoting RequestUserInput to durable Message with id: must keep digest 100% identical!
  const modelSwitchedVariant = [
    {
      id: 'sys-0',
      role: 'system',
      content: [{ type: 'text', text: 'You are an AI agent powered by DeepSeek Harness.\n\nYou are a coding agent powered by the gemini-4-neon model.\n\nSystem v1' }],
      source: { kind: 'system-prompt' },
    },
    base[1],
    { id: 'u-1', role: 'user', content: [{ type: 'text', text: 'Hello repo' }], source: { kind: 'user' } },
    base[3],
    base[4],
  ];
  assert.equal(computeHistoryPrefixDigest(base), computeHistoryPrefixDigest(modelSwitchedVariant));

  // Genuine system instruction rewrite changes digest
  const rewrittenSystem = [
    { ...base[0], content: [{ type: 'text', text: 'System v2 (rewritten)' }] },
    ...base.slice(1),
  ];
  assert.notEqual(computeHistoryPrefixDigest(base), computeHistoryPrefixDigest(rewrittenSystem));

  // Genuine developer surface rewrite before anchor changes digest
  const rewrittenDev = [
    base[0],
    { ...base[1], content: [{ type: 'text', text: 'Task: step 1 REWRITTEN' }] },
    ...base.slice(2),
  ];
  assert.notEqual(computeHistoryPrefixDigest(base), computeHistoryPrefixDigest(rewrittenDev));

  // Tool result pruning (e.g. dsh-compaction-tool-result-pruner) changes digest
  const prunedTool = [...base.slice(0, 4), { ...base[4], content: [{ type: 'text', text: '[... tool result pruned ...]' }] }];
  assert.notEqual(computeHistoryPrefixDigest(base), computeHistoryPrefixDigest(prunedTool));

  // Tool error flag flip changes digest
  const erroredTool = [...base.slice(0, 4), { ...base[4], isError: true }];
  assert.notEqual(computeHistoryPrefixDigest(base), computeHistoryPrefixDigest(erroredTool));

  // Image offload (e.g. dsh-compaction-image-offload) changes digest
  const withImage = [
    {
      role: 'user',
      content: [
        { type: 'text', text: 'attached' },
        {
          type: 'image',
          attachment: { attachmentId: 'img-1', mediaType: 'image/png', bytes: 64, width: 10, height: 10 },
        },
      ],
    },
  ];
  const offloadedImage = [
    {
      role: 'user',
      content: [
        { type: 'text', text: 'attached' },
        {
          type: 'image',
          offloaded: true,
          attachment: { attachmentId: 'img-1', mediaType: 'image/png', bytes: 64, width: 10, height: 10 },
        },
      ],
    },
  ];
  assert.notEqual(computeHistoryPrefixDigest(withImage), computeHistoryPrefixDigest(offloadedImage));
});

test('planSessionSync detects compacted or replaced prefix before anchorIdx while keeping tail appends warm', async () => {
  const toolsFp = computeToolsFingerprint([{ name: 'read', description: 'Read', parameters: {} }]);

  const sysBefore = {
    id: 'sys-1',
    role: 'system',
    content: [{ type: 'text', text: 'You are a coding agent powered by the gemini-4-carbon model.\nBase rules v1' }],
    source: { kind: 'system-prompt' },
  };
  const devBefore = {
    id: 'dev-1',
    role: 'developer',
    content: [{ type: 'text', text: 'Context OS: plan v1' }],
    source: { kind: 'developer' },
  };
  const u1 = { id: 'u-1', role: 'user', content: [{ type: 'text', text: 'Read file A' }], source: { kind: 'user' } };
  const notice1 = { id: 'n-1', role: 'user', content: [{ type: 'text', text: 'Working dir: /repo' }], source: { kind: 'runtime-context', form: 'snapshot' } };
  const a1 = {
    id: 'a-1',
    role: 'assistant',
    content: [{ type: 'tool-call', id: 'tc-1', name: 'read', arguments: '{"path":"a.ts"}' }],
    source: { provider: 'agy-acp', model: 'gemini-4-carbon' },
  };
  const t1 = {
    id: 't-1',
    role: 'tool',
    toolCallId: 'tc-1',
    content: [{ type: 'text', text: 'const a = 1;' }],
    source: { kind: 'tool' },
  };
  const u2 = { id: 'u-2', role: 'user', content: [{ type: 'text', text: 'Summarize A' }], source: { kind: 'user' } };

  const prefixBeforeAnchor = [sysBefore, devBefore, u1, notice1, a1, t1, u2];
  const savedPrefixDigest = computeHistoryPrefixDigest(prefixBeforeAnchor, prefixBeforeAnchor.length);

  const anchor = {
    id: 'a-anchor',
    role: 'assistant',
    content: [{ type: 'text', text: 'File A exports constant a = 1.' }],
    source: {
      provider: 'agy-acp',
      model: 'gemini-4-carbon',
      replayState: {
        response: {
          version: 1,
          acpSessionId: 'sess-1',
          rawModelId: 'gemini-4-carbon-medium',
          syncedMessageCount: prefixBeforeAnchor.length + 1,
          toolsFingerprint: toolsFp,
          historyPrefixDigest: savedPrefixDigest,
        },
      },
    },
  };
  const u3 = { id: 'u-3', role: 'user', content: [{ type: 'text', text: 'Next step?' }], source: { kind: 'user' } };

  // 1. Range compaction: replace u1/notice1/a1/t1 before anchorIdx with a summary node -> cold-full-replay
  const compactedSummary = {
    id: 'summary-1',
    role: 'user',
    content: [{ type: 'text', text: '[Compressed conversation section] User inspected a.ts (a = 1).' }],
    source: { kind: 'compact-checkpoint' },
  };
  const compactedWarm = planSessionSync({
    messages: [sysBefore, devBefore, compactedSummary, u2, anchor, u3],
    activeAcpSessionId: 'sess-1',
    lastToolsFingerprint: toolsFp,
    currentToolsFingerprint: toolsFp,
    lastHistoryPrefixDigest: savedPrefixDigest,
    pendingToolCallIds: new Set(),
    allowResumeFromReplay: true,
  });
  assert.deepEqual(compactedWarm, {
    kind: 'cold-full-replay',
    reason: 'history-prefix-mutated-or-compacted',
  });

  // Compaction is also detected on resume-incremental
  const compactedResume = planSessionSync({
    messages: [sysBefore, devBefore, compactedSummary, u2, anchor, u3],
    activeAcpSessionId: undefined,
    currentToolsFingerprint: toolsFp,
    pendingToolCallIds: new Set(),
    allowResumeFromReplay: true,
  });
  assert.deepEqual(compactedResume, {
    kind: 'cold-full-replay',
    reason: 'history-prefix-mutated-or-compacted',
  });

  // 2. In-place replacement of developer or system before anchorIdx -> clean cold-full-replay (no silent drift!)
  const devMutated = {
    id: 'dev-1',
    role: 'developer',
    content: [{ type: 'text', text: 'Context OS: plan v2 (replaced in-place)' }],
    source: { kind: 'developer' },
  };
  const replacedPrefixDecision = planSessionSync({
    messages: [sysBefore, devMutated, u1, notice1, a1, t1, u2, anchor, u3],
    activeAcpSessionId: 'sess-1',
    lastToolsFingerprint: toolsFp,
    currentToolsFingerprint: toolsFp,
    lastHistoryPrefixDigest: savedPrefixDigest,
    pendingToolCallIds: new Set(),
    allowResumeFromReplay: true,
  });
  assert.deepEqual(replacedPrefixDecision, {
    kind: 'cold-full-replay',
    reason: 'history-prefix-mutated-or-compacted',
  });

  // 3. In-history tail appends (system prompt update, runtime snapshot, user message) + model-persona switch stay warm-incremental!
  const sysModelSwitched = {
    ...sysBefore,
    content: [{ type: 'text', text: 'You are a coding agent powered by the gemini-4-neon model.\nBase rules v1' }],
  };
  const sysTailPlanMode = {
    id: 'sys-2',
    role: 'system',
    content: [{ type: 'text', text: 'You are a coding agent powered by the gemini-4-neon model.\nBase rules v1\nPlan mode active.' }],
    source: { kind: 'system-prompt' },
  };
  const noticeAfter = {
    id: 'n-2',
    role: 'user',
    content: [{ type: 'text', text: 'Clock: 12:05' }],
    source: { kind: 'time-context', form: 'snapshot' },
  };
  const normalDecision = planSessionSync({
    messages: [sysModelSwitched, devBefore, u1, notice1, a1, t1, u2, anchor, sysTailPlanMode, u3, noticeAfter],
    activeAcpSessionId: 'sess-1',
    lastToolsFingerprint: toolsFp,
    currentToolsFingerprint: toolsFp,
    lastHistoryPrefixDigest: savedPrefixDigest,
    pendingToolCallIds: new Set(),
    allowResumeFromReplay: true,
  });
  assert.equal(normalDecision.kind, 'warm-incremental');
  assert.equal(normalDecision.tailMessages.length, 3);
  const warmParts = await buildIncrementalPromptParts({
    tailMessages: normalDecision.tailMessages,
    priorSystemPrompt: normalDecision.priorSystemPrompt,
  });
  const warmJoined = warmParts.map((p) => p.text).join('\n');
  assert.match(warmJoined, /<workspace_guidelines_update>[\s\S]*Plan mode active/);
  assert.ok(warmJoined.indexOf('Clock: 12:05') < warmJoined.indexOf('Next step?'));

  // 4. Warm rewind/fork to an earlier anchor triggers cold replay when slot digest is ahead of anchor
  const laterSlotDigest = computeHistoryPrefixDigest([...prefixBeforeAnchor, anchor, u3]);
  const rewindDecision = planSessionSync({
    messages: [...prefixBeforeAnchor, anchor, u3],
    activeAcpSessionId: 'sess-1',
    lastToolsFingerprint: toolsFp,
    currentToolsFingerprint: toolsFp,
    lastHistoryPrefixDigest: laterSlotDigest,
    pendingToolCallIds: new Set(),
    allowResumeFromReplay: true,
  });
  assert.deepEqual(rewindDecision, {
    kind: 'cold-full-replay',
    reason: 'history-prefix-mutated-or-compacted',
  });

  // 5. Stale slot digest must not poison resume-incremental when resuming a restarted session
  const resumeWithStaleSlotState = planSessionSync({
    messages: [...prefixBeforeAnchor, anchor, u3],
    activeAcpSessionId: undefined,
    lastToolsFingerprint: 'stale-tools-fp',
    currentToolsFingerprint: toolsFp,
    lastHistoryPrefixDigest: 'stale-prefix-digest',
    pendingToolCallIds: new Set(),
    allowResumeFromReplay: true,
  });
  assert.equal(resumeWithStaleSlotState.kind, 'resume-incremental');
  assert.equal(resumeWithStaleSlotState.resumeAcpSessionId, 'sess-1');
});

test('planSessionSync detects mid-turn compaction during resolve-pending-tools and stays backward compatible with legacy envelopes', () => {
  const toolsFp = computeToolsFingerprint([{ name: 'compress', description: 'Compress', parameters: {} }]);

  const dev = {
    id: 'dev-1',
    role: 'developer',
    content: [{ type: 'text', text: 'Todo: compress old turns' }],
    source: { kind: 'developer' },
  };
  const u1 = { id: 'u-1', role: 'user', content: [{ type: 'text', text: 'Old turn 1' }], source: { kind: 'user' } };
  const a1 = { id: 'a-1', role: 'assistant', content: [{ type: 'text', text: 'Old answer 1' }], source: { provider: 'agy-acp', model: 'gemini-4-carbon' } };
  const u2 = { id: 'u-2', role: 'user', content: [{ type: 'text', text: 'Now compress and inspect' }], source: { kind: 'user' } };

  const prefixBeforeToolTurn = [dev, u1, a1, u2];
  const savedPrefixDigest = computeHistoryPrefixDigest(prefixBeforeToolTurn, prefixBeforeToolTurn.length);

  const asstTool = {
    id: 'a-tool',
    role: 'assistant',
    content: [{ type: 'tool-call', id: 'tc-compress', name: 'compress', arguments: '{}' }],
    source: {
      provider: 'agy-acp',
      model: 'gemini-4-carbon',
      replayState: {
        response: {
          version: 1,
          acpSessionId: 'sess-mid-1',
          rawModelId: 'gemini-4-carbon-medium',
          syncedMessageCount: prefixBeforeToolTurn.length + 1,
          toolsFingerprint: toolsFp,
          historyPrefixDigest: savedPrefixDigest,
        },
      },
    },
  };
  const toolRes = {
    id: 't-res',
    role: 'tool',
    toolCallId: 'tc-compress',
    content: [{ type: 'text', text: 'Compressed range m00001-m00002.' }],
    source: { kind: 'tool' },
  };

  // Intact prefix + trailing developer/user additions after toolTurnStartIdx resolve pending tools warmly
  const devTail = { id: 'dev-2', role: 'developer', content: [{ type: 'text', text: 'Todo: step 2 in progress' }], source: { kind: 'developer' } };
  const tailMidTurn = planSessionSync({
    messages: [dev, u1, a1, u2, asstTool, toolRes, devTail],
    activeAcpSessionId: 'sess-mid-1',
    lastToolsFingerprint: toolsFp,
    currentToolsFingerprint: toolsFp,
    lastHistoryPrefixDigest: savedPrefixDigest,
    pendingToolCallIds: new Set(['tc-compress']),
    allowResumeFromReplay: true,
  });
  assert.equal(tailMidTurn.kind, 'resolve-pending-tools');
  assert.deepEqual(tailMidTurn.resolutions[0].trailingNotes, ['Todo: step 2 in progress']);

  // Mid-turn range compaction replaces [u1, a1] with summary node while tc-compress is pending -> cold-full-replay
  const summaryNode = {
    id: 'sum-mid',
    role: 'user',
    content: [{ type: 'text', text: '[Compressed conversation section] Old turn 1 summary' }],
    source: { kind: 'user' },
  };
  const compactedMidTurn = planSessionSync({
    messages: [dev, summaryNode, u2, asstTool, toolRes],
    activeAcpSessionId: 'sess-mid-1',
    lastToolsFingerprint: toolsFp,
    currentToolsFingerprint: toolsFp,
    lastHistoryPrefixDigest: savedPrefixDigest,
    pendingToolCallIds: new Set(['tc-compress']),
    allowResumeFromReplay: true,
  });
  assert.deepEqual(compactedMidTurn, {
    kind: 'cold-full-replay',
    reason: 'history-prefix-mutated-or-compacted',
  });

  // Backward compatibility: legacy ReplayEnvelope without historyPrefixDigest must NOT force cold replay
  const legacyAnchor = {
    id: 'a-legacy',
    role: 'assistant',
    content: [{ type: 'text', text: 'Legacy answer' }],
    source: {
      provider: 'agy-acp',
      model: 'gemini-4-carbon',
      replayState: {
        response: {
          version: 1,
          acpSessionId: 'sess-legacy',
          rawModelId: 'gemini-4-carbon-medium',
          syncedMessageCount: 2,
          toolsFingerprint: toolsFp,
        },
      },
    },
  };
  const legacyWarm = planSessionSync({
    messages: [u1, legacyAnchor, u2],
    activeAcpSessionId: 'sess-legacy',
    lastToolsFingerprint: toolsFp,
    currentToolsFingerprint: toolsFp,
    pendingToolCallIds: new Set(),
    allowResumeFromReplay: true,
  });
  assert.equal(legacyWarm.kind, 'warm-incremental');

  const legacyResume = planSessionSync({
    messages: [u1, legacyAnchor, u2],
    activeAcpSessionId: undefined,
    currentToolsFingerprint: toolsFp,
    pendingToolCallIds: new Set(),
    allowResumeFromReplay: true,
  });
  assert.equal(legacyResume.kind, 'resume-incremental');
  assert.equal(legacyResume.resumeAcpSessionId, 'sess-legacy');
});

test('reconcileFinalStepUsage preserves exact cumulative billing while keeping final-step context occupancy single-step', () => {
  // Warm 6-step turn where prior turn established cachedPromptBaseline = 175_000
  const turnState = createInFlightTurnUsageState();
  let baseline = 175_000;
  const emitted = [];

  for (const used of [180_000, 182_000, 184_000, 186_000, 188_000]) {
    turnState.toolBatchCount += 1;
    const { usage, nextCachedPromptBaseline } = estimateIntermediateStepUsage({
      stepUsedTokens: used,
      emittedOutputChars: 800,
      cachedPromptBaseline: baseline,
    });
    baseline = nextCachedPromptBaseline;
    turnState.emittedInput += usage.inputTokens;
    turnState.emittedOutput += usage.outputTokens;
    turnState.emittedCacheRead += usage.cacheReadTokens ?? 0;
    turnState.emittedSteps += 1;
    emitted.push(usage);
  }

  turnState.lastStepUsed = 190_000;
  const { usage: finalUsage } = reconcileFinalStepUsage(
    {
      inputTokens: 19_223,
      outputTokens: 3_745,
      cachedReadTokens: 1_089_736,
      cachedWriteTokens: 0,
      thoughtTokens: 2_404,
      totalTokens: 1_112_704,
    },
    turnState,
  );

  assert.ok(finalUsage);
  const all = [...emitted, finalUsage];
  const totalInput = all.reduce((s, u) => s + u.inputTokens, 0);
  const totalCacheRead = all.reduce((s, u) => s + (u.cacheReadTokens ?? 0), 0);
  const totalPrompt = totalInput + totalCacheRead;
  const totalOut = all.reduce((s, u) => s + u.outputTokens, 0);
  assert.equal(totalInput, 19_223, 'Cumulative uncached input must match ACP exact total without 0.8x inflation');
  assert.equal(totalCacheRead, 1_089_736, 'Cumulative cache read must match ACP exact total without clamping loss');
  assert.equal(totalPrompt, 19_223 + 1_089_736);
  assert.equal(totalOut, 3_745);

  const finalPressure = finalUsage.inputTokens + (finalUsage.cacheReadTokens ?? 0);
  assert.ok(finalPressure < 220_000, `Final step pressure (${finalPressure}) must stay ~19%, never 1.1M`);

  // Fallback when an older binary emits zero intermediate usage_update frames during tool batches
  const noUpdatesState = createInFlightTurnUsageState();
  noUpdatesState.toolBatchCount = 5;
  const { usage: fallbackUsage } = reconcileFinalStepUsage(
    {
      inputTokens: 18_000,
      outputTokens: 3_600,
      cachedReadTokens: 1_080_000,
      totalTokens: 1_101_600,
    },
    noUpdatesState,
  );
  assert.ok(fallbackUsage);
  assert.equal(fallbackUsage.inputTokens + (fallbackUsage.cacheReadTokens ?? 0), 183_000);
});

test('cold replay loads only current images and keeps consecutive teammate reports after the last tool result', async () => {
  const text = value => ({ type: 'text', text: value });
  const image = id => ({ type: 'image', attachment: { attachmentId: id, width: 10, height: 20 } });
  const read = [];
  const parts = await buildFullReplayPromptParts({
    messages: [
      { role: 'user', content: [image('old.png')] },
      { role: 'assistant', content: [text('Checking')] },
      { role: 'tool', toolCallId: 'a', content: [text('Checked'), image('tool.png')] },
      { role: 'user', content: [text('Programmer report')] },
      { role: 'developer', content: [text('Current team context')] },
      { role: 'user', content: [text('Reviewer report'), image('current.png')] },
      { role: 'user', content: [text('Coordinator report')] },
    ],
    loadImage: async ref => {
      read.push(ref.attachmentId);
      return { mimeType: 'image/png', base64: 'eA==' };
    },
  });
  assert.deepEqual(read, ['current.png']);
  assert.equal(parts.filter(p => p.type === 'image').length, 1);
  const [history, current] = parts.map(p => p.text ?? '[pixels]').join('\n').split('<current_input>');
  assert.match(history, /old\.png/);
  assert.match(history, /tool\.png/);
  assert.doesNotMatch(history, /report|pixels/);
  assert.match(current, /Programmer report[\s\S]*Reviewer report[\s\S]*\[pixels\][\s\S]*Coordinator report/);
  assert.equal(current.match(/<user_message>/g)?.length, 3);
});

test('multi-step cold-start and heavy-reasoning turns satisfy DSH token-meter invariants and preserve exact cache-hit ratio', () => {
  const turnState = createInFlightTurnUsageState();
  let baseline = 0;
  const steps = [];

  for (const [used, outChars, thoughtChars] of [
    [22_400, 1_600, 1_200],
    [30_500, 2_000, 1_600],
    [45_600, 2_400, 2_000],
  ]) {
    turnState.toolBatchCount += 1;
    const { usage, nextCachedPromptBaseline } = estimateIntermediateStepUsage({
      stepUsedTokens: used,
      emittedOutputChars: outChars,
      emittedThoughtChars: thoughtChars,
      cachedPromptBaseline: baseline,
    });
    baseline = nextCachedPromptBaseline;
    turnState.emittedInput += usage.inputTokens;
    turnState.emittedOutput += usage.outputTokens;
    turnState.emittedCacheRead += usage.cacheReadTokens ?? 0;
    turnState.emittedCacheWrite += usage.cacheWriteTokens ?? 0;
    turnState.emittedReasoning += usage.reasoningTokens ?? 0;
    turnState.emittedSteps += 1;
    steps.push(usage);
  }

  turnState.lastStepUsed = 55_300;
  turnState.finalStepObservedUsed = 55_300;
  const { usage: finalStep, nextCachedPromptBaseline: nextTurnBaseline } = reconcileFinalStepUsage(
    {
      inputTokens: 56_500,
      outputTokens: 1_800,
      cachedReadTokens: 95_200,
      cachedWriteTokens: 0,
      thoughtTokens: 1_450,
      totalTokens: 153_500,
    },
    turnState,
  );

  assert.ok(finalStep);
  steps.push(finalStep);

  for (const [idx, u] of steps.entries()) {
    assert.equal(typeof u.cacheReadTokens, 'number', `Step ${idx + 1} must define cacheReadTokens`);
    assert.equal(typeof u.cacheWriteTokens, 'number', `Step ${idx + 1} must define cacheWriteTokens`);
    assert.equal(typeof u.reasoningTokens, 'number', `Step ${idx + 1} must define reasoningTokens`);
    assert.ok(
      u.reasoningTokens <= u.outputTokens,
      `Step ${idx + 1} reasoningTokens (${u.reasoningTokens}) must not exceed outputTokens (${u.outputTokens})`,
    );
    assert.equal(
      u.totalTokens,
      u.inputTokens + u.cacheReadTokens + u.cacheWriteTokens + u.outputTokens,
      `Step ${idx + 1} totalTokens must equal knownPrompt + outputTokens`,
    );
  }

  assert.equal(steps.reduce((s, u) => s + u.inputTokens, 0), 56_500);
  assert.equal(steps.reduce((s, u) => s + u.cacheReadTokens, 0), 95_200);
  assert.equal(steps.reduce((s, u) => s + u.outputTokens, 0), 1_800);
  assert.ok(nextTurnBaseline >= 54_000, `Next turn baseline (${nextTurnBaseline}) should retain full final prompt`);
});

test('AgyModelCatalog caps Gemini 1M contextWindow to 500k by default so DSH compaction preempts agy silent CHECKPOINT', () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'agy-cap-test-'));
  try {
    assert.equal(DEFAULT_MAX_CONTEXT_WINDOW, 500_000);
    assert.equal(AGY_PRE_CHECKPOINT_CONTEXT_CAP, 500_000);
    assert.equal(clampEffectiveContextWindow(1_048_576, 500_000), 500_000);
    assert.equal(clampEffectiveContextWindow(200_000, 500_000), 200_000);
    assert.equal(clampEffectiveContextWindow(1_048_576, 0), 1_048_576);

    let dynamicCap = 500_000;
    const catalog = new AgyModelCatalog({
      cacheDir: tmpDir,
      getMaxContextWindow: () => dynamicCap,
    });
    catalog.update([
      { modelId: 'gemini-4-carbon-medium', name: 'Gemini 4 Carbon (Medium)' },
      { modelId: 'gemini-4-carbon-high', name: 'Gemini 4 Carbon (High)' },
      { modelId: 'claude-sonnet-5.5', name: 'Claude Sonnet 5.5' },
    ]);
    catalog.recordContextWindow('gemini-4-carbon-medium', 1_048_576);

    assert.equal(catalog.toResolvedModelInfo('agy-acp', 'gemini-4-carbon').context.contextWindow, 500_000);
    assert.equal(catalog.toResolvedModelInfo('agy-acp', 'claude-sonnet-5.5').context.contextWindow, 200_000);

    dynamicCap = 0; // Uncapped
    assert.equal(catalog.toResolvedModelInfo('agy-acp', 'gemini-4-carbon').context.contextWindow, 1_048_576);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

test('estimateIntermediateStepUsage & reconcileFinalStepUsage handle agy mid-turn CHECKPOINT cliff and prevent 0-pressure collapse', () => {
  const turnState = createInFlightTurnUsageState();
  // Step 1 hits 495k context before agy triggers hidden 500k CHECKPOINT
  turnState.toolBatchCount += 1;
  const s1 = estimateIntermediateStepUsage({
    stepUsedTokens: 495_000,
    emittedOutputChars: 1_200,
    cachedPromptBaseline: 480_000,
    authoritative: true,
  });
  assert.equal(s1.compacted, false);
  turnState.emittedInput += s1.usage.inputTokens;
  turnState.emittedOutput += s1.usage.outputTokens;
  turnState.emittedCacheRead += s1.usage.cacheReadTokens ?? 0;
  turnState.emittedSteps += 1;

  // Step 2 drops sharply to 85k after internal CHECKPOINT summarization
  turnState.toolBatchCount += 1;
  const s2 = estimateIntermediateStepUsage({
    stepUsedTokens: 85_000,
    emittedOutputChars: 800,
    cachedPromptBaseline: s1.nextCachedPromptBaseline,
    authoritative: true,
  });
  assert.equal(s2.compacted, true);
  assert.equal(s2.usage.cacheReadTokens, 0);
  turnState.emittedInput += s2.usage.inputTokens;
  turnState.emittedOutput += s2.usage.outputTokens;
  turnState.emittedSteps += 1;

  // Final step runs at 90k observed context, even though cumulative turn prompt includes hidden 500k checkpoint call
  turnState.lastStepUsed = 90_000;
  turnState.finalStepObservedUsed = 90_000;
  const { usage: finalUsage, nextCachedPromptBaseline } = reconcileFinalStepUsage(
    {
      inputTokens: 210_000,
      outputTokens: 3_000,
      cachedReadTokens: 960_000,
      cachedWriteTokens: 0,
      totalTokens: 1_173_000,
    },
    turnState,
  );
  assert.ok(finalUsage);
  const finalPressure = finalUsage.inputTokens + (finalUsage.cacheReadTokens ?? 0);
  assert.ok(finalPressure >= 80_000 && finalPressure <= 95_000, `Final pressure (${finalPressure}) must match post-checkpoint ~88k`);
  assert.ok(nextCachedPromptBaseline >= 80_000 && nextCachedPromptBaseline <= 95_000);
});

test('planSessionSync reuses warm and resumable ACP sessions for purpose=compaction with tools intact', () => {
  const toolsFp = computeToolsFingerprint([{ name: 'read', description: 'Read file', parameters: {} }]);
  const u1 = { id: 'u1', role: 'user', content: [{ type: 'text', text: 'First task' }], source: { kind: 'user' } };
  const a1Digest = computeHistoryPrefixDigest([u1]);
  const a1 = {
    id: 'a1',
    role: 'assistant',
    content: [{ type: 'text', text: 'Completed first task anchor' }],
    source: {
      provider: 'agy-acp',
      model: 'gemini-4-carbon',
      replayState: {
        response: {
          version: 1,
          dshSessionId: 'dsh-s1',
          acpSessionId: 'acp-s1',
          rawModelId: 'gemini-4-carbon-medium',
          syncedMessageCount: 2,
          toolsFingerprint: toolsFp,
          historyPrefixDigest: a1Digest,
        },
      },
    },
  };
  const u2 = { id: 'u2', role: 'user', content: [{ type: 'text', text: 'Retained tail turn' }], source: { kind: 'user' } };
  const a2 = { id: 'a2', role: 'assistant', content: [{ type: 'text', text: 'Retained tail answer' }], source: { provider: 'agy-acp', model: 'gemini-4-carbon' } };
  const fullSessionDigest = computeHistoryPrefixDigest([u1, a1, u2, a2]);
  const directive = {
    role: 'user',
    content: [{ type: 'text', text: 'You are now acting as a compaction engine.' }],
  };

  // Warm session with retained tail (`[u2, a2]` omitted from compaction input): must stay warm-incremental!
  const warmDecision = planSessionSync({
    messages: [u1, a1, directive],
    activeAcpSessionId: 'acp-s1',
    lastToolsFingerprint: toolsFp,
    currentToolsFingerprint: toolsFp,
    lastHistoryPrefixDigest: fullSessionDigest,
    pendingToolCallIds: new Set(),
    allowResumeFromReplay: true,
    dshSessionId: 'dsh-s1',
    purpose: 'compaction',
  });
  assert.equal(warmDecision.kind, 'warm-incremental');
  assert.equal(warmDecision.tailMessages.length, 1);
  assert.match(warmDecision.tailMessages[0].content[0].text, /\[Compaction boundary:/);
  assert.match(warmDecision.tailMessages[0].content[0].text, /Completed first task anchor/);

  // Restarted worker (`activeAcpSessionId === undefined`): must resume `acp-s1` via session/load!
  const resumeDecision = planSessionSync({
    messages: [u1, a1, directive],
    activeAcpSessionId: undefined,
    currentToolsFingerprint: toolsFp,
    pendingToolCallIds: new Set(),
    allowResumeFromReplay: true,
    dshSessionId: 'dsh-s1',
    purpose: 'compaction',
  });
  assert.equal(resumeDecision.kind, 'resume-incremental');
  assert.equal(resumeDecision.resumeAcpSessionId, 'acp-s1');
});

test('planSessionSync preserves ACP session via resume-incremental on legacy anchors with dshSessionId and on mid-session tool updates', () => {
  const u1 = { id: 'u1', role: 'user', content: [{ type: 'text', text: 'First' }] };
  const u2 = { id: 'u2', role: 'user', content: [{ type: 'text', text: 'Second' }] };
  const fpA = computeToolsFingerprint([{ name: 'read', description: 'Read', parameters: { type: 'object' } }]);
  const fpB = computeToolsFingerprint([
    { name: 'read', description: 'Read', parameters: { type: 'object' } },
    { name: 'grep', description: 'Grep', parameters: { type: 'object' } },
  ]);

  // 1. Legacy persisted anchor without dshSessionId resumes when DSH passes dshSessionId
  const legacyAnchor = {
    id: 'a-legacy-real',
    role: 'assistant',
    content: [{ type: 'text', text: 'Done' }],
    source: {
      replayState: {
        response: {
          version: 1,
          acpSessionId: 'acp-legacy-1',
          rawModelId: 'gemini-4-carbon-medium',
          syncedMessageCount: 2,
          toolsFingerprint: fpA,
        },
      },
    },
  };
  const legacyWithDshId = planSessionSync({
    messages: [u1, legacyAnchor, u2],
    activeAcpSessionId: undefined,
    currentToolsFingerprint: fpA,
    pendingToolCallIds: new Set(),
    allowResumeFromReplay: true,
    dshSessionId: 'session-live-abc',
  });
  assert.equal(legacyWithDshId.kind, 'resume-incremental');
  assert.equal(legacyWithDshId.resumeAcpSessionId, 'acp-legacy-1');

  // 2. Warm session where DSH upper-layer ToolSearch activates a new tool reloads MCP via resume-incremental (never discards ACP session!)
  const modernAnchor = {
    id: 'a-mod-1',
    role: 'assistant',
    content: [{ type: 'text', text: 'Done' }],
    source: {
      replayState: {
        response: {
          version: 1,
          dshSessionId: 'session-live-abc',
          acpSessionId: 'acp-warm-1',
          rawModelId: 'gemini-4-carbon-medium',
          syncedMessageCount: 2,
          toolsFingerprint: fpA,
        },
      },
    },
  };
  const warmToolAdded = planSessionSync({
    messages: [u1, modernAnchor, u2],
    activeAcpSessionId: 'acp-warm-1',
    lastToolsFingerprint: fpA,
    currentToolsFingerprint: fpB,
    pendingToolCallIds: new Set(),
    allowResumeFromReplay: true,
    dshSessionId: 'session-live-abc',
  });
  assert.equal(warmToolAdded.kind, 'resume-incremental');
  assert.equal(warmToolAdded.resumeAcpSessionId, 'acp-warm-1');

  // 3. Fork with a different dshSessionId still isolates cleanly
  const forked = planSessionSync({
    messages: [u1, modernAnchor, u2],
    activeAcpSessionId: undefined,
    currentToolsFingerprint: fpA,
    pendingToolCallIds: new Set(),
    allowResumeFromReplay: true,
    dshSessionId: 'session-forked-xyz',
  });
  assert.equal(forked.kind, 'cold-full-replay');
});

test('McpSessionEndpoint and computeToolsFingerprint filter deferLoading tools until activated by DSH', () => {
  const toolsInitial = [
    { name: 'tool_search', description: 'Search tools', parameters: { type: 'object' } },
    { name: 'heavy_db_query', description: 'Deferred tool', parameters: { type: 'object' }, deferLoading: true },
  ];
  const toolsActivated = [
    { name: 'tool_search', description: 'Search tools', parameters: { type: 'object' } },
    { name: 'heavy_db_query', description: 'Deferred tool', parameters: { type: 'object' } },
  ];

  const endpoint = new McpSessionEndpoint({
    token: 'tok',
    url: 'http://127.0.0.1:9999/mcp',
    maxToolOutputBytes: 10000,
    batchWindowMs: 0,
    onDispose() {},
  });

  endpoint.setTools(toolsInitial);
  assert.deepEqual(endpoint.bridgeToolNames, ['tool_search']);
  assert.equal(endpoint.isAllowedPermissionRequest('mcp_dsh_heavy_db_query'), false);

  endpoint.setTools(toolsActivated);
  assert.deepEqual(endpoint.bridgeToolNames, ['tool_search', 'heavy_db_query']);
  assert.equal(endpoint.isAllowedPermissionRequest('mcp_dsh_heavy_db_query'), true);
  assert.notEqual(computeToolsFingerprint(toolsInitial), computeToolsFingerprint(toolsActivated));
});

test('convertDshMessagesToAgySteps & hydrateAgyConversationDb convert non-Gemini history into native agy SQLite trajectory', async () => {
  const { DatabaseSync } = await import('node:sqlite');
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'agy-traj-unit-'));
  const dbPath = path.join(tmpDir, 'sess-1.db');

  try {
    const db = new DatabaseSync(dbPath);
    db.exec(`
      CREATE TABLE trajectory_meta (trajectory_id text PRIMARY KEY, cascade_id text, trajectory_type integer, source integer);
      CREATE TABLE steps (idx integer PRIMARY KEY, step_type integer NOT NULL DEFAULT 0, status integer NOT NULL DEFAULT 0, has_subtrajectory numeric NOT NULL DEFAULT false, metadata blob, error_details blob, permissions blob, task_details blob, render_info blob, step_payload blob, step_format integer NOT NULL DEFAULT 0);
    `);
    db.prepare('INSERT INTO trajectory_meta VALUES (?, ?, 4, 17)').run('traj-unit-1', 'sess-1');
    db.close();

    const messages = [
      { role: 'system', content: [{ type: 'text', text: 'You are a coding agent powered by deepseek-v4.\n\nAlways verify exit codes.' }] },
      { role: 'developer', content: [{ type: 'text', text: 'Context OS workspace card' }] },
      { role: 'user', content: [{ type: 'text', text: 'Clock snapshot' }], source: { kind: 'time-context', form: 'snapshot' } },
      { role: 'user', content: [{ type: 'text', text: 'Inspect alpha and beta.' }], source: { kind: 'user' } },
      {
        role: 'assistant',
        content: [
          { type: 'reasoning', text: 'Foreign DeepSeek chain-of-thought that must be stripped' },
          { type: 'text', text: 'Checking both tools.' },
          { type: 'tool-call', id: 'tc-ok-1', name: 'read', arguments: '{"file_path":"a.txt"}' },
          { type: 'tool-call', id: 'tc-err-2', name: 'bash', arguments: '{"command":"false"}' },
          { type: 'tool-call', id: 'tc-orphan-aborted', name: 'bash', arguments: '{"command":"aborted"}' },
        ],
      },
      { role: 'tool', toolCallId: 'tc-ok-1', content: [{ type: 'text', text: 'ALPHA_CONTENT' }] },
      { role: 'tool', toolCallId: 'tc-err-2', isError: true, content: [{ type: 'text', text: 'exit code 1' }] },
      {
        role: 'assistant',
        content: [
          { type: 'reasoning', text: 'Another foreign reasoning block' },
          { type: 'text', text: 'Alpha succeeded and bash failed.' },
        ],
      },
      { role: 'user', content: [{ type: 'text', text: 'Now summarize on Gemini.' }], source: { kind: 'user' } },
    ];

    const split = splitMessagesForTrajectorySync(messages);
    assert.equal(split.hasPriorTurns, true);
    assert.equal(split.tailMessages.length, 1);

    const steps = convertDshMessagesToAgySteps({
      historyMessages: split.historyMessages,
      tools: [
        { name: 'read', description: 'Read', parameters: { type: 'object' } },
        { name: 'bash', description: 'Bash', parameters: { type: 'object' } },
      ],
      bridgeToolNames: ['read', 'bash'],
    });

    // Expected logical steps:
    // 0: user-input (coalesced guidelines + developer + runtime + user)
    // 1: planner-text ("Checking both tools." — split from tool-calls, reasoning stripped!)
    // 2: planner-tool-batch (tc-ok-1, tc-err-2; tc-orphan-aborted skipped!)
    // 3: planner-text ("Alpha succeeded and bash failed.")
    assert.deepEqual(
      steps.map((s) => s.kind),
      ['user-input', 'planner-text', 'planner-tool-batch', 'planner-text'],
    );
    assert.match(steps[0].text, /<workspace_guidelines>/);
    assert.doesNotMatch(steps[0].text, /powered by deepseek-v4/);
    assert.ok(steps[0].text.indexOf('Clock snapshot') < steps[0].text.indexOf('Inspect alpha and beta.'));
    assert.equal(steps[2].calls.length, 2, 'Orphaned aborted tool call must be skipped');
    assert.equal(steps[2].calls[0].qualifiedAcpName, 'mcp_dsh_read');
    assert.match(steps[2].calls[1].outputText, /\[Tool execution error\]/);

    const { insertedRows } = hydrateAgyConversationDb({
      dbPath,
      sessionId: 'sess-1',
      steps,
    });
    // 1 user + 1 planner-text + 1 planner-tool-batch + 2 mcp-tool results + 1 final planner-text = 6 SQLite rows
    assert.equal(insertedRows, 6);

    const verifyDb = new DatabaseSync(dbPath);
    const rows = verifyDb.prepare('SELECT idx, step_type, step_payload FROM steps ORDER BY idx ASC').all();
    verifyDb.close();
    assert.deepEqual(
      rows.map((r) => r.step_type),
      [
        AGY_STEP_TYPE_USER_INPUT,
        AGY_STEP_TYPE_PLANNER_RESPONSE,
        AGY_STEP_TYPE_PLANNER_RESPONSE,
        AGY_STEP_TYPE_MCP_TOOL,
        AGY_STEP_TYPE_MCP_TOOL,
        AGY_STEP_TYPE_PLANNER_RESPONSE,
      ],
    );
    const allBytes = Buffer.concat(rows.map((r) => Buffer.from(r.step_payload)));
    assert.ok(allBytes.includes(Buffer.from(GEMINI_DUMMY_THOUGHT_SIGNATURE, 'utf8')));
    assert.ok(!allBytes.includes(Buffer.from('Foreign DeepSeek chain-of-thought', 'utf8')));
    assert.ok(!allBytes.includes(Buffer.from('tc-orphan-aborted', 'utf8')));
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});
