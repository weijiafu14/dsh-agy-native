#!/usr/bin/env node
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import { createInterface } from 'node:readline';

if (process.argv.includes('--help')) {
  console.log('  --process-per-session  Isolate session workers');
  process.exit(0);
}

const rl = createInterface({ input: process.stdin });
const sessions = new Map();
let sessionCounter = 0;

function send(obj) {
  process.stdout.write(JSON.stringify({ jsonrpc: '2.0', ...obj }) + '\n');
}

async function callMcp(mcpServer, method, params) {
  const headers = { 'content-type': 'application/json' };
  for (const h of mcpServer.headers ?? []) {
    headers[h.name] = h.value;
  }
  const resp = await fetch(mcpServer.url, {
    method: 'POST',
    headers,
    body: JSON.stringify({ jsonrpc: '2.0', id: Math.floor(Math.random() * 1e6), method, params }),
  });
  return resp.json();
}

const permissions = new Map();
let permissionSeq = 0;
function permission(sessionId, name) {
  const id = `perm-${++permissionSeq}`;
  return new Promise(resolve => {
    permissions.set(id, resolve);
    send({ id, method: 'session/request_permission', params: {
      sessionId, toolCall: { name },
      options: ['allow_once', 'reject_once'].map(kind => ({ kind, optionId: kind })),
    } });
  });
}

rl.on('line', async (line) => {
  if (!line.trim()) return;
  const msg = JSON.parse(line);
  if (permissions.has(msg.id)) {
    permissions.get(msg.id)(msg.result.outcome.optionId);
    permissions.delete(msg.id);
    return;
  }
  if (msg.method === 'initialize') {
    send({
      id: msg.id,
      result: {
        protocolVersion: 1,
        agentCapabilities: {
          loadSession: true,
          promptCapabilities: { image: true },
          sessionCapabilities: { load: {}, close: {} },
          _meta: { antigravity: { tier: true, systemPrompt: { append: true } } },
        },
      },
    });
    return;
  }

  if (msg.method === 'session/new') {
    const meta = msg.params?._meta?.antigravity ?? {};
    if (msg.params.mcpServers?.length && !process.argv.includes('--process-per-session')) {
      send({ id: msg.id, error: { code: -32602, message: 'Missing per-session HTTP MCP worker' } });
      return;
    }
    if (meta.excludeDefaultComponents === true) {
      send({
        id: msg.id,
        error: { code: -32001, message: 'TOS violation hazard: excludeDefaultComponents must not be true' },
      });
      return;
    }
    if (meta.systemPrompt !== undefined) {
      send({
        id: msg.id,
        error: { code: -32002, message: 'WAF 429 hazard: systemPrompt must never mutate official systemInstruction' },
      });
      return;
    }
    const sessionId = `fake-sess-${process.pid}-${++sessionCounter}`;
    sessions.set(sessionId, {
      mcpServer: msg.params?.mcpServers?.[0],
      modelId: 'gemini-4-carbon-medium',
      prompts: 0,
    });
    send({
      id: msg.id,
      result: {
        sessionId,
        models: {
          currentModelId: 'gemini-4-carbon-medium',
          availableModels: [
            { modelId: 'gemini-4-carbon-medium', name: 'Gemini 4 Carbon (Medium)' },
            { modelId: 'gemini-4-carbon-high', name: 'Gemini 4 Carbon (High)' },
            { modelId: 'gemini-4-hydrogen-xhigh', name: 'Gemini 4 Hydrogen (XHigh)' },
            { modelId: 'gemini-4-hydrogen-high', name: 'Gemini 4 Hydrogen (High)' },
          ],
        },
        _meta: {
          antigravity: {
            tier: { name: 'Google AI Ultra', id: 'ultra-tier' },
          },
        },
      },
    });
    return;
  }

  if (msg.method === 'session/load') {
    const sessionId = msg.params.sessionId;
    // The crash scenario saves one session. Other fixtures allow resuming any recorded ID.
    if (fs.existsSync('saved-session') && fs.readFileSync('saved-session', 'utf8') !== sessionId) {
      send({ id: msg.id, error: { code: -32000, message: 'No saved session' } });
      return;
    }
    const existing = sessions.get(sessionId);
    sessions.set(sessionId, {
      mcpServer: msg.params?.mcpServers?.[0] ?? existing?.mcpServer,
      modelId: existing?.modelId ?? 'gemini-4-carbon-medium',
      prompts: existing?.prompts ?? 1,
    });
    send({ method: 'session/update', params: { sessionId, update: {
      sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'historical load event' },
    } } });
    send({ id: msg.id, result: {} });
    return;
  }

  if (msg.method === 'session/set_model') {
    const sess = sessions.get(msg.params.sessionId);
    if (sess) sess.modelId = msg.params.modelId;
    send({ id: msg.id, result: {} });
    return;
  }

  if (msg.method === 'session/cancel') {
    const sess = sessions.get(msg.params.sessionId);
    if (sess) sess.cancelled = true;
    return;
  }

  if (msg.method === 'session/close') {
    sessions.delete(msg.params.sessionId);
    send({ id: msg.id, result: {} });
    return;
  }

  if (msg.method === 'session/prompt') {
    const { sessionId, prompt } = msg.params;
    const sess = sessions.get(sessionId);
    sess.prompts += 1;

    const textJoined = prompt
      .filter((p) => p.type === 'text')
      .map((p) => p.text)
      .join('\n');
    const imagePart = prompt.find((p) => p.type === 'image');

    send({
      method: 'session/update',
      params: {
        sessionId,
        update: { sessionUpdate: 'usage_update', used: 512, size: 1048576 },
      },
    });

    send({
      method: 'session/update',
      params: {
        sessionId,
        update: {
          sessionUpdate: 'agent_thought_chunk',
          content: { type: 'text', text: 'Thinking step...' },
        },
      },
    });

    if (textJoined.includes('INSPECT_CURRENT_INPUT')) {
      const current = textJoined.split('<current_input>').at(-1);
      send({ method: 'session/update', params: { sessionId, update: {
        sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: current },
      } } });
      send({ id: msg.id, result: { stopReason: 'end_turn' } });
      return;
    }
    if (textJoined.includes('SAVE_FOR_LOAD') && !fs.existsSync('saved-session')) fs.writeFileSync('saved-session', sessionId);
    if (textJoined.includes('EXIT_FIXTURE_PROCESS')) process.exit(24);
    const member = /CALL_PTC_MEMBER_(\d+)/.exec(textJoined)?.[1];
    if (member !== undefined) {
      const [own, foreign, unknown] = await Promise.all([
        permission(sessionId, `mcp_dsh_member_${member}`),
        permission(sessionId, `mcp_dsh_member_${Number(member) + 1}`),
        permission('missing-session', 'mcp_dsh_run_code'),
      ]);
      const route = createHash('sha256').update(sess.mcpServer.url).digest('hex').slice(0, 16);
      const reply = await callMcp(sess.mcpServer, 'tools/call', {
        name: 'run_code', arguments: { member, route, pid: process.pid, own, foreign, unknown },
      });
      send({ method: 'session/update', params: { sessionId, update: {
        sessionUpdate: 'agent_message_chunk',
        content: { type: 'text', text: `member=${member} result=${reply.result.content[0].text} active=${sessions.size}` },
      } } });
      send({ id: msg.id, result: { stopReason: 'end_turn' } });
      return;
    }

    if (textJoined.includes('SIMULATE_UPSTREAM_429')) {
      send({
        method: 'session/update',
        params: {
          sessionId,
          update: {
            sessionUpdate: 'agent_message_chunk',
            content: {
              type: 'text',
              text: '\n⚠ API error (attempt 1): RESOURCE_EXHAUSTED (code 429): Resource has been exhausted (e.g. check quota).\n',
            },
          },
        },
      });
      await new Promise((r) => setTimeout(r, 80));
      if (!sess.cancelled) {
        send({
          method: 'session/update',
          params: {
            sessionId,
            update: {
              sessionUpdate: 'agent_message_chunk',
              content: {
                type: 'text',
                text: '\n⚠ API error (attempt 2): SHOULD_NEVER_REACH_ATTEMPT_2\n',
              },
            },
          },
        });
      }
      send({ id: msg.id, result: { stopReason: sess.cancelled ? 'cancelled' : 'end_turn' } });
      return;
    }

    if (
      textJoined.includes('CALL_PARALLEL_TOOLS') &&
      !textJoined.includes('<conversation_history>') &&
      sess.mcpServer
    ) {
      // Fire 2 concurrent MCP tools/call requests within 5ms!
      const [r1, r2] = await Promise.all([
        callMcp(sess.mcpServer, 'tools/call', {
          name: 'get_left',
          arguments: { q: 'L', toolSummary: 'left' },
        }),
        callMcp(sess.mcpServer, 'tools/call', {
          name: 'get_right',
          arguments: { q: 'R', toolSummary: 'right' },
        }),
      ]);
      if (sess.cancelled || r1?.error || r2?.error || !r1?.result || !r2?.result) {
        send({ id: msg.id, result: { stopReason: 'cancelled' } });
        return;
      }
      const combined = `${r1.result.content[0].text}+${r2.result.content[0].text}`;
      send({
        method: 'session/update',
        params: {
          sessionId,
          update: { sessionUpdate: 'usage_update', used: 938, size: 1048576 },
        },
      });
      send({
        method: 'session/update',
        params: {
          sessionId,
          update: {
            sessionUpdate: 'agent_message_chunk',
            content: { type: 'text', text: `Combined=${combined}` },
          },
        },
      });
    } else if (textJoined.includes('CALL_SIX_STEP_LOOP') && sess.mcpServer) {
      // Simulate 5 sequential tool steps (~180k context each) + final step 6 (cumulative > 1M window!)
      for (let i = 1; i <= 5; i++) {
        send({
          method: 'session/update',
          params: {
            sessionId,
            update: { sessionUpdate: 'usage_update', used: 180_000 + i * 2_000, size: 1048576 },
          },
        });
        const r = await callMcp(sess.mcpServer, 'tools/call', {
          name: 'get_left',
          arguments: { q: `step-${i}` },
        });
        if (sess.cancelled || r?.error || !r?.result) {
          send({ id: msg.id, result: { stopReason: 'cancelled' } });
          return;
        }
      }
      send({
        method: 'session/update',
        params: {
          sessionId,
          update: { sessionUpdate: 'usage_update', used: 192_000, size: 1048576 },
        },
      });
      send({
        method: 'session/update',
        params: {
          sessionId,
          update: {
            sessionUpdate: 'agent_message_chunk',
            content: { type: 'text', text: 'Six-step loop complete' },
          },
        },
      });
      send({
        id: msg.id,
        result: {
          stopReason: 'end_turn',
          usage: {
            inputTokens: 19_200,
            outputTokens: 3_600,
            cachedReadTokens: 1_099_200,
            cachedWriteTokens: 0,
            thoughtTokens: 1_200,
            totalTokens: 1_122_000,
          },
        },
      });
      return;
    } else {
      const imgTag = imagePart ? ` [img:${imagePart.mimeType}:${imagePart.data.length}]` : '';
      const echoTag = textJoined.includes('ECHO_PROMPT')
        ? ` Prompt=${JSON.stringify(textJoined)}`
        : '';
      send({
        method: 'session/update',
        params: {
          sessionId,
          update: {
            sessionUpdate: 'agent_message_chunk',
            content: {
              type: 'text',
              text: `Session=${sessionId} Turn=${sess.prompts} Model=${sess.modelId}${imgTag}${echoTag}`,
            },
          },
        },
      });
    }

    send({
      id: msg.id,
      result: {
        stopReason: 'end_turn',
        usage: {
          inputTokens: 1000,
          outputTokens: 50,
          cachedReadTokens: 400,
          cachedWriteTokens: 0,
          thoughtTokens: 15,
          totalTokens: 1450,
        },
      },
    });
  }
});
