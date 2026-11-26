# dsh-agy-native

[简体中文](./README.zh-CN.md) | [Security & TOS Policy](./SECURITY.md)

**TOS-safe native DeepSeek Harness (`dsh`) LLM provider for Google AI Pro / Ultra subscriptions via the official Antigravity CLI (`agy acp --stdio`).**

`dsh-agy-native` connects your existing Google AI Pro / Ultra subscription (`agy login`) into DeepSeek Harness as a first-class `LlmAdapter` — **without** third-party OAuth token scraping, **without** unofficial `cloudcode-pa` HTTP requests, and **without** brittle OpenAI proxy wrappers.

![How DSH and AGY collaborate in one turn](https://raw.githubusercontent.com/weijiafu14/dsh-agy-native/main/docs/how-one-turn-works.png)

---

## Why `dsh-agy-native`?

Third-party OAuth proxies that impersonate Antigravity clients directly against Google's internal `v1internal` endpoints face active prompt-fingerprint detection (`429 RESOURCE_EXHAUSTED` -> `403 TOS_VIOLATION` two-strike account bans). Meanwhile, naive CLI wrappers (`agy -p`) hide tool execution from DeepSeek Harness, breaking DSH approvals, subagents, and context orchestrators.

`dsh-agy-native` solves both problems cleanly:

- **100% Official Binary Egress & Untouched `systemInstruction` (Zero OAuth Scraping)** — Every upstream model request is authenticated and sent exclusively by your locally installed, Google-signed `agy` binary over the Agent Client Protocol (`agy acp --stdio`). Official `systemInstruction` components remain 100% untouched (`excludeDefaultComponents` is never set and third-party persona prompts are never appended into `systemInstruction`), backed by an **attempt-1 upstream `429`/`403` circuit breaker** that immediately cancels `agy`'s internal 8-attempt retry loop to prevent retry storms.
- **First-Class Eager DSH Tools (`eagerMcpServers`)** — Mounts a loopback-only (`127.0.0.1:0`) Streamable-HTTP MCP bridge with per-session cryptographic tokens so DSH tools appear to Gemini/Claude as native first-class tools (no `call_mcp_tool` lazy-loading indirection or 2 KB output truncation).
- **Parallel Tool Micro-Batching & Argument Sanitization** — Coalesces concurrent MCP `tools/call` arrivals into unified DSH `ToolCallBlock` batches and strips `agy`-injected `toolSummary` fields to preserve `additionalProperties: false` schemas.
- **Shared ACP transport** — Conversations and auxiliary requests have separate ACP session IDs and authenticated MCP endpoints. Idle cleanup closes individual sessions. Cold replay represents historical images as text and forwards all new messages after the last assistant or tool result.
- **Full Model & Thinking Level Support** — Automatically discovers all models available to your subscription tier (`Gemini 4 Neon` including `xhigh`, `Gemini 4 Carbon`, `Gemini 4 Argon`, `Gemini 4 Hydrogen` including `xhigh`, `Gemini 3.8 Flash`, `Claude Opus 5.5`, `Claude Sonnet 5.5`, `Orchestrator`, etc.), native multimodal `ImageBlock` inputs, streaming `reasoning` chunks, and disjoint `TokenUsage` metrics.

---

## Prerequisites

1. **Node.js** `>= 22.13.0`
2. **Official Antigravity CLI (`agy`)** installed (e.g., `~/.local/bin/agy` or in `PATH`) and signed in:
   ```bash
   agy login
   ```
3. Verify your local environment (uses **zero** model tokens):
   ```bash
   npx dsh-agy-native doctor
   ```

---

## Installation in DeepSeek Harness

### Install from npm

```bash
# Add to your Web GUI profile
dsh plugin --profile web add -w dsh-agy-native

# And/or add to your TUI / headless CLI profiles
dsh plugin --profile tui add -w dsh-agy-native
dsh plugin --profile headless add -w dsh-agy-native
```

Or install directly inside the **DeepSeek Harness Web GUI** → **Settings** → **Plugins** → search **`dsh-agy-native`**.

### Install from a local build

```bash
git clone https://github.com/weijiafu14/dsh-agy-native.git
cd dsh-agy-native
npm install
npm test
npm pack
dsh plugin --profile web add -w ./dsh-agy-native-0.2.0.tgz
```

Once enabled, select **Antigravity (Google AI Ultra / Pro)** in the DSH model picker and choose both your model family (e.g., `Gemini 4 Carbon`, `Gemini 4 Hydrogen`) and reasoning effort (`Low`, `Medium`, `High`, `XHigh`).

---

## Live Plugin Settings

`dsh-agy-native` exposes live-editable settings in **DSH Settings** (via `@deepseek-ai/schemastery` volatile fields):

| Option | Default | Description |
| :--- | :--- | :--- |
| `providerId` | `"agy-acp"` | Provider route ID registered in `ctx.llm`. |
| `providerName` | `"Antigravity (AGY ACP)"` | Display name in selectors (auto-enriched with subscription tier, e.g. `Antigravity (Google AI Ultra)`). |
| `agyPath` | `""` | Explicit path to `agy`; auto-discovered from `AGY_BIN`, `~/.local/bin/agy`, `PATH`, or standard install paths when empty. |
| `idleMinutes` | `60` | Minutes before recycling an idle session (`0` keeps sessions warm indefinitely). |
| `batchWindowMs` | `25` | Micro-batch window (ms) for coalescing concurrent MCP tool invocations. |
| `maxToolOutputBytes` | `180000` | UTF-8 safe byte cap per tool output returned over the MCP bridge. |
| `resumeSessions` | `true` | Attempt ACP `session/load` recovery across worker restarts before falling back to structured transcript replay. |
| `forwardContextUpdates` | `true` | Forward mutated `developer` context surfaces incrementally without tearing down warm ACP sessions. |

---

## Architecture Overview

```text
DeepSeek Harness (Agent Loop + Context OS + Tool Sandbox + Attachments)
   │
   │  ctx.llm.registerAdapter(['agy-acp'], AgyAcpAdapter)
   ▼
AgyWorkerPool (Sticky conversation slots + isolated #auxSlot for title/summary)
   │
   ├──► stdio (JSON-RPC 2.0 ACP) ──► Official `agy acp --stdio` binary ──► Google Cloud Code (Official Auth)
   │                                    • enabledTools: []
   │                                    • eagerMcpServers: ['dsh']
   │                                    • default prompt components preserved
   │
   └──◄ 127.0.0.1:0 Streamable HTTP MCP (`x-dsh-mcp-token`) ◄── Eager native tool calls (`mcp_dsh_*`)
        (Suspends MCP response -> yields ToolCallBlock to DSH -> resolves when DSH returns ToolResultMessage)
```

---

### Tool routing

The adapter uses ACP's `--process-per-session` flag when the installed CLI supports it. This lets each new ACP session discover its own HTTP MCP catalog while conversation turns can still reuse a warm session. Older binaries retain their existing launch arguments. [Official ACP flag reference](https://antigravity.google/docs/ide/extensions/acp/#flags)

In standard DSH sessions the model receives the exact eager names exposed by the session's `dsh` bridge. In PTC sessions the outer call is `mcp_dsh_run_code`, and the program uses the names in the DSH SDK. Ambient MCP servers can have similar names. Permission to enter the bridge applies only to a registered tool on that bridge. DSH handles execution and its own approvals.

## CLI Diagnostics & Testing

```bash
# Run unit, mock ACP, and real Cordis + LlmRuntime integration tests
npm test

# Run zero-token diagnostic check against your local official `agy` binary
npm run doctor

# Run opt-in end-to-end live test against your signed-in `agy` binary
npm run test:live
```

---

## Publishing to `dsh-plugin-catalog`

1. Publish to npm (`npm publish --access public`). Ensure `"dsh-plugin"` remains in `package.json#keywords` and `"dsh.bundle.patch"` points to `./cordis.patch.yml`.
2. The official DeepSeek Harness plugin crawler indexes public npm packages tagged with `dsh-plugin`, or you can submit the package to [awesome-dsh-plugin.com](https://awesome-dsh-plugin.com).

---

## License

[MIT](./LICENSE)
