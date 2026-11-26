# dsh-agy-native

[English](./README.md) | [安全与合规边界说明](./SECURITY.md)

**面向 DeepSeek Harness (`dsh`) 的零风控原生 Antigravity ACP 模型适配器 —— 通过本机官方签名的 `agy` CLI (`agy acp --stdio`) 安全接入 Google AI Pro / Ultra 订阅。**

![一张图看懂 DSH 与 AGY 协同原理](https://raw.githubusercontent.com/weijiafu14/dsh-agy-native/main/docs/how-one-turn-works.png)

---

## 为什么需要 `dsh-agy-native`？

在 DeepSeek Harness 中接入 Google AI Pro / Ultra 会员额度时，以往两类方案各有痛点：

1. **第三方 OAuth 直连代理风险极高**：私自提取 OAuth Client ID 直连 `cloudcode-pa.googleapis.com` 会撞上 Google 网关的实时系统提示词指纹扫描与遥测信封校验（先返回无 `Retry-After` 的假 `429 RESOURCE_EXHAUSTED`，重试即升级为 `403 TOS_VIOLATION`「两振出局」封禁）。
2. **简单命令行包装 (`agy -p`) 或粗糙套壳体验割裂**：`-p` 单次调用会把工具执行吞在 `agy` 内部，导致 DSH 审批策略、子代理编排以及 `dsh-context-os` 上下文系统全部失效；而懒加载 MCP 桥接又会出现 `call_mcp_tool` 套娃、2 KB 输出被强行截断、10 分钟断连失忆、历史消息改写即冷重启等严重问题。

`dsh-agy-native` 从协议底层彻底解决了上述全部问题：

- **死守保号红线（100% 官方二进制出网 + 零 `systemInstruction` 污染 + 首轮熔断保护）**：包内**零**第三方 OAuth 客户端凭据、**零**私自发包逻辑；所有模型流量 100% 由本机官方签名的 `agy` 二进制通过标准 ACP (`agy acp --stdio`) 发出，**绝不开启** `excludeDefaultComponents: true` 且**绝不向 `systemInstruction` 注入第三方 Agent 身份提示词**（彻底避开 Cloud Code 边缘网关针对第三方系统提示词的假 `429` 拦截）；内置上游 `429/403` **第 1 次即刻熔断器**（立刻下发 `session/cancel` 阻断 `agy` 内置 8 次重试风暴并向 DSH 抛出不可重试错误），同时自动剥离环境变量中的 `GEMINI_API_KEY` / `GOOGLE_API_KEY` 防止误扣按量计费额度。
- **一级原生工具直连 (`eagerMcpServers`)**：在 `127.0.0.1:0` 启动带 192-bit 随机令牌鉴权的 Streamable-HTTP MCP 反向桥，配合 `enabledTools: []` + `eagerMcpServers: ['dsh']` 关闭 `agy` 内置工具、将 DSH 工具作为一级原生工具直接挂载，彻底告别 `call_mcp_tool` 绕路与 2 KB 截断，并自动清洗 `agy` 注入的 `toolSummary` 字段以兼容严格 Schema。
- **微批并发工具聚合**：在同一微批窗口（默认 `25ms`）内自动合并多个并发 `tools/call` 请求，一次性生成多条 `ToolCallBlock` 交由 DSH 并行执行。
- **共享 ACP 连接**：每个对话及辅助请求各用独立的 ACP 会话和经鉴权的 MCP 端点。默认 60 分钟后回收闲置会话。冷回放用文字描述历史图片，把最后一次模型回复或工具结果之后收到的消息全部作为当前输入。
- **全模型家族、思考档位与多模态支持**：自动发现订阅账号下的全部模型与思考等级（涵盖 `Gemini 4 Neon` 含 `xhigh`、`Gemini 4 Carbon`、`Gemini 4 Argon`、`Gemini 4 Hydrogen` 含 `xhigh`、`Gemini 3.8 Flash`、`Claude Opus 5.5`、`Claude Sonnet 5.5`、`Orchestrator` 等），原生支持 DSH 图片块 (`ImageBlock`) 直传、流式思考块 (`reasoning`) 展示与精确的 `TokenUsage` 拆分统计，且使用 32 MiB JSON-RPC 帧缓冲（无 256 KiB 模型目录溢出或 56 KiB 输入截断限制）。

![dsh-agy-native 整体技术架构图](https://raw.githubusercontent.com/weijiafu14/dsh-agy-native/main/docs/architecture-overview.png)

---

## 环境准备

1. **Node.js** `>= 22.13.0`
2. 已安装官方 **Antigravity CLI (`agy`)**（例如 `~/.local/bin/agy`）并完成订阅登录：
   ```bash
   agy login
   ```
3. 一键运行零 Token 消耗的环境与模型池诊断：
   ```bash
   npx dsh-agy-native doctor
   ```

---

## 安装与启用

### 方式一：通过 npm / DSH 插件命令安装

```bash
# 安装到 Web GUI Profile
dsh plugin --profile web add -w dsh-agy-native

# 或安装到 TUI / Headless CLI Profile
dsh plugin --profile tui add -w dsh-agy-native
dsh plugin --profile headless add -w dsh-agy-native
```

### 方式二：从本地源码构建安装

```bash
git clone https://github.com/weijiafu14/dsh-agy-native.git
cd dsh-agy-native
npm install
npm test
npm pack
dsh plugin --profile web add -w ./dsh-agy-native-0.2.0.tgz
```

安装后在 DSH 模型选择器中选择 **Antigravity (Google AI Ultra / Pro)**，即可自由切换模型（如 `Gemini 4 Carbon`、`Gemini 4 Hydrogen`）及推理档位（`Low` / `Medium` / `High` / `XHigh`）。

---

## 插件配置项（支持 DSH Web 设置页热更新）

| 配置项 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `providerId` | `"agy-acp"` | 注册到 DSH `ctx.llm` 的 Provider 路由 ID。 |
| `providerName` | `"Antigravity (AGY ACP)"` | 显示名称（探测到订阅等级后会自动显示为 `Antigravity (Google AI Ultra)` 等）。 |
| `agyPath` | `""` | 官方 `agy` 可执行文件路径；留空自动从 `AGY_BIN`、`~/.local/bin/agy`、`PATH` 及标准安装目录发现。 |
| `idleMinutes` | `60` | 空闲会话 回收分钟数（设为 `0` 则常驻不回收）。 |
| `batchWindowMs` | `25` | 并发 MCP 工具调用的微批聚合窗口（毫秒）。 |
| `maxToolOutputBytes` | `180000` | 单次工具输出返回给 `agy` 的 UTF-8 安全字节上限（防多字节截断乱码）。 |
| `resumeSessions` | `true` | Worker 重启后优先尝试 `session/load` 恢复会话，失败时自动回退至保留完整工具历史的结构化重放。 |
| `forwardContextUpdates` | `true` | 当 `dsh-context-os` 等插件更新历史 `developer` 上下文面时，以增量方式同步而不重置热会话。 |

---

## 工具路由与多会话隔离

当本机安装的 `agy` CLI 支持 `--process-per-session` 时，适配器会自动启用该官方 ACP 参数，使每个新建 ACP 会话独立拉取自己的 HTTP MCP 工具目录（同时同会话内的多轮对话依然复用热会话）；旧版本二进制会自动保持原有启动参数。[官方 ACP 参数参考](https://antigravity.google/docs/ide/extensions/acp/#flags)

在标准 DSH 会话中，模型会获得当前会话 `dsh` 桥接端实际注册的精确 Eager 工具名；在 PTC 会话中，外层调用为 `mcp_dsh_run_code`，程序内部则使用 DSH SDK 中的原始工具名。即使环境中存在同名的外部 MCP 服务器，ACP 权限校验也仅放行当前 `dsh` 桥接端已注册的工具，工具执行与审批继续完全由 DSH 接管。

---

## 开源许可证

[MIT](./LICENSE)
