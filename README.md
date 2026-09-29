# LangChain TypeScript 学习笔记

> 本文根据“学习 LangChain TypeScript”长对话整理。重复验证、试错过程已合并为结论，保留关键心智模型、API、完整示例与常见误区。
>
> LangChain / LangGraph 更新较快。本文以学习对话中的 TypeScript API 为主，实际项目请锁定依赖版本，并以文末官方文档为准。

## 目录

1. LangChain、Agent 与 LangGraph 的关系
2. 项目初始化与 DeepSeek 接入
3. 消息、模型与流式输出
4. Tool 与 Agent Loop
5. 结构化输出与 Zod
6. Middleware：在模型和工具调用前后插入逻辑
7. 记忆：消息历史、线程与长期记忆
8. LangGraph 基础：State、Node、Edge
9. 条件路由、循环与结束条件
10. Human-in-the-loop：interrupt 与恢复
11. 并行、Join 与 Reducer
12. Subgraph：状态隔离与组合
13. Checkpoint：持久化、恢复与历史
14. Time Travel、Replay 与 updateState
15. Send：动态并行与 Map-Reduce
16. RetryPolicy、恢复与 CachePolicy
17. 长耗时任务与回调驱动架构
18. 生产实践与复习清单

---

# 第一部分：LangChain 基础

## 1. LangChain、Agent 与 LangGraph 的关系

可以先建立三层心智模型：

```text
模型（LLM）
  └─ 只负责根据消息生成下一条消息，可能提出工具调用

LangChain Agent
  └─ 管理“模型 → 工具 → 模型”的循环

LangGraph
  └─ 管理可持久化、可暂停、可恢复、可分支和并行的工作流
```

### 1.1 什么时候只用模型

适合一次输入、一次输出：翻译、改写、摘要、简单问答。

### 1.2 什么时候用 Agent

任务需要模型自行判断是否调用工具、调用哪个工具，以及是否继续调用。

### 1.3 什么时候用 LangGraph

流程具有明确步骤，或需要以下能力：

- 条件路由与循环；
- 多节点并行；
- 人工审批；
- 持久化和失败恢复；
- 子流程复用；
- 对每个步骤进行观察、重试和审计。

Agent 擅长“让模型决定”，Graph 擅长“让工程系统控制”。两者可以组合。

## 2. 项目初始化与 DeepSeek 接入

### 2.1 安装依赖

```bash
pnpm add langchain @langchain/core @langchain/openai @langchain/langgraph zod dotenv
pnpm add -D typescript vite vite-node @types/node
```

`.env`：

```dotenv
DEEPSEEK_API_KEY=你的密钥
```

不要把 `.env` 提交到 Git。

### 2.2 创建兼容 OpenAI 协议的 DeepSeek 模型

```ts
import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";

export const model = new ChatOpenAI({
  model: "deepseek-chat",
  apiKey: process.env.DEEPSEEK_API_KEY,
  temperature: 0,
  configuration: {
    baseURL: "https://api.deepseek.com",
  },
});
```

关键点：`ChatOpenAI` 是客户端适配器，不意味着只能调用 OpenAI；只要服务兼容对应协议，就可以通过 `baseURL` 接入。

## 3. 消息、模型与流式输出

### 3.1 消息角色

```ts
import {
  SystemMessage,
  HumanMessage,
  AIMessage,
  ToolMessage,
} from "@langchain/core/messages";
```

- `SystemMessage`：系统规则和角色约束；
- `HumanMessage`：用户输入；
- `AIMessage`：模型回复，也可能包含 `tool_calls`；
- `ToolMessage`：工具执行结果，必须用正确的 `tool_call_id` 对应请求。

### 3.2 普通调用

```ts
const response = await model.invoke([
  new SystemMessage("你是一名 TypeScript 教师。"),
  new HumanMessage("解释 Promise。"),
]);

console.log(response.content);
```

### 3.3 流式调用

```ts
const stream = await model.stream([
  new HumanMessage("用三句话解释 LangGraph。"),
]);

for await (const chunk of stream) {
  // chunk 是 AIMessageChunk，不只是纯字符串。
  process.stdout.write(String(chunk.content));
}
```

如果产品既需要把文字实时显示给用户，又需要展示工具执行进度，应分别处理：

```text
模型 token/chunk → 文本流
工具开始/结束事件 → 状态事件流
最终回答 → 完成事件
```

不要把整块 `AIMessageChunk` 直接输出给用户，否则会看到大量元数据。

---

# 第二部分：Tool、Agent 与结构化输出

## 4. Tool 与 Agent Loop

## 4.1 定义工具

```ts
import { tool } from "langchain";
import { z } from "zod";

const getWeather = tool(
  async ({ city }) => {
    // 示例中使用模拟数据；真实项目可调用天气 API。
    const temperatureMap: Record<string, number> = {
      北京: 28,
      上海: 32,
    };

    return {
      city,
      temperature: temperatureMap[city] ?? 25,
    };
  },
  {
    name: "get_weather",
    description: "查询指定城市当前气温",
    schema: z.object({
      city: z.string().describe("城市名称，例如北京"),
    }),
  },
);

const calculate = tool(
  async ({ a, b, operation }) => {
    if (operation === "subtract") return a - b;
    if (operation === "add") return a + b;
    throw new Error(`不支持的操作：${operation}`);
  },
  {
    name: "calculate",
    description: "执行确定性的加减法计算",
    schema: z.object({
      a: z.number(),
      b: z.number(),
      operation: z.enum(["add", "subtract"]),
    }),
  },
);
```

Zod 通过字段名而不是函数位置参数来保证参数含义，因此 `a`、`b` 的顺序由结构化对象固定。

## 4.2 Agent Loop 的真实流程

```text
用户问题
  ↓
模型判断是否需要工具
  ↓
AIMessage.tool_calls
  ↓
程序按 name 找到工具并传入 args
  ↓
工具结果包装为 ToolMessage
  ↓
连同历史消息再次交给模型
  ↓
模型继续调用工具，或生成最终答案
```

模型不是直接执行函数。它只生成“调用哪个工具、传什么参数”的请求；程序才是真正的执行者。

### 为什么先查天气，后计算温差

计算工具需要具体温度。在获得北京和上海的温度之前，模型没有合法输入，因此合理顺序是：

```text
get_weather(北京) ─┐
                   ├─ calculate(上海温度, 北京温度, subtract)
get_weather(上海) ─┘
```

## 4.3 手写 Agent Loop（理解原理）

```ts
import { HumanMessage, ToolMessage } from "@langchain/core/messages";

const tools = [getWeather, calculate];
const toolsByName = new Map(tools.map((item) => [item.name, item]));
const modelWithTools = model.bindTools(tools);

const messages = [
  new HumanMessage("北京和上海哪个更热？温差是多少？"),
];

const MAX_ITERATIONS = 8;

for (let iteration = 0; iteration < MAX_ITERATIONS; iteration += 1) {
  const aiMessage = await modelWithTools.invoke(messages);
  messages.push(aiMessage);

  // 没有工具请求，说明模型已给出最终回答。
  if (!aiMessage.tool_calls?.length) {
    console.log(aiMessage.content);
    break;
  }

  for (const call of aiMessage.tool_calls) {
    const selectedTool = toolsByName.get(call.name);

    if (!selectedTool) {
      messages.push(
        new ToolMessage({
          tool_call_id: call.id!,
          content: `未知工具：${call.name}`,
        }),
      );
      continue;
    }

    try {
      const result = await selectedTool.invoke(call.args);
      messages.push(
        new ToolMessage({
          tool_call_id: call.id!,
          content: JSON.stringify(result),
        }),
      );
    } catch (error) {
      // 将可恢复错误告诉模型，使其有机会修正参数或换方案。
      messages.push(
        new ToolMessage({
          tool_call_id: call.id!,
          content: `工具执行失败：${String(error)}`,
        }),
      );
    }
  }

  if (iteration === MAX_ITERATIONS - 1) {
    throw new Error("Agent 超过最大迭代次数，可能陷入工具调用循环");
  }
}
```

`MAX_ITERATIONS` 是安全阀，防止模型在工具之间无限循环；它不是业务成功次数。

## 4.4 使用 createAgent

```ts
import { createAgent } from "langchain";

const agent = createAgent({
  model,
  tools: [getWeather, calculate],
  systemPrompt: "需要实时数据或精确计算时使用工具。",
});

const result = await agent.invoke({
  messages: [
    { role: "user", content: "北京和上海哪个更热？温差是多少？" },
  ],
});

console.dir(result, { depth: null });
```

### 是否把系统所有接口都注册成工具

技术上可行，工程上不建议一次暴露全部接口：

- 工具越多，模型选择成本越高；
- 描述会占用上下文；
- 写操作涉及权限、审计和幂等性；
- 相似工具容易误选。

推荐按业务场景动态提供最小工具集，并为敏感工具增加权限校验和人工审批。

## 5. 结构化输出与 Zod

结构化输出的目标不是让模型“看起来像 JSON”，而是让程序得到满足 Schema 的对象。

```ts
import { z } from "zod";

const ReviewSchema = z.object({
  approved: z.boolean(),
  score: z.number().min(0).max(100),
  suggestions: z.array(z.string()),
});

const structuredModel = model.withStructuredOutput(ReviewSchema);

const review = await structuredModel.invoke(
  "审核这篇文章，并返回是否通过、分数和修改建议。",
);

console.log(review.score);
```

如果只使用服务商的 JSON Mode，通常只能保证“输出是合法 JSON”，未必保证字段齐全、类型正确。仍应执行：

```ts
const parsed = ReviewSchema.safeParse(rawObject);

if (!parsed.success) {
  console.error(parsed.error.issues);
  // 可选择：重新提示模型修正，或直接向上抛出。
}
```

处理策略：

- 格式偶发错误：把校验错误反馈给模型并限制重试次数；
- 关键业务数据：校验失败即阻断，不要静默填默认值；
- 用户输入错误：返回明确提示；
- 程序错误：抛出并记录日志。

## 6. Middleware

Middleware 用于在 Agent 的模型或工具调用前后统一插入逻辑，例如：

- 动态系统提示词；
- 上下文压缩；
- 动态工具列表；
- 权限校验；
- 日志、计费和审计；
- 工具输入输出清洗；
- 拒绝敏感操作。

心智模型：

```text
用户请求
  ↓
Model Middleware → 模型
  ↓
模型提出 Tool Call
  ↓
Tool Middleware → 工具
  ↓
ToolMessage 返回模型
```

如果中间件拒绝工具调用，需要返回一个与原 `tool_call_id` 对应的 `ToolMessage`。否则消息链会残留一个没有响应的工具请求。

```ts
return new ToolMessage({
  tool_call_id: request.toolCall.id!,
  content: "操作被拒绝：当前用户没有权限。",
});
```

这不是“清空工具调用”，而是给该调用一个正式结果。模型收到拒绝原因后，可以向用户解释或选择其他路径。

---

# 第三部分：记忆与 LangGraph 基础

## 7. 记忆：消息历史、线程与长期记忆

### 7.1 三种不同的“记忆”

```text
当前 messages
  └─ 本次上下文中的消息

线程 Checkpoint
  └─ 由 thread_id 关联的工作流状态和消息历史

长期记忆 / 外部检索
  └─ 跨线程保存的用户事实、摘要、向量检索或数据库记录
```

可以把短期消息看作内部维护的列表，并由 `thread_id` 找到对应会话；但超长历史不应无限塞入上下文。

常见策略：

1. 保留最近 N 条原始消息；
2. 把更早消息压缩为 summary；
3. 重要事实写入长期记忆；
4. summary 中没有的信息，再从历史数据库或 RAG 检索召回。

## 8. LangGraph 基础：State、Node、Edge

### 8.1 State 是共享数据，不是全局可变对象

```ts
import { Annotation } from "@langchain/langgraph";

const GraphState = Annotation.Root({
  essay: Annotation<string>(),
  score: Annotation<number>(),
  reviewCount: Annotation<number>(),
});
```

所有节点可以读取 State；节点返回的是局部更新：

```ts
async function reviewNode(state: typeof GraphState.State) {
  return {
    score: state.score + 30,
    reviewCount: state.reviewCount + 1,
  };
}
```

不要把 State 理解为多个异步函数共同修改的普通 JavaScript 对象。更准确的模型是：

```text
节点读取一个 State 快照
  ↓
返回 State Update
  ↓
LangGraph 按 Channel/Reducer 规则合并
```

### 8.2 Node、Edge、START、END

```ts
import { START, END, StateGraph } from "@langchain/langgraph";

const graph = new StateGraph(GraphState)
  .addNode("review", reviewNode)
  .addNode("publish", publishNode)
  .addEdge(START, "review")
  .addEdge("publish", END)
  .compile();
```

内部可以把特殊节点理解为：

```text
__start__ → review → publish → __end__
```

- `addNode`：注册函数；
- `addEdge`：定义固定流转；
- `addConditionalEdges`：根据状态决定下一步；
- `START`、`END`：特殊起点和终点。

## 9. 条件路由、循环与结束条件

```ts
function routeAfterReview(state: typeof GraphState.State) {
  return state.score >= 60 ? "pass" : "rewrite";
}

const workflow = new StateGraph(GraphState)
  .addNode("submit", submitNode)
  .addNode("review", reviewNode)
  .addNode("rewrite", rewriteNode)
  .addNode("pass", passNode)
  .addEdge(START, "submit")
  .addEdge("submit", "review")
  .addConditionalEdges("review", routeAfterReview, {
    pass: "pass",
    rewrite: "rewrite",
  })
  .addEdge("rewrite", "review")
  .addEdge("pass", END)
  .compile();
```

这里必须区分：

```text
业务不通过（score < 60）
  └─ 函数正常 return，Task 执行成功，用 Router 进入 rewrite

程序执行失败（throw / Promise reject）
  └─ Task 失败，交给 RetryPolicy 或失败恢复
```

循环必须有边界，例如最大重写次数，否则可能永远运行：

```ts
if (state.reviewCount >= 3) return "manualReview";
```

---

# 第四部分：人工审批、并行与子图

## 10. Human-in-the-loop：interrupt 与恢复

`interrupt()` 会暂停当前 Task，把数据暴露给外部，等待后续输入。要跨调用恢复，Graph 必须配置 Checkpointer，并且两次调用使用同一个 `thread_id`。

### 10.1 最小完整示例

```ts
import {
  Annotation,
  Command,
  END,
  interrupt,
  MemorySaver,
  START,
  StateGraph,
} from "@langchain/langgraph";

const State = Annotation.Root({
  draft: Annotation<string>(),
  approved: Annotation<boolean>(),
  comment: Annotation<string>(),
});

async function createDraft() {
  return { draft: "待审核的文章草稿" };
}

async function humanReview(state: typeof State.State) {
  // 注意：恢复时节点会从头重新执行，所以 interrupt 前不要做非幂等副作用。
  const decision = interrupt({
    type: "article_review",
    content: state.draft,
  }) as { approved: boolean; comment: string };

  return {
    approved: decision.approved,
    comment: decision.comment,
  };
}

function route(state: typeof State.State) {
  return state.approved ? "approved" : "rejected";
}

const checkpointer = new MemorySaver();

const graph = new StateGraph(State)
  .addNode("createDraft", createDraft)
  .addNode("humanReview", humanReview)
  .addNode("approved", async () => ({}))
  .addNode("rejected", async () => ({}))
  .addEdge(START, "createDraft")
  .addEdge("createDraft", "humanReview")
  .addConditionalEdges("humanReview", route)
  .addEdge("approved", END)
  .addEdge("rejected", END)
  .compile({ checkpointer });

const config = {
  configurable: { thread_id: "article-001" },
};

// 第一次运行到 interrupt() 后暂停。
const first = await graph.invoke({}, config);
console.dir(first, { depth: null });

// 外部系统稍后提交审批结果，并用同一个 thread_id 恢复。
const final = await graph.invoke(
  new Command({
    resume: {
      approved: true,
      comment: "审核通过",
    },
  }),
  config,
);

console.dir(final, { depth: null });
```

### 10.2 多个并行 interrupt

并行审批时，结果中会有多个 `__interrupt__`，每个都有独立 `id` 和 `value`：

```text
__interrupt__:
  - id: xxx
    value: { type: "characters_review", ... }
  - id: yyy
    value: { type: "scenes_review", ... }
```

外部系统应保存审批任务 ID，而不是仅靠 `type` 猜测。批量恢复可按中断 ID 提交结果：

```ts
new Command({
  resume: {
    [charactersInterrupt.id]: {
      approved: true,
      comment: "人物设定通过",
    },
    [scenesInterrupt.id]: {
      approved: false,
      comment: "请调整场景三",
    },
  },
});
```

`interrupt({ type, content })` 中的内容是提供给外部 UI 的业务数据；`resume` 中的对象是外部返回给节点的审批结果。

### 10.3 幂等性规则

恢复时，包含 `interrupt()` 的节点会从头执行。因此：

- `interrupt()` 尽量放在节点开头；
- 之前的日志可能重复；
- 之前不要扣款、发邮件或写入不可重复的数据；
- 必须做副作用时使用幂等键。

## 11. 并行、Join 与 Reducer

### 11.1 固定并行

```ts
.addEdge("generateOutline", "generateCharacters")
.addEdge("generateOutline", "generateScenes")
```

同一个上游指向多个下游时，下游可在同一轮并行执行。

### 11.2 Superstep 心智模型

```text
Superstep 0: generateOutline
        ↓
Superstep 1: generateCharacters || generateScenes
        ↓
合并这一轮 State Update
        ↓
Superstep 2: generateScript
```

Graph 保证阶段顺序，不保证并行任务的完成顺序。

### 11.3 Join

如果下游必须等待两个分支都完成：

```ts
.addEdge(
  ["charactersDone", "scenesDone"],
  "generateScript",
)
```

空的 `charactersDone` / `scenesDone` 节点可以充当明确的完成标记，让 Join 依赖的是“整个分支完成”，而不是分支中某个可能循环或中断的中间节点。

### 11.4 并行写冲突与 Reducer

两个并行节点写同一字段，如果该 Channel 没有 reducer，通常会产生并发更新冲突。数组收集示例：

```ts
const State = Annotation.Root({
  summaries: Annotation<Array<{ index: number; text: string }>>({
    reducer: (current, update) => [...current, ...update],
    default: () => [],
  }),
});
```

并行任务的完成顺序不能作为业务顺序。携带 `index`，最后显式排序：

```ts
const ordered = [...state.summaries]
  .sort((a, b) => a.index - b.index);
```

实践原则：

- 并行分支尽量只读共同输入；
- 输出写到各自字段；
- 必须写同一字段时声明 reducer；
- 串行更新不存在同一轮竞态，但仍要保持节点职责清晰。

## 12. Subgraph：状态隔离与组合

复杂流程可拆为独立子图，例如：

```text
父图：生成大纲
  ├─ 人物子图：生成人物 → 审核 → 修改
  └─ 场景子图：生成场景 → 审核 → 修改
        ↓
父图：生成剧本
```

推荐分别定义：

- `input`：子图允许接收的字段；
- `stateSchema`：子图内部完整状态；
- `output`：子图允许返回父图的字段。

```ts
const CharacterInput = Annotation.Root({
  outline: Annotation<string>(),
});

const CharacterState = Annotation.Root({
  outline: Annotation<string>(),
  characters: Annotation<string>(),
  approved: Annotation<boolean>(),
  comment: Annotation<string>(),
});

const CharacterOutput = Annotation.Root({
  characters: Annotation<string>(),
});

const characterSubgraph = new StateGraph({
  stateSchema: CharacterState,
  input: CharacterInput,
  output: CharacterOutput,
});
```

可以把子图理解成：父图把输入状态投影给子图，子图在自己的状态空间运行，最后把声明的输出更新合并回父图；不要把它想成多个函数共享同一个可变内存对象。

如果父图没有子图声明的输入字段，该字段自然无法得到有效值；如果并行子图同时向父图同一字段写不同值，也必须用 reducer 或重构状态设计。

`Command.goto` 能让节点动态指定下一步，但过度使用会把流程藏进函数，降低拓扑可读性。优先级可记为：

```text
addEdge              固定流程，最清晰
addConditionalEdges  动态路由，结构仍可见
Command.goto         高度动态，谨慎使用
```

子图中的 `Command.PARENT` 可用于跳转父图，但应只用于确实需要跨图控制的场景。

---

# 第五部分：持久化、回溯与动态并行

## 13. Checkpoint：持久化、恢复与历史

### 13.1 Checkpointer 与 Checkpoint

- Checkpointer：保存器，例如 `MemorySaver`、`PostgresSaver`；
- Checkpoint：某一时刻的工作流快照和执行现场。

Checkpoint 不只保存 State，还保存：

- 当前执行到哪里；
- 下一步 Task；
- 当前 Superstep 的任务信息；
- 配置与元数据；
- 并行任务的 pending writes。

### 13.2 MemorySaver 与 PostgresSaver

`MemorySaver` 适合本地学习，进程退出后数据消失。生产环境应使用持久化 Saver。

```ts
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";

const checkpointer = PostgresSaver.fromConnString(
  process.env.POSTGRES_URL!,
);

// 首次使用时初始化 LangGraph 所需表结构。
await checkpointer.setup();

const graph = workflow.compile({ checkpointer });
```

LangGraph 会管理 Checkpoint 内部表；业务数据库和 Checkpoint 数据库可以采用不同的扩展、分库和生命周期策略。

### 13.3 thread_id

```ts
const config = {
  configurable: {
    thread_id: "short-drama-001",
  },
};
```

`thread_id` 是一条持久化工作流实例的主线标识。只有 `thread_id` 时，通常定位该线程的最新状态；同时提供 `checkpoint_id` 时，精确定位某个历史快照。

### 13.4 invoke 结果、getState 与 getStateHistory

- `invoke()` 结果：本次调用结束时可见的 State；
- `getState(config)`：从持久化层读取当前 StateSnapshot；
- `getStateHistory(config)`：读取该线程的历史快照。

不要认为“每次 `invoke` 只产生一个快照”。Graph 通常会在执行步骤/超步边界创建多个 Checkpoint。

```ts
const current = await graph.getState(config);

for await (const snapshot of graph.getStateHistory(config)) {
  console.log({
    values: snapshot.values,
    next: snapshot.next,
    tasks: snapshot.tasks,
    config: snapshot.config,
    metadata: snapshot.metadata,
  });
}
```

## 14. Time Travel、Replay 与 updateState

### 14.1 从历史 Checkpoint 继续

```ts
await graph.invoke(null, historicalSnapshot.config);
```

这里决定“从哪里执行”的核心是 config 中的 `thread_id + checkpoint_id`。`input` 则是本次调用要追加到 State 的输入：

- `null`：不提供新输入，按历史现场继续；
- `{}`：在没有字段更新时效果通常近似 `null`；
- `{ value: "new" }`：先按 State Channel 规则更新，再继续。

从历史快照运行会形成新的执行分支，不会把原历史原地改掉。

### 14.2 updateState

```ts
const newConfig = await graph.updateState(
  snapshot.config,
  { value: "人工修改后的值" },
  "A",
);
```

语义是：基于指定 Checkpoint 创建一个新 Checkpoint，提交一次 State Update，并把它视为节点 `A` 产生的更新。

第三个参数 `asNode` 不是“下一个节点”，而是“把这次更新归属于哪个已完成节点”。这会影响 Graph 根据哪个节点的 Edge 计算下一步。

### 14.3 选择正确的历史位置

若目标是“修改 A 的产物，然后重新执行 B、C”，应该选择 A 完成、B 尚未执行的快照：

```text
values: { value: "A" }
next: ["B"]
```

若要让 A 自己重新执行，通常要从 A 尚未执行的快照开始，而不是选择 A 已完成后的快照。

### 14.4 Replay 不等于简单重跑

LangGraph 可能识别并复用历史上已执行过的步骤。观察日志时，不要只看最终 State；同时看：

- `next`；
- `tasks`；
- `checkpoint_id`；
- 节点实际打印日志；
- 新分支的历史。

### 14.5 前端流程图如何映射到执行记录

不能只用 `nodeName`，因为同一节点可能循环执行多次，也可能被 `Send` 并行执行多次。业务层应为一次节点执行建立独立记录，例如：

```text
workflow_execution
  - execution_id
  - thread_id
  - graph_version

node_execution
  - node_execution_id
  - execution_id
  - node_name
  - checkpoint_before_id
  - checkpoint_after_id
  - task_id
  - status
  - started_at
  - finished_at
```

前端点击的是 `node_execution_id`，后端再解析对应的 checkpoint/task。不要复制保存整份 Checkpoint History；Saver 已持久化内部历史，业务表只需保存 UI 和业务执行的映射及审计信息。

## 15. Send：动态并行与 Map-Reduce

固定并行在编译图时就知道分支数量；`Send` 用于运行时才知道任务数量的场景。

### 15.1 完整示例

```ts
import {
  Annotation,
  END,
  Send,
  START,
  StateGraph,
} from "@langchain/langgraph";

type Summary = {
  index: number;
  chapter: string;
  summary: string;
};

const State = Annotation.Root({
  chapters: Annotation<string[]>(),
  summaries: Annotation<Summary[]>({
    reducer: (current, update) => [...current, ...update],
    default: () => [],
  }),
  finalText: Annotation<string>(),
});

const ChapterInput = Annotation.Root({
  index: Annotation<number>(),
  chapter: Annotation<string>(),
});

async function generateChapters() {
  return {
    chapters: ["第一章", "第二章", "第三章"],
  };
}

function dispatchChapters(state: typeof State.State) {
  return state.chapters.map(
    (chapter, index) =>
      new Send("processChapter", {
        index,
        chapter,
      }),
  );
}

async function processChapter(
  input: typeof ChapterInput.State,
) {
  // 每个 Send 都创建一次独立 Task，并携带独立输入。
  return {
    summaries: [
      {
        index: input.index,
        chapter: input.chapter,
        summary: `${input.chapter}的摘要`,
      },
    ],
  };
}

async function mergeResults(state: typeof State.State) {
  const ordered = [...state.summaries]
    .sort((a, b) => a.index - b.index);

  return {
    finalText: ordered
      .map((item) => `${item.chapter}：${item.summary}`)
      .join("\n"),
  };
}

const graph = new StateGraph(State)
  .addNode("generateChapters", generateChapters)
  .addNode("processChapter", processChapter)
  .addNode("mergeResults", mergeResults)
  .addEdge(START, "generateChapters")
  .addConditionalEdges(
    "generateChapters",
    dispatchChapters,
    ["processChapter"],
  )
  .addEdge("processChapter", "mergeResults")
  .addEdge("mergeResults", END)
  .compile();

console.dir(await graph.invoke({ summaries: [] }), {
  depth: null,
});
```

核心记忆：

> `Send` 不创建新 Node；它为已有 Node 创建一次 Task，并为该 Task 提供独立输入。

`addConditionalEdges(..., ["processChapter"])` 的第三个参数声明可能的目标节点。Router 返回字符串时表达“去哪里”，返回 `Send[]` 时表达“创建哪些任务、每个任务带什么输入”。

### 15.2 同名与不同名任务

`Send[]` 可以指向同一个 Node，也可以指向多个不同 Node。只要任务处于同一 Superstep，LangGraph 会先完成这一轮，合并 State Update，再计算下一轮；不是某个并行 Task 一结束就无限向下独自跑完整个分支。

### 15.3 部分失败与 pending writes

```text
Superstep N
  Task A ✓
  Task B ✗
  Task C ✓
```

有 Checkpointer 时，成功 Task 的结果可作为 pending writes 保存。恢复后，框架可以只重新执行未完成/失败 Task，而不必重复 A、C。

并行任务即使属于同一个 Node，也有独立 `task_id`，因此框架能区分：

```text
processChapter / Task #1 ✓
processChapter / Task #2 ✗
processChapter / Task #3 ✓
```

---

# 第六部分：容错、缓存与生产架构

## 16. RetryPolicy、恢复与 CachePolicy

## 16.1 Task 如何判定成功失败

```text
Promise resolve / 正常 return → Task 成功
throw / Promise reject        → Task 失败
```

LangGraph 不知道业务结果好不好：`return { approved: false }` 仍是执行成功，应通过 Router 处理；底层 API 异常冒出节点才是执行失败。

如果这样吞掉错误：

```ts
catch (error) {
  return { result: "" };
}
```

框架会认为节点成功，RetryPolicy 不会触发。需要记录日志时应重新抛出：

```ts
catch (error) {
  console.error(error);
  throw error;
}
```

## 16.2 RetryPolicy

重试策略配置在 Node 上，而不是在节点函数内部调用某个 `retry()`：

```ts
.addNode("callModel", callModel, {
  retryPolicy: {
    maxAttempts: 3,
    initialInterval: 1,
    backoffFactor: 2,
  },
})
```

`maxAttempts: 3` 表示总尝试次数最多为 3，不是“首次失败后再重试 3 次”。重试的是当前 Task，不是整个 Graph。

适合重试：

- 网络超时；
- 429 限流；
- 502/503 等临时服务故障；
- 暂时性的数据库连接失败。

不适合直接重试：

- 参数和 Schema 错误；
- 401/403 权限问题；
- 明确的业务拒绝；
- 未实现幂等性的付款、发券、发消息等副作用操作。

项目中可以抽取默认策略，但应允许节点覆盖，不能无脑全局应用。

## 16.3 Retry 与 Checkpoint Resume

```text
RetryPolicy
  └─ 当前 invoke 内，Task 异常后自动重新调用

Checkpoint Resume
  └─ invoke 已失败或暂停，之后用同一 thread_id 从现场继续
```

完整容错链：

```text
Task 失败
  ↓
RetryPolicy 自动重试
  ↓ 仍失败
Graph Run 失败，但 Checkpoint 保留现场
  ↓
人工修复外部问题，必要时 updateState
  ↓
graph.invoke(null, sameConfig)
  ↓
从 next/tasks 继续
```

这里没有名为 `resume()` 的普通 API。`graph.invoke(null, config)` 在同一个持久化线程上继续，常被描述为 resume。

## 16.4 Checkpoint Resume 与 CachePolicy

两者都可能表现为“节点函数没有再次调用”，但依据完全不同：

| 机制 | 判断问题 | 作用范围 |
|---|---|---|
| Checkpoint Resume | 这次工作流执行到哪里，哪些 Task 已完成？ | 同一持久化执行现场 |
| CachePolicy | 这个 Node 对相同输入以前是否计算过？ | 可跨新的工作流执行 |

一句话：

> Checkpoint 决定接下来调度哪个 Task；CachePolicy 决定已经被调度的 Task 能否用缓存结果代替真正执行。

因此，即使 `updateState()` 改变了某些字段，只要执行位置仍是 `next = ["B"]`，A 也不会自动重新执行；不是因为 A 命中缓存，而是 A 根本没有被调度。

缓存只适合输入相同就可安全复用的纯计算或读取操作，不应缓存付款、发送消息等唯一动作；还要考虑 TTL、数据新鲜度、模型版本和提示词是否进入 Cache Key。

## 17. 长耗时任务与回调驱动架构

不要让 Graph 节点一直 `await` 数小时的视频生成任务。推荐：

```text
Graph 节点提交视频任务
  ↓
外部视频系统返回 job_id
  ↓
Graph 保存 job_id 和 running 状态后 interrupt
  ↓
HTTP 请求结束，前端显示“处理中”
  ↓
视频系统完成后回调业务服务
  ↓
业务服务校验回调、写入结果
  ↓
用同一 thread_id + Command(resume) 恢复 Graph
  ↓
后续节点继续执行并通知用户
```

回调处理必须考虑：

- `job_id`、`thread_id`、中断 ID 的映射；
- 回调签名校验；
- 重复回调幂等；
- 任务超时与取消；
- 回调早于事务提交的竞态；
- 失败后的人工补偿。

若同一工作流里一个分支暂停、其他并行分支仍在运行，外部看到的是一组未完成 Task，而不是简单的“整个进程卡死”。是否允许其他分支继续，以及何时合并，取决于 Superstep 和图结构。

## 18. 生产实践与复习清单

### 18.1 设计工作流的顺序

1. 先画业务步骤，识别固定流程、条件、循环、并行和人工输入；
2. 设计 State，只保存原始业务数据，不保存大量格式化 Prompt；
3. 每个 Node 尽量单一职责；
4. 明确每个错误由系统重试、模型修正、用户修正还是开发者处理；
5. 再连接 Edge、Router、Join 和子图；
6. 为外部副作用设计幂等键；
7. 加 Checkpointer、日志和执行记录；
8. 最后才做可视化编辑器和动态配置。

### 18.2 Web 可视化工作流的风险

拖拽式编排可以实现，但不能只允许用户随意连线。至少需要静态校验：

- 是否有 START 和可达 END；
- 循环是否有退出条件和最大次数；
- 并行写同一 State Channel 是否有 reducer；
- Join 是否等待正确的完成节点；
- interrupt 节点是否配置持久化与审批 Schema；
- 工具和节点参数类型是否兼容；
- 子图输入输出是否与父图字段映射一致；
- 有副作用的节点是否可安全重试。

### 18.3 高频易错点

- 把业务不通过当作程序异常；
- `catch` 后返回空对象，导致重试失效；
- 把 `asNode` 理解成“下一个节点”；
- 认为一次 `invoke` 只产生一个 Checkpoint；
- 用 `nodeName` 唯一标识一次执行；
- 依赖并行完成顺序排列业务数据；
- 在 `interrupt()` 之前执行不可重复副作用；
- 误以为 `addEdge` 自己表达“等待所有并行任务”；
- 把 Checkpointer 当作 Checkpoint；
- 把恢复时未执行节点误认为命中了缓存。

### 18.4 一页复习

```text
Model              生成消息或 Tool Call
Tool               受 Schema 约束的可执行函数
Agent Loop         模型 ↔ 工具，直到最终回答
Middleware         在模型/工具调用前后统一拦截

State              Graph 的共享数据模型
Node               读取 State，返回 State Update
Edge               固定流转
Conditional Edge   根据 State 路由
Command.goto       节点内部动态跳转
interrupt          暂停并等待外部输入
Command.resume     为中断提供外部输入

Superstep          Graph 推进的一轮
Task               某个 Node 的一次具体执行
Reducer            合并同一 Channel 的多个更新
Send               运行时动态创建 Task
Subgraph           可复用、可隔离状态的子流程

Checkpointer       持久化实现
Checkpoint         State + 执行进度快照
thread_id          工作流实例主线
checkpoint_id      某个历史快照
updateState        基于快照创建人工 State Update

RetryPolicy        当前 Task 异常后的自动重试
Resume             从持久化 next/tasks 继续
CachePolicy        已调度 Task 的输入命中缓存后复用结果
```

---

# 官方资料

- [LangChain JavaScript 文档](https://docs.langchain.com/oss/javascript/langchain/overview)
- [LangGraph JavaScript 文档](https://docs.langchain.com/oss/javascript/langgraph/overview)
- [Thinking in LangGraph](https://docs.langchain.com/oss/javascript/langgraph/thinking-in-langgraph)
- [LangGraph Persistence](https://docs.langchain.com/oss/javascript/langgraph/persistence)
- [LangGraph Interrupts](https://docs.langchain.com/oss/javascript/langgraph/interrupts)
- [LangGraph Subgraphs](https://docs.langchain.com/oss/javascript/langgraph/use-subgraphs)
- [DeepSeek JSON Mode](https://api-docs.deepseek.com/zh-cn/guides/json_mode)

## 建议学习顺序

```text
模型与消息
  → Tool 与 Agent Loop
  → 结构化输出
  → Middleware 与记忆
  → StateGraph 基础
  → Router 与循环
  → interrupt
  → 并行、Join、Reducer
  → Subgraph
  → Checkpoint 与 Time Travel
  → Send
  → Retry、Resume、Cache
  → 生产架构
```

每学完一章，建议亲手修改示例中的一个条件并观察日志、`next`、`tasks` 和 State History。LangGraph 最适合通过“运行—观察—修改—再次运行”建立心智模型。
