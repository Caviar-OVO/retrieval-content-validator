# 检索内容校验 skill (retrieval-content-validator)

**中文名：检索内容校验 skill** ｜ 代码内实现为 `rva/`（RVA = Response Validation Agent，本地别名）

一个用于审查大语言模型（LLM）在读取网络搜索结果后"如何筛选信息"的校验代理。
它把模型的回答拆解为认知单元，核验其证据与来源，输出**可追溯、可质疑、分层化**
的结论，并以 **Factuality 事实性**（锚定外部可靠来源，五级判定）作为首要评估指标；
另有**三重校验**：证据效力（五级判定）、**可审计性**（criteria/分层证据/修订日志）、**诚实性**（H1–H5 披露检查）；外加一份**独立的 Grounding（接地性）复核**（可选 Ragas / TruLens judge），全部**分开报告**。旧名 Faithfulness 保留为兼容别名。

> 项目背景：搜索引擎后端按语义相似度和商业流量因素排序，却缺乏真实性验证。
> LLM 接收这些有偏见的文档后，会依据"相关性 / 受欢迎程度"来选择、保留或丢弃信息，
> 从而放大认知风险。本 Agent 的目标是审查这类过滤行为，评估其伦理合理性，并提出
> 恢复知识传播客观性的机制。
>
> **角色设定（双阶段）**：本 Agent 同时担任两个角色，先后接力完成校验——
> 阶段一 **Critic 批判性核验者**（调查记者人设：拆解声明、检索核验、给出判定），
> 阶段二 **Reviser 审慎修订者**（编辑人设：依据核验报告做最小修订，不引入新事实，保持结构、风格与篇幅）。
> 两阶段共享同一组伦理判准：能力要可证 · 路径要可审 · 态度要诚实 · 结果要有据。
> 角色定义见 `rva/roles.py`，`python -m rva.cli --roles` 可打印。

---

## 架构与决策

| 层次 | 选定方案 |
|------|----------|
| 输入锚点 | **(question, answer) 成对输入**；question 用于给 claim 定锚（off-topic → `D5`），非流程阶段 |
| 流水线设计 | 陈述抽取（事实 / 意见等分类）→ 向量检索 + NLI 蕴含判定 → 透明、可质疑的输出 |
| 代码与规则 | **纯 Python + Pydantic**，结构化数据 + 显式规则评估 |
| 评估指标 | **Factuality 事实性**（五级判定，锚定外部可靠来源；默认 deterministic critic 本地离线，判决不外包给 grounding judge） |
| 可选复核 | **Grounding 接地性**（Ragas / TruLens judge，锚定传入的检索上下文，单独报告） |
| 诚实性 | **Honesty 诚实性**（H1–H5 披露检查：无依据断言 / 传播度 / 冲突 / 范围 / 问题相关性） |
| 修订 | **Reviser 修订**（可选，`--revise`；原回答永不被就地修改） |

> **输入锚点：question-answer pair**。question 是**架构级输入**，不是流程阶段：
> critic 抽取 claim 时以它为锚点（与 LLM Auditor 一致，(question, answer) 成对进入）。
> 每条 claim 记录 `question_anchor`（词法重叠率 `relevance` + `reactant`）；与问题零共享词的
> 知识性 claim 判为 off-topic（判据 `D5`）。下游 7 个阶段不变，只是全部继承了问题锚点。

```
输入：question（锚点） + LLM answer + 检索到的文档
        │
        ▼
┌───────────────────────────┐
│ 1. 认知解构 decompose      │  事实 / 逻辑推理 / 主观意见 / 推论；每条 claim 附
│                             │  question_anchor（相关性 / reactant）
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ 2. 证据与来源验证 verify   │  向量检索（ADK RAG 式 top-k）+ NLI 三态判定、来源分级、
│                             │  领域可信度、流行度偏置、矛盾检测
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ 3. 透明分层输出 output     │  可信度评分、取舍标准、边界条件、
│                             │  分层证据视图
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ 4. 事实性评估 evaluate     │  每条陈述核对外部可靠来源（五级判定，本地）
└───────────────────────────┘
        │
        ▼  （可选，--judge ragas/trulens）
┌───────────────────────────┐
│ 5. Grounding 复核 judge    │  是否扎根于传入的检索上下文；单独报告，不改写五级判决
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ 6. 诚实性 honesty          │  H1–H5 披露检查：无依据断言 / 传播度 / 冲突 / 范围 / 问题相关性
└───────────────────────────┘
        │
        ▼  （可选，--revise）
┌───────────────────────────┐
│ 7. 修订 revise (Reviser)   │  按判决改写，带 [C#] 标记；原回答不被改写
└───────────────────────────┘
```

## 快速开始

需要 Python ≥ 3.10 与 `pydantic>=2`。

```bash
# 运行内置示例（报告 + 事实性）
python -m rva.cli --answer-file examples/answer.txt --sources examples/sources.json --domain public-health

# 只看事实性 / 只看诚实性
python -m rva.cli --answer-file examples/answer.txt --sources examples/sources.json --factuality
python -m rva.cli --answer-file examples/answer.txt --sources examples/sources.json --honesty

# 带问题（claim 锚点）+ 修订
python -m rva.cli --answer-file examples/answer.txt --question-file examples/question.txt --sources examples/sources.json --revise

# 输出完整 JSON（四个阶段的所有中间结果）
python -m rva.cli --answer-file examples/answer.txt --sources examples/sources.json --json

# 运行测试（仅标准库）
python -m unittest discover -s tests
```

### 交互式校验台（Streamlit）

不想敲命令？用浏览器界面试跑校验（可编辑 sources、切换后端、逐 claim 看证据与修订）：

```bash
pip install streamlit        # 首次使用，或 pip install -e ".[ui]"
python run_ui.py             # 等价于：streamlit run app/streamlit_app.py
```

打开后点「📥 载入示例」→「▶️ 运行校验」，即可看到五级事实性、claim 报告、诚实性、
接地性与修订/复查六个面板；侧边栏可切换 `critic_backend` / `reviser_backend` / `judge`
等后端。UI 只是 `rva.validate(...)` 的渲染层，不含任何新校验逻辑。

### 部署成网页（让别人也能打开）

仓库已备好部署产物（`requirements.txt` / `.streamlit/config.toml`），两条路可选，
完整步骤见 `docs/deploy.md`：

- **局域网 / 临时分享（零账号）**：`python run_ui.py --server.address 0.0.0.0`，
  同网段的人访问 `http://<你的内网IP>:8501` 即可（首次可能需放行防火墙端口）。
- **公网（Streamlit Community Cloud，免费）**：把仓库推到 GitHub → 在
  share.streamlit.io 连接该仓库 → 主文件填 `app/streamlit_app.py` → Deploy，
  得到形如 `https://<app>.streamlit.app` 的公开链接。**默认全离线可跑，不配密钥
  也能体验**；要启用 LLM / judge 轨，在站点的 Secrets 里填 `OPENAI_API_KEY`（云端
  会把 `st.secrets` 自动桥接进配置，本地 `.env` 行为不变）。

作为库使用：

```python
from rva import validate, Source, format_report, format_factuality, format_honesty

sources = [Source(id="S1", url="https://www.who.int/x", text="……")]
result = validate(
    question="空气污染每年导致多少人过早死亡？",   # 必需：claim 锚点
    answer="根据我检索到的资料，空气污染每年导致约420万人过早死亡……",
    sources=sources,
    domain="public-health",
    revise=True,                                    # 可选：产出修订文本
)

print(format_report(result.report))
print(format_factuality(result.factuality))
print(result.factuality.score)        # 0..1  (result.faithfulness 为兼容别名)
print(format_honesty(result.honesty)) # 诚实性披露检查（独立报告）
print(result.grounding)               # Grounding 报告，judge=local 时为 None
print(result.revision.revised_text)   # 修订文本，未开 revise 时为 None
```

## Factuality 事实性（首要指标：五级判定）

每条 claim 对照**外部可靠来源**判为五级之一（标签沿用 ADK `hallucinations_v1` 与 LLM Auditor 的命名）：

> **锚点差异（关键）**：ADK 的 `hallucinations_v1` 锚定**上下文 / 工具输出**（回答是否忠实于"被告知的内容"，即内部忠实性）；本项目锚定**外部可靠来源**（回答是否为真，即事实效度）。因此本项目的 `Unsupported` 可能事实上为真，只是检索集中没有可靠来源证实它；两者分数**不可直接比较**。

| Verdict | 含义 | 对分数的贡献 |
|---|---|---|
| `Accurate` | 可靠来源证实 | 计入正确 |
| `Inaccurate` | 可靠来源反驳 | 0 |
| `Disputed` | 可靠来源彼此冲突 | 0 |
| `Unsupported` | 找不到可靠来源证实 | 0 |
| `Not Applicable` | 主观意见 / 个人信念，无需外部核验 | 计入正确 |

```
score = (n_accurate + n_not_applicable) / n_total_claims      # 来源锚定准确率
```

- **锚点是外部可靠来源，不是模型自评**：只有可靠性 ≥ 阈值的来源能判 `Accurate` / `Inaccurate`；当支持与反驳双方都达到阈值时才是 `Disputed`。低可信度博客反驳 WHO 事实，只会导致 `Unsupported`，不会把事实判成 `Inaccurate`。
- **五级中只有 `Not Applicable` 免于外部核验**：主观意见不触发任何 API 调用；事实 / 推论 / 逻辑论证都会被判。
- **五级判决不外包给 grounding judge**：`evaluate_factuality(claims, verifications)` **没有** judge 参数，默认（deterministic critic）`method` 恒为 `local_source_anchored`。任何 grounding judge 的分数都无法进入五级判决——这是刻意设计（见下）。
- **证据效力检查（per-claim）**：每条 claim 输出 `source_quality`（来源权威性评级：official / professional / self_media / unverifiable）+ `freshness`（时效性，动态事实记 `checked_at`、无日期标 `stale`）+ `support_score`（权威加权支持，按一致性违规与流量驱动折减）；另含 `consistency`（数字/限定词篡改、夸大、断章取义）与 `popularity_suspect_source_ids`（流量偏置嫌疑，不因热度默认采信）。
- **Critic 双轨**：`critic_backend="deterministic"`（默认，离线规则）｜`"llm"`（注入 `CRITIC.prompt`，一次完成 claim 抽取 + 五级判定，`method="llm_critic"`）；两轨输出 schema 完全一致（claim / cues / verdict / rationale / evidence）。LLM 轨每条判决必须带 `justification` + 来源 `evidence`，否则回退规则轨。
- **已知覆盖限制**：`Unsupported` 只表示**当前检索集**中没有可靠来源证实该 claim，**不代表它假**。判定有多"外部"，取决于证据管线的覆盖广度。

### Critic 双轨后端（deterministic / llm / auto）

Critic 后端由 `critic_backend=`（或 `RVA_CRITIC`）选择，四条约定：

1. **`auto` 语义**：`critic_backend="auto"` 只有 **key 且 sources 同时存在**时才解析为 `llm`；缺任一项都回退 `deterministic`。
2. **默认值**：`build_critic` 的 `has_sources` **默认 `False`**——无来源时默认**不开 LLM 轨**（严格 source-gating）。
3. **显式 `llm`**：显式 `critic_backend="llm"` **即使无来源也放行**（调用方明示优先于 `auto`），但记一条 WARNING。
4. **回退留痕**：降级不静默——WARNING 说明原因，`report` 如实记 `critic_backend="deterministic"`。

> `auto` / `llm` 是唯一会用到 key 的两条轨（`deterministic` 完全离线、零成本）；显式值始终优先于 `auto`，因此 `critic_backend="deterministic"` 可完全关闭 LLM 轨。

### Grounding 复核（可选 judge，单独报告）

Ragas / TruLens 只能看到传入的**检索上下文**，它们回答的是"claim 是否扎根于这些上下文"——
ADK 式的**内部忠实性**，而**不是**外部事实效度。因此它们接的是 **grounding 钩子**，结果放进
`result.grounding`，与上面的五级判决**分开报告**：

```
factuality : 五级判决（Accurate / Inaccurate / Disputed / Unsupported / Not Applicable）— 锚定外部可靠来源
grounding  : {grounded | ungrounded | 无判决} + judge 原始分数              — 锚定传入的检索上下文
```

为什么必须拆成两份：一个"扎根于上下文"的 claim 完全可能**在现实中为假**（如果检索集本身偏窄/有偏），
所以 judge 分数**绝不能**覆盖五级判决。旧版会把 judge 的 0.5 直接改写成 `Unsupported`——这正是本次拆分要修掉的错误。

- **判定规则**：分数 ≥ `GROUNDED_THRESHOLD`(0.75) → `grounded`，否则 `ungrounded`；`None`（无上下文 / NaN / 后端失败）→ **不下判决**。
- **类型过滤**：只有知识性陈述（事实 / 推论 / 逻辑论证）进入 judge；主观意见直接跳过、不消耗 API 调用，任何针对意见的 judge 回包都被丢弃。
### 接入真实的 Ragas / TruLens（grounding-only judge）

已内置两个可选后端（`rva/integrations.py`）。它们需要联网 + LLM API key，且只产出 grounding 复核：

```bash
pip install -e ".[judges]"          # 安装 ragas / trulens / openai（版本已固定在可用组合）
# 把 key 写进项目根目录的 .env（见下节），或：
export OPENAI_API_KEY=sk-...        # Windows: $env:OPENAI_API_KEY="sk-..."

# 运行后，报告里 factuality（五级判决）与 grounding（judge）是两段
python -m rva.cli --answer-file examples/answer.txt --sources examples/sources.json --judge ragas
python -m rva.cli --answer-file examples/answer.txt --sources examples/sources.json --judge trulens
python -m rva.cli --answer-file examples/answer.txt --sources examples/sources.json --judge ragas --grounding   # 只看 grounding

# 查看后端是否可用（不调用 API）
python -m rva.cli --judge-status
rva --judge-status
```

编程方式：

```python
from rva import validate, decompose, evaluate_factuality, evaluate_grounding
from rva.integrations import build_judge
from rva.stages.verify import verify

result = validate(answer=answer, sources=sources, judge="ragas")   # 或 "trulens"
result.factuality   # 五级判决（本地、来源锚定）
result.grounding    # Grounding 报告；judge="local" 时为 None

# 底层：把 judge 接到 grounding 钩子（factuality 侧没有 judge 钩子）
claims, _ = decompose(answer)
verifications = verify(claims, sources)
g = evaluate_grounding(claims, verifications, batch_judge=build_judge("ragas", sources))

# 自定义 judge：返回 {claim_id: record}；record 可写 (grounded, score, rationale) 位置元组、
# 带同名字段的 dict，或暴露同名属性的对象。回包形状不合法会显式 raise ValueError；
# 只有 None / NaN / 无上下文才算"无判决"
def my_judge(claim, verification):
    ...
evaluate_grounding(claims, verifications, judge=my_judge)
```

**健壮性**：Ragas / TruLens 依赖网络与 API key。若缺 key、网络失败、或 judge 返回 `NaN`，
对应陈述会标记为**"无 grounding 判决"**（`grounded=None`），绝不崩溃，也**不影响 factuality**。

### 阶段二的检索与蕴含判定（向量检索 + NLI）

阶段二不再用词重叠匹配，而是两步：

1. **向量检索**（`rva/retrieval.py`）：把来源按句切块后建索引，对每条陈述取 `similarity_top_k` 个最相关块（默认 5），并丢弃相似度低于 `vector_distance_threshold` 的候选。接口对应 Google ADK 的 RAG 引擎（`VertexAiRagRetrieval` 的 `similarity_top_k` / `vector_distance_threshold`）。
2. **NLI 三态判定**（`rva/nli.py`）：判断每个召回块与陈述的关系是 `support`（蕴含）/ `refute`（反驳）/ `neutral`（中立），据此归入 `supports / contradicts / partials`；被阈值筛掉的来源进入 `screened`，可复查淘汰理由。

后端可切换（默认完全离线）：

| 变量 | 取值 | 说明 |
|------|------|------|
| `RVA_RETRIEVER` | `hash`（默认）/ `openai` / `auto` | `hash` 为确定性离线向量（签名字符 n-gram + 语料 IDF）；`openai` 用真实 embedding |
| `RVA_NLI` | `heuristic`（默认）/ `llm` / `auto` | `llm` 用 LLM judge 做三态蕴含（与 Ragas/TruLens 相同的接线） |
| `RVA_EMBED_MODEL` | 默认 `text-embedding-3-small` | `openai` 检索使用的 embedding 模型 |

```bash
python -m rva.cli --answer-file examples/answer.txt --sources examples/sources.json --retriever openai --nli llm
```

```python
validate(answer=answer, sources=sources, retriever="openai", nli="llm")
```

缺 key 或网络失败时自动退回离线后端，不崩溃。

**版本注意**：`ragas 0.2.15` 需要较旧的 `langchain 0.3.x` 核心（最新的 `langchain-community`
移除了 ragas 硬依赖的 `chat_models.vertexai`）。`pyproject.toml` 的 `[judges]` 已锁定经过验证的组合。

## 配置（.env）

项目根目录有一个 `.env` 配置文件，**把你的 key 写进去即可**，程序会自动读取：

```
# .env
OPENAI_API_KEY=sk-你的key
OPENAI_BASE_URL=          # 可选：代理或 OpenAI 兼容端点，留空用默认
RVA_JUDGE_MODEL=gpt-4o-mini
RVA_JUDGE=local           # grounding 后端：local（关）| ragas | trulens
RVA_RETRIEVER=hash        # 阶段二检索：hash（离线）| openai | auto
RVA_NLI=heuristic         # 阶段二蕴含判定：heuristic（离线）| llm | auto
RVA_EMBED_MODEL=text-embedding-3-small
RVA_DOMAIN=general
RVA_REVISE=0              # 1/true 时默认产出修订文本
```

- `.env` 已在 `.gitignore` 中，**不会被提交**。模板见 `.env.example`。
- 优先级：真实环境变量 > `.env` 文件 > 内置默认值。
- 查看当前配置（key 会脱敏）：`python -m rva.cli --show-config`
- 查看后端可用性：`python -m rva.cli --judge-status`

> 注意：仅 `ragas` / `trulens`（grounding 复核）以及 `openai` / `llm`（阶段二检索与 NLI）需要 key；`local` 与 `hash` / `heuristic` 后端完全离线，不需要任何密钥。

## 输出契约（完整结构）

> 独立的对外契约文档：[`docs/output-contract.md`](docs/output-contract.md)（供外部对接方引用；
> 文中字段/枚举与代码由 `tests/test_contract_doc.py` **交叉校验**，不会漂移）。

`validate()` 返回一个 `ValidationResult`；顶层字段与四份独立报告如下。所有产物均为 Pydantic 模型，
`model_dump_json()` 可直接导出。

### 顶层 `ValidationResult`

| 字段 | 类型 | 说明 |
|------|------|------|
| `answer` / `question` | str | 原始回答 / 输入锚点（原回答永不被改写） |
| `domain` / `source_count` | str / int | 领域、检索来源数 |
| `critic_backend` | str | 报告级追溯：实际产出判决的 critic 后端（`deterministic` \| `llm`） |
| `stage1_stats` | `DecompositionStats` | `total` / `fact` / `reasoning` / `opinion` / `inference` |
| `claims` | `Claim[]` | 认知解构产物（含 `question_anchor`） |
| `verifications` | `Verification[]` | 证据与来源验证（source 映射所在处） |
| `report` | `Report` | 透明分层输出 |
| `factuality` | `FactualityResult` | **报告①**：五级判定（本地、来源锚定） |
| `grounding` | `GroundingResult \| null` | **报告②**：接地性复核（judge=local 时为 null） |
| `honesty` | `HonestyResult` | **报告③**：诚实性披露检查 |
| `revision` | `RevisionResult \| null` | **报告④**：修订输出（未开 `revise` 时为 null） |
| `summary` | `ValidationSummary \| null` | 可选汇总：五级分布 / 通过率 + grounding 0–100 + honesty 比例（只重述，**不是**总分；`summary=True` 才附） |

### ① 五级 verdict — `factuality` (`FactualityResult`)

| 字段 | 说明 |
|------|------|
| `score` | 来源锚定准确率 `(accurate + not_applicable) / total_claims` |
| `accurate` / `inaccurate` / `disputed` / `unsupported` / `not_applicable` | 五级计数 |
| `total_claims` / `verifiable_claims` / `applicable_claims` | 总数 / 进入判定的知识性 claim / 非 exempt |
| `method` | 恒为 `local_source_anchored`（判决不可外包） |
| `verdicts[]` | `ClaimVerdict[]`，见下 |

`ClaimVerdict`：`claim_id` · `claim_text` · `claim_type` · `verdict`（五级）· `score`（该判决的规范值 0/1）·
`evidence_domains[]` · `rationale`（人类可读理由）。

### ② 双栏：grounding / factuality 分开报告

两者**从不同锚点回答不同问题**，绝不互相覆盖：

| 报告 | 锚点 | 问的问题 | 取值 |
|------|------|----------|------|
| `factuality` | **外部可靠来源**（事实效度） | 陈述是否为真？ | 五级 verdict（`accurate`/`inaccurate`/`disputed`/`unsupported`/`not_applicable`） |
| `grounding` | **传入的检索上下文**（内部忠实性） | 陈述是否扎根于给定上下文？ | `grounded` / `ungrounded` / 无判决（+ judge 原始分数 0..1） |

`GroundingResult`：`method` · `judged_claims` / `grounded_count` / `ungrounded_count` / `skipped_claims` /
`total_claims` · `mean_score` · `items[]`。
`ClaimGrounding`：`claim_id` · `claim_text` · `claim_type` · `grounded`（bool/None）· `score`（float/None）· `rationale`。

### ③ 分层审计日志

**`report`（Stage 3，`Report`）**

| 字段 | 说明 |
|------|------|
| `summary` | `total` / `kept` / `demoted` / `dropped` / `contextualized` / `avg_credibility` |
| `items[]` | `ReportItem[]`，逐 claim 见下 |
| `criteria` | 全部判据定义（`K*`/`D*`/`X*`/`N*`/`D5`） |
| `supplemented_text` | 开了 `revise` 时=修订文本，否则 null |

`ReportItem`：`claim_id` · `text` · `type` · `credibility`(0..1) · `decision`(`keep`/`demote`/`drop`/`contextualize`) ·
`decision_label` · `criteria[]` · `boundary_conditions[]` · `layers` · `popularity_bias` · `decomposition` · `challengeable`。

`layers`（`Layers`）：`l1_primary`（高可靠一手证据）/ `l2_secondary`（弱/部分证据）/ `l3_conflicting`（冲突来源）——
每一项都是 `EvidenceRef`。

**逐 claim 溯源双源**

- `claims[]`（Stage 1，`Claim`）：`id` · `text` · `type` · `confidence` · `cues[]` · `score_breakdown` · `note` ·
  `verifiable` · **`question_anchor`**（`question` / `relevance` / `reactant` / `note`）。`cues` 与
  `score_breakdown` 是**分类为何如此**的凭证。
- `verifications[]`（Stage 2，`Verification`）：`claim_id` · `claim_text` · `claim_type` · `evidence` ·
  `support_score` · `independent_domains` · `conflict` · `popularity_bias` · `retrieval_backend` · `nli_backend` ·
  `screened_count` · **`question_relevance`** · **`off_topic`**。

**source 映射**在 `verification.evidence`（`Evidence`）：

| 字段 | 说明 |
|------|------|
| `supports` / `partials` / `contradicts` | 三态判定归属：支持 / 中立 / 反驳 |
| `screened` | 被相似度阈值筛掉的候选（**保留**，可复查淘汰理由） |
| `claim_negation` | 命中的否定式（若有） |

`EvidenceRef`（每条来源）：`source_id` · `title` · `url` · `domain` · `reliability`(0..1) · `tier` ·
`similarity` · `coverage` · `shared_tokens` · `traffic` · `negation` · `stance`。

### ④ 诚实性凭证 — `honesty` (`HonestyResult`)

| 字段 | 说明 |
|------|------|
| `score` | 适用披露检查的通过比例（0..1） |
| `passed` | `score >= 0.7` 且无 `H1` 硬失败 |
| `flags[]` | `HonestyFlag`：`code`(`H1`–`H5`) · `label` · `detail` · `claim_ids[]` |
| `checked_claims` / `exempt_claims` | 受检 claim 数（意见豁免） |
| `method` | `local_heuristic` |

| 代码 | 检查 |
|------|------|
| `H1` | 无依据断言未加限定（硬失败） |
| `H2` | 传播度驱动未坦白 |
| `H3` | 来源冲突未承认 |
| `H4` | 检索范围未说明 |
| `H5` | claim 未回应问题（需 question） |

### ⑤ 修订输出 — `revision` (`RevisionResult`)

| 字段 | 说明 |
|------|------|
| `revised_text` | 修订后文本：改动过的 claim 带 `[C#]` 标记可回溯；未改动的原样保留；被剔除的 claim 移入 `[Validator supplement]` 块 |
| `supplemented` | 是否由外部 reviser 后端生成 |
| `changed` | 被改动（annotate/qualify/drop）的 claim 数 |
| `revision_log[]` | `RevisionItem`：`claim_id` · `action`(`keep`/`annotate`/`qualify`/`drop`) · `original_text` · `revised_text` · `reason` · `criteria[]` |
| `method` | `local_rewriter` 或 `llm_reviser` |
| `diff` / `input_verdicts` | 修订前后统一 diff + Reviser 据以改写的 verdict 清单（两轨都落） |
| `end_marker` / `end_marker_present` | LLM 轨编辑终止标记 `---END-OF-EDIT---` 及其是否出现 |
| `fallback` | LLM 后端失败并回退到确定性改写器 |
| `recheck` | 修订→复查闸门报告 `RecheckResult`（未跑为 null） |

**双轨后端 + 修订→复查闸门**：`local`（默认，离线可审计、零成本）与 `llm`（编辑人设，`REVISER.prompt`）共享同一
契约。因 LLM 修订是**新生成文本**，每次修订后都对修订文本**重跑来源锚定的五级事实性判定**；若引入
「新增未获证 claim」（inaccurate / disputed / unsupported）且分数低于阈值，则再改一轮（最多 2 轮），保留分数最高的候选——闸门不会让产出比首轮更差。

### 格式选项

CLI 选择性输出：`--factuality`（报告①）· `--grounding`（报告②，需 `--judge ragas/trulens`）·
`--honesty`（报告③）· `--revise` / `--revised-text-only`（报告④）· `--json`（完整 `ValidationResult`）。
默认输出依次打印 `report` → `factuality` → `grounding` → `honesty` → `revision`。

## 决策标准（明示、可质疑）

| 类别 | 规则 |
|------|------|
| 保留 keep | `K1` 有可靠性≥0.7来源支持 · `K2` 非流量驱动 · `K3` 无同等/更高可靠性来源反驳 · `K4` 推理前提得到可靠支持 |
| 降级 demote | `D1` 主观意见或无依据推论 · `D2` 仅低可靠来源弱支持 · `D3` 依赖流行度 · `D4` 推理前提缺乏支持 · `D5` 未回应用户问题（off-topic） |
| 剔除 drop | `X1` 被更高可靠性来源反驳 · `X2` 可证伪的事实陈述毫无证据 |
| 语境化 contextualize | `N1` 主观意见不作为既定事实 · `N2` 明确标注其主观属性 |

## 项目结构

```
rva/
  models.py            Pydantic 模型（全部中间与最终产物均为强类型）
  text.py              分词、分句（中英混合，CJK 用字符二元组）
  retrieval.py         向量检索（ADK RAG 式 top-k / 距离阈值，可插拔 embedding）
  nli.py               三态蕴含判定（支持 / 反驳 / 中立）
  stages/
    decompose.py       阶段1：认知解构
    verify.py          阶段2：证据与来源验证
    output.py          阶段3：透明分层输出
  critic.py            Critic 双轨后端（deterministic 规则 / llm 人设）
  efficacy.py          per-claim 证据效力检查（权威性 / 时效性 / 一致性 / 流量偏置）
  evaluate.py          事实性评估（五级判定）+ grounding 复核（可插拔 judge）
  honesty.py           诚实性与可审计性披露检查（H1–H5）
  revise.py            修订阶段（Reviser 双轨后端 + 修订→复查闸门，可选）
  roles.py             角色设定（Critic / Reviser 双阶段 + 共享伦理判准）
  integrations.py      Ragas / TruLens grounding 后端适配器
manifest.yaml          ADK 风格 recipe 元信息
SKILL.md               检索内容校验 skill 的元信息与说明
agents/openai.yaml     skill UI/调用策略
.codex/agents/         子代理外壳（critic.toml / reviser.toml，转发 rva/roles.py 的人设）
  config.py            .env / 环境变量配置加载
  cli.py               命令行入口
  __init__.py          validate() 编排
examples/
  answer.txt           示例 LLM 回答
  sources.json         示例检索结果（含高/低可靠性来源）
app/
  streamlit_app.py     Streamlit 交互式校验台（validate() 的可视化渲染层）
run_ui.py              启动校验台（等价于 streamlit run app/streamlit_app.py）
requirements.txt       Streamlit Cloud 构建依赖（含可选 judge 的注释行）
.streamlit/config.toml Streamlit 运行/主题配置（headless、关闭 usage stats）
tests/
  test_pipeline.py     行为测试（unittest）
docs/
  design.md            设计与方法
  ethics.md            伦理评估
  output-contract.md   对外输出契约（与代码交叉校验）
  deploy.md            部署指南（局域网分享 / Streamlit Cloud）
```

## 设计取舍

- **纯 Python + Pydantic**：所有产物强类型、可序列化、可校验；`model_dump_json()` 即可导出。
- **离线、零额外依赖、可审计**：分类与验证用显式规则和词典，任何人可复查"为什么"。
- **可靠性加权而非计数加权**：一条 WHO 证据胜过十条博客，直接对抗"流量即真相"。
- **向量检索 + NLI 三态判定**：先按语义召回证据块，再判 support / refute / neutral，替代词重叠匹配。
- **区分"程序性否定"与"断言性否定"**：避免把"尚未经同行评审"误判为对结论的反驳。
- **事实性与报告口径一致**：低可信度反驳不压低事实性判定，避免自相矛盾。

## 已知限制

- 规则/启发式方法对隐含语义、反语、讽刺不敏感，属于"辅助审查"而非最终裁决。
- 证据匹配已改为向量检索 + NLI 三态判定。默认离线后端（hash 向量 + 启发式 NLI）语义能力弱于真实 embedding/LLM；可切换 openai / llm 提升。
- 若需更强的**语义接地（grounding）**复核，可接入 Ragas / TruLens judge（独立报告）；但它锚定的是传入的检索上下文，**不能**替代外部事实效度判定。
- 可信度评分与五级判决都只反映**当前检索集**，不代表整个文献；`Unsupported` 不等于"假"。

## 许可

MIT
