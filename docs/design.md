# 设计与方法

## 1. 问题陈述

搜索引擎后端对结果排序时，主要依据**语义相似性**与**商业/流量因素**，缺乏对信息
真实性的验证。LLM 在此基础上进一步筛选（选择、保留、丢弃），会继承并放大这些偏置，
产生认知风险。本 Agent 的立场是：**LLM 应为自身的筛选决策负责**——评估证据质量、
给出可追溯且可质疑的推理、明示不确定性、拒绝无依据的断言，并把认知准确性置于用户满意度之上。

技术栈按既定决策采用 **纯 Python + Pydantic**：所有中间与最终产物都是强类型模型
（`rva/models.py`），可序列化、可校验、可复查。首要评估指标为 **Factuality 事实性**（锚定外部可靠来源；旧名 Faithfulness 保留为别名），并附**诚实性**披露检查（§7）与**可选修订**（§8）。

### 1.1 输入锚点：question（架构级输入，非流程阶段）

本 skill 与 LLM Auditor 一致，读的是 **(question, answer) 对**。question 不是流水线的第 8 个阶段，
而是**输入侧锚点**：critic 抽取 claim 时以它为基准，下游 7 个阶段都继承该锚点。缺少 question 时
claim 无锚点，无法判断相关性（详见 §6）。因此 `validate()` 的签名以 `question` 为正式入参：

```
validate(question, answer, sources, ...)   # (question, answer) 成对进入
```

## 2. 阶段一：认知解构（rva/stages/decompose.py）

输入：question（锚点）+ LLM 的原始回答。
输出：一组带类型的 `Claim`，每条附 `question_anchor`。

1. **分句**：剥离 markdown 标记，按中英文句末标点切分（`text.split_claims`）。
2. **四分类**：每条陈述对 `fact / reasoning / opinion / inference` 打分：
   - 事实：量化值、年份、归因短语、专有名词。
   - 逻辑推理：强演绎连接词（因此/所以）权重高，弱因果动词（导致/leads to）权重低。
   - 主观意见：第一人称观点、价值判断、情感词、should。
   - 推论：模糊限制词（可能/推测/suggests/likely）。
3. **置信度**：由最高分与次高分之差归一化，反映分类"清晰程度"而非"正确概率"。
   所有触发词记录在 `cues` 中，可复核。

## 3. 阶段二：证据与来源验证（rva/stages/verify.py）

### 3.1 来源可靠性分级

`DEFAULT_REGISTRY` 按**来源类型**给出 0..1 的可靠性：

| 层级 | 示例 | 分数 |
|------|------|------|
| primary-official | .gov / WHO / gov.cn | 0.94–0.95 |
| peer-reviewed | nature / science / NEJM | 0.92–0.93 |
| academic | .edu | 0.90 |
| wire | Reuters / AP / 新华社 | 0.85 |
| major-news | BBC / NYT / 人民网 / 央视 | 0.75–0.82 |
| preprint | arXiv | 0.70 |
| encyclopedic | Wikipedia | 0.65 |
| portal | 新浪 / 搜狐 / 头条 | 0.50–0.55 |
| blog / ugc / social | Medium / 知乎 / 微博 | 0.30–0.45 |
| unknown | 未分类 | 0.50 |

**匹配规则**：按 host 逐级去后缀匹配（`news.cctv.com` → `cctv.com` → `com`），
精确键优先；再退回调用方的 `tag`。因此**伪装域名**（如 `gov.cn.evil.com`）不会继承
可信标签，只会落到 `unknown`。

**关键阈值**：只有可靠性 ≥ `RELIABILITY_THRESHOLD`(0.7) 的来源能判 `Accurate` /
`Inaccurate`。`unknown`(0.50) 与 `portal`(0.50–0.55) **低于**该阈值——这不是判定
失败，而是"该来源尚未被评级，故不足以佐证claim"。需要引入新域名时，可扩展内置表，
或用 `registry` 参数整体替换为真实评级服务。

### 3.2 证据匹配：向量检索 + NLI 蕴含判定

不再使用词重叠匹配，改为两步（对标主流 claim-verification 流水线与 Google ADK 的 RAG 引擎）：

1. **向量检索**（`rva/retrieval.py`）：来源按句切块（`chunk_text`）后建索引；对每条陈述取
   `similarity_top_k`（默认 5）个最相关块，丢弃相似度低于 `vector_distance_threshold` 的候选。
   接口对应 ADK `VertexAiRagRetrieval` 的 `similarity_top_k` / `vector_distance_threshold`。
   - `HashingEmbedder`：确定性离线向量（签名字符 n-gram + 语料 IDF，L2 归一化），默认后端。
   - `OpenAIEmbedder`：真实 embedding（`RVA_EMBED_MODEL`，默认 `text-embedding-3-small`）。
2. **NLI 三态判定**（`rva/nli.py`）：对每个召回块判断与陈述的关系
   `support`（蕴含）/ `refute`（反驳）/ `neutral`（中立），据此归入 `supports / contradicts / partials`。
   - `LLMNLIJudge`：LLM judge 输出三态标签（与 Ragas/TruLens 相同的接线方式）。
   - `HeuristicNLI`：离线线索判定（断言级否定 + 共享 token 覆盖率），默认后端。

被距离阈值筛掉的来源不会消失，而是进入 `evidence.screened`，可复查"为什么被淘汰"。

### 3.3 矛盾检测：区分两类否定（`HeuristicNLI` 内部使用）

- **断言性否定**（strong）：没有 / 并非 / 不是 / 夸大 / 辟谣 / debunked …… 窗口 14 字符。
- **程序性否定**（weak）：未 / 尚未 / not yet …… 窗口 4 字符；
  如"尚未经同行评审"不应被当作对结论的反驳。

### 3.4 流行度偏置检测

对支持/部分支持的来源，若**平均流量 >0.6 且平均可靠性 <0.7**，标记 `popularity_bias.biased`
——"因为流行而被选中，而不是因为可靠"。

### 3.5 证据效力检查（rva/efficacy.py，per-claim）

在 Stage 2 之后、Stage 3 之前，对每条 claim 跑四项离线检查，产出 `source_quality` / `freshness` /
`support_score`（外加 `consistency` 与 `popularity_suspect_source_ids`），挂在 `Verification` 上。

| 检查 | 输出 | 说明 |
|------|------|------|
| 来源权威性评级 | `SourceQuality.grade` | `official` / `professional` / `self_media` / `unverifiable`；取实际参与论证来源的最高级 |
| 时效性 | `Freshness` | 动态事实记 `checked_at`；无日期或过旧 → `stale` |
| 一致性比对 | `Consistency` | 数字缺失 → `number_mismatch`；限定词强于来源 → `qualifier_overreach` |
| 流量 vs 效度 | `popularity_suspect_source_ids` | 高流量低可靠来源进支持集 → 流量偏置嫌疑，联动 `D3`/`H2` |

`support_score` = 支持来源可靠性均值 ×（一致性违规 0.7）×（流量驱动 0.85）；无违规时与旧值逐字一致。
两轨 critic 共用本检查（`LLMCritic` 与 `DeterministicCritic` 都调用 `verify()`）。

## 4. 阶段三：透明分层输出（rva/stages/output.py）

### 4.1 决策

`keep / demote / drop / contextualize` 四类，每类由明示规则（`K* / D* / X* / N*`）驱动，
规则随结果一并返回。

### 4.2 可信度评分

```
score = 可靠性加权支持度
      + min(0.15, 0.05 × 独立来源数)      # 独立佐证加成
      - 0.25 × 反驳来源数                 # 冲突惩罚
      - 0.15（若流量驱动）
类型修正：opinion ≤0.15；inference ×0.85；reasoning ×0.9；无支持 ≤0.1
```

### 4.3 分层证据视图与边界条件

- `l1_primary`：可靠性 ≥0.7 的支持证据
- `l2_secondary`：部分相关的弱证据
- `l3_conflicting`：相冲突来源
- 自动生成边界条件，如"这只是推论""仅有单一领域佐证""可信度只反映当前检索集"。

## 5. 评估指标：Factuality 事实性 — 五级判定 + 来源锚定准确率（rva/evaluate.py）

每条 claim 对照**外部可靠来源**判为五级之一（标签沿用 ADK `hallucinations_v1` 与 LLM Auditor 的命名）：

> **锚点差异（关键，勿混淆）**：ADK 的 `hallucinations_v1` 锚定的是**上下文 / 工具输出**，衡量的是
> "回答是否忠实于被告知的内容"（内部忠实性 / groundedness）。本项目锚定的是**外部可靠来源**，
> 衡量的是"回答是否为真"（事实效度）。因此本项目的一个 `Unsupported` 可能**事实上为真**，
> 只是当前检索集中没有可靠来源证实它；两者的分数**不可直接比较**。ADK 的 `not_applicable`
> 也比本项目宽（凡无法用上下文核验者皆可），本项目仅把主观意见设为 `Not Applicable`。

| Verdict | 借用 ADK 标签 | 含义（锚点＝外部可靠来源） | 分数 |
|---|---|---|---|
| `Accurate` | `supported` | 可靠来源证实 | 1 |
| `Inaccurate` | `contradictory` | 可靠来源反驳 | 0 |
| `Disputed` | `disputed` | 可靠来源彼此冲突 | 0 |
| `Unsupported` | `unsupported` | 无可靠来源证实 | 0 |
| `Not Applicable` | `not_applicable` | 主观意见/信念，无需核验 | 1 |

```
score = (n_accurate + n_not_applicable) / n_total_claims      # 来源锚定准确率
```

**锚点是外部可靠来源**（而非模型自评）：

- 只有可靠性 ≥ `RELIABILITY_THRESHOLD`(0.7) 的来源能判 `Accurate` / `Inaccurate`；
- 支持与反驳双方均达阈值 → `Disputed`；
- 仅低可靠来源反驳 → `Unsupported`（不升级为 `Inaccurate`）；
- 有相关来源但均不达阈值 → `Unsupported`。

**类型过滤**：`VERIFIABLE_TYPES = (fact, inference, reasoning)`。事实、推论、逻辑论证都可以被判；
只有主观意见免于核验，直接判 `Not Applicable`（不触发 API 调用）。过滤在
`evaluate_factuality` 的输入侧生效，适配器内部再兜底。

**五级判决不外包给 grounding judge**：`evaluate_factuality(claims, verifications)` **没有** judge 参数，
判据只有 `_verdict_for`：来源可靠性分档。任何 grounding judge（Ragas / TruLens）的分数都无法进入五级判决——
这是刻意设计，因为"是否扎根于给定上下文"与"是否与外部事实相符"是两个问题。
默认（deterministic）轨 `method` 恒为 `local_source_anchored`。

**Critic 双轨（新增）**：critic 本身可选后端（见 §2.1）。`critic_backend="llm"` 时五级判决由 LLM 产出
（`method="llm_critic"`），但 **grounding judge 仍然无法写入判决**——两条通道彼此独立。LLM critic 的
判定必须逐条带 `justification`（写入 `rationale`）与来源 `evidence`（写入 `evidence_domains`），
未获证的判决缺一即回退规则轨。

**已知覆盖限制**：`Unsupported` 只表示**当前检索集**中没有可靠来源证实该 claim，不代表它假。
判定能有多"外部"，取决于证据管线覆盖得多广。

### 5.0 Critic 双轨（rva/critic.py）

| 后端 | 选择方式 | 行为 |
|------|----------|------|
| `deterministic`（默认） | `critic_backend="deterministic"` / `RVA_CRITIC=deterministic` | `decompose → verify → evaluate_factuality`；离线、可审计、零成本；`method="local_source_anchored"` |
| `llm` | `critic_backend="llm"` / `critic=LLMCritic(...)` | 注入 `CRITIC.prompt` + few-shot，模型一次完成 claim 抽取 + 五级判定（对齐 LLM Auditor critic）；`method="llm_critic"` |
| `auto` | `critic_backend="auto"` | 有 key **且** 有来源才用 llm，否则 deterministic（无来源时安全回退、不报错、记 WARNING）；显式 `llm`/`deterministic` 始终优先于 `auto`；显式 `llm` 即使无来源也放行（调用方明示），但同样记 WARNING |

报告级 `ValidationResult.critic_backend` 记录**实际**产出 claim + 判决的后端（回退时记 `deterministic`），供追溯；`rationale` 复用 `justification`，不另存字段。

**schema 完全一致**：两轨都返回 `Claim`（含 `cues` / `note`）、`Verification`（证据分层）、`FactualityResult`
（`verdicts[]` = `verdict` + `rationale` + `evidence_domains`）。LLM 轨照样填 `Claim.cues`（LLM 给的
`cues` 或规则兜底）与 `ClaimVerdict.rationale`（= `justification`），下游 `render` / `honesty` / `revision`
一字不改。

**可证伪性标准（审计）**：每条 claim 必须带 `justification` 与 `evidence`；`accurate/inaccurate/disputed`
**必须至少引用一个来源 id**，否则视为 schema 违规并**整体回退**到规则轨（`fallback=True`）。主观意见
一律强制 `not_applicable`，知识性 claim 不得为 `not_applicable`。

**严格 source-gating**：`build_critic` 的 `has_sources` **默认 `False`**——调用方不确认「有来源可判」时，`auto` 不开 LLM 轨。

**回退留痕（不静默）**：`auto` 无 key 或无来源时**安全回退**规则轨、不报错；LLM 轨任何解析/传输/schema 失败都**整体回退**规则轨——两种情况都记一条 WARNING 日志（`rva.critic` logger），同时 `critic_backend` 如实记为 `deterministic`，降级可见、可追溯，绝不静默。回退还保留完整 claim 级输出（原回答仍被解构、核验、判定）。显式 `llm` 即使无来源也放行（调用方明示优先于 `auto`），但同样记 WARNING。

**不做二次校验**：critic 是"审查"而非"生成新文本"，故不接修订→复查闸门（该闸门只属于 Reviser）。
可审计性由强制的 `justification` + `evidence` 保证。

### 5.1 Grounding-only judge：真实 Ragas / TruLens 后端（rva/integrations.py）

Ragas / TruLens 只能看到 `_contexts_for` 传进去的**检索上下文**，它们回答的是
"claim 是否扎根于这些上下文"——ADK 式的**内部忠实性**，而**不是**外部事实效度。
因此它们接入的是 **grounding 钩子**，结果放在 `ValidationResult.grounding`，与五级判决**分开报告**。

```
factuality : 五级判决（Accurate / Inaccurate / Disputed / Unsupported / Not Applicable）— 锚定外部可靠来源
grounding  : {grounded | ungrounded | 无判决} + judge 原始分数            — 锚定传入的检索上下文
```

- **Ragas**：把每条知识性陈述构造成 `SingleTurnSample(user_input=claim, response=claim,
  retrieved_contexts=证据文本)`，对所有陈述**一次** `evaluate`，读取 Ragas 自身的
  `faithfulness` 分数（这是 Ragas 的指标名，我们按原义当作 grounding 用）。
- **TruLens**：调用 `groundedness_measure_with_cot_reasons(contexts, statement)`，
  其分数区间是 0..3，统一除以 3 归一到 0..1。
- **判定规则**：分数 ≥ `GROUNDED_THRESHOLD`(0.75) → `grounded`，否则 `ungrounded`；
  `None`（无上下文 / NaN / 后端失败）→ **不下判决**，只记录"无 grounding 判决"。
- **接线方式**：`evaluate_grounding(claims, verifications, batch_judge=...)`，`method` 标为
  `external_judge`。绝不读写 `FactualityResult`，所以一个 0.2 的 judge 分数不会把
  `Accurate` 改成 `Unsupported`（旧版会，这正是本次拆分的原因）。
- **类型过滤**：只有知识性陈述进入 judge；主观意见直接跳过，不消耗 API 调用，任何针对意见的
  judge 回包都被丢弃（记 `skipped_claims`）。
- **健壮性**：无 key / 网络失败 / `NaN` 时，对应陈述标记为"无 grounding 判决"（`NaN` 被
  `_finite()` 过滤），绝不崩溃，也不影响 factuality。**但"无判决"专指上述情形**：judge 回包
  形状无法识别（非映射 / 非 `(grounded, score, rationale)` / 非带同名属性的对象）时
  `evaluate_grounding` 显式 `raise ValueError`，不再静默当作"无判决"——否则坏适配器会被
  误读为合法空回包。记录兼容 dict / 属性对象 / 位置元组三种等价写法。

> **术语**：这里的 `grounded` 是 ADK 意义上的 grounding（内部忠实性），所以结果类型叫
> `GroundingResult`，不叫 factuality、也不复用 `Verdict` 五级枚举——两个词汇表刻意不混。
> 旧名 `faithfulness` / `--faithfulness` 仍是 factuality 的兼容别名。

> 版本锁定：`ragas 0.2.15` + `langchain 0.3.27` 系列 + `trulens 2.14`。原因是 ragas 硬 import
> `langchain_community.chat_models.vertexai`，而最新的 `langchain-community` 已移除该模块。
> 这些版本已在 `.venv` 中实测互通（含对本地 OpenAI 兼容桩服务的端到端验证）。

## 6. 输入锚点：question（question-answer pair）

LLM Auditor 读的是 **(question, answer) 对**；本 skill 沿用该契约。`decompose(answer, question=...)`
为每条 claim 记录 `question_anchor`：与问题的**词法重叠率** `relevance` 和布尔 `reactant`。

- 与问题**零共享词**的知识性 claim 记为 **off-topic**（`Verification.off_topic`），在 Stage 3 触发
  `D5` 降级并扣减 credibility，边界条件中显式说明"该 claim 未回应用户的问题"。
- **没有 question 时**不判 off-topic（无法判断），`relevance=0`、`reactant=True`（未知而非否定）。
- 主观意见 exempt：不作为 off-topic 处罚。

为什么必需：claim 是"对某问题的断言"，脱离问题就无法判断相关性；LLM Auditor 正是以此为过滤基准。

## 7. 诚实性与可审计性（rva/honesty.py）

"诚实性"不是真值判断，而是问：**回答是否交代了自己的底细**。五项检查全部离线、可审计，
各自锚定在管线已有的证据上，分数 = 适用检查的通过比例：

| 代码 | 检查 | 触发条件 |
|------|------|----------|
| `H1` | 无依据断言未加限定 | 可验证 claim 无可靠支持，且正文无 hedge 词（`可能/据称/尚未证实`…） |
| `H2` | 传播度驱动未坦白 | `popularity_bias.biased` 但回答未提及 |
| `H3` | 来源冲突未承认 | `conflict` 为真但回答未提及 |
| `H4` | 检索范围未说明 | 回答未出现"检索/现有资料/来源"等范围声明 |
| `H5` | claim 未回应问题 | 有 question 且该知识性 claim 为 off-topic |

四查措辞与字段名（映射表 `rva.honesty.HONESTY_FOUR_CHECKS`；结果落在 `HonestyResult`
上的同名字段，True = 诚实）：

| Step 4 四查字段 | H 码 | 含义 |
|------|------|------|
| `unsupported_rejected` | `H1` | 无依据断言被拒斥或加限定 |
| `sycophancy_checked` | `H2` | 流量驱动支持被核验而非迎合（流量偏置不得默认采信） |
| `conflicts_preserved` | `H3` | 来源冲突被保留而非屏蔽 |
| `uncertainty_marked` | `H4` | 不确定 / 检索范围被显式标注 |

`H5` 是问题相关性检查（不是披露检查），只经 `flags` 报告，不在四查之列。

`H1` 是硬失败项（平铺直叙一个无依据断言 = 不诚实），其余计入比例。主观意见 exempt。
`HonestyResult` 与 `FactualityResult`、`GroundingResult` **三者互补、互不写入**。

## 8. 修订（rva/revise.py，Reviser 角色）

对齐 LLM Auditor 的 Reviser：**审计只判断，修订才改写**。契约要点：

- `revise=True` 才运行；**原回答永不被就地修改**（`out.answer` 始终是原文）。
- 逐 claim 按其判决改写：`keep` 原样、`annotate`（意见）与 `qualify`（降级/off-topic）保留原句但加 `[C#]`
  标记与警示、`drop` 移入 `[Validator supplement]` 块并保留标记与理由。**改动过的 claim 才带标记**，未改动原样通过。
- `revision_log` 每条记录 `action / criteria / reason`，与上游判决一一对应，保证可追溯。
- `reviser=` 可注入自定义后端（如 LLM），失败自动回退确定性改写器；`method` 标为
  `local_rewriter` 或 `llm_reviser`。

### 8.0 双轨后端

| 后端 | 选择方式 | 特点 |
|------|----------|------|
| `local`（默认） | `reviser_backend="local"` / `RVA_REVISER=local` | 确定性、离线、零成本；审计底线，标准 4 的日志凭证**不依赖外部 API** |
| `llm` | `reviser_backend="llm"` / `reviser=LLMReviser(...)` | 编辑人设（`REVISER.prompt`，含 few-shot）；输出须以 `---END-OF-EDIT---` 结束；失败回退本地（`fallback=True`） |
| `auto` | `reviser_backend="auto"` | 有 key 用 llm，否则 local |

两轨共享同一契约（`reviser(text, report) -> str | None`），并**都落审计凭证**：`diff`（修订前后统一 diff）、
`input_verdicts`（据以改写的 verdict 清单）、`end_marker` / `end_marker_present`。

### 8.0.1 修订→复查闸门（`RecheckResult`）

LLM 修订是**新生成的文本**——没闸门就可能引入新幻觉（Wallat 2025：引用忠实 ≠ 正确）。接线：

```
critic(五级 verdict) → reviser(编辑人设 / 确定性改写)
  → 修订文本二次 factuality 复查 → 新增未获证 claim 且低于阈值 → 再改(≤2 轮) → 否则出报告
```

- 每次修订后 `build_recheck_fn` 对修订文本重跑 `decompose → verify → evaluate_factuality`。
- 出现「**新增未获证 claim**」（inaccurate / disputed / unsupported；输入 verdict 未标为未获证、复查新出现的 claim）且分数低于
  `RVA_RECHECK_THRESHOLD`（默认 0.7）时，强制再改一轮，最多 `RVA_RECHECK_MAX_ROUNDS`（默认 2）。
- **保留分数最高的候选**：闸门不会让产出比首轮更差；`rounds[]` 逐轮留痕，`leftover_inaccurate` 报告残留。
- `recheck=False` / `RVA_RECHECK=0` / `--no-recheck` 可关闭（默认开启）。

## 8.1 角色设定（双阶段，对应 LLM Auditor 的 Critic / Reviser）

同一 auditor 兼任两顶帽子，**先后接力**，**共享同一组伦理判准**。定义在 `rva/roles.py`（`tests/test_roles.py` 钉住）：

| 阶段 | 角色 | 人设 | 契约要点 |
|------|------|------|----------|
| 一 | **Critic 批判性核验者** | 高可信度刊物的调查记者 | 拆解声明（decompose）→ 检索核验（verify）→ 给出判定（evaluate / output）；每条判定须附判据 |
| 二 | **Reviser 审慎修订者** | 同一刊物的专业编辑 | 依据核验报告做**最小修订**；只按判决 annotate / qualify / drop，**不引入新事实**，保持结构、风格与篇幅；每个动作记入 `revision_log` |

**共享伦理判准（两阶段一致，不可分工）**：

| id | 判准 | 落点 |
|----|------|------|
| `ability_evidenced` | 能力要可证 | 来源分级 + 可靠性加权 + NLI 三态判定（Stage 2） |
| `path_auditable` | 路径要可审 | criteria 判据 + 分层证据 + boundary_conditions（Stage 3） |
| `attitude_honest` | 态度要诚实 | H1–H5 披露检查（honesty）；H1–H4 即四查 `unsupported_rejected` / `uncertainty_marked` / `conflicts_preserved` / `sycophancy_checked` |
| `outcome_evidenced` | 结果要有据 | `[C#]` 标记 + `revision_log`；原回答不被就地修改 |

> Critic 只判断、Reviser 只改写。`CRITIC.prompt` 供 `critic_backend="llm"`（`LLMCritic`）注入，`REVISER.prompt` 供 LLM reviser（`reviser=`）注入；
> `roles.validate_revision()` 把 Reviser 的动作限定在既有 `REVISION_ACTIONS` 内，防止越权改写。

## 9. 可扩展点

| 位置 | 现状 | 可替换为 |
|------|------|----------|
| `decompose.classify` | 规则词典 | 已实现为 `critic_backend="llm"`（LLM 五级分类，仍保留 cues） |
| `DEFAULT_REGISTRY` | 内置来源分级 | 外部来源信誉 API |
| 证据匹配 | 向量检索 + NLI 三态判定（默认 hash/heuristic，离线） | 真实 embedding + LLM judge（`openai` / `llm`） |
| `evaluate_factuality` | 本地启发式 + 类型过滤（无 judge 钩子，判决不可外包） | 更广的外部证据检索（`rva/retrieval.py`）；不改判据 |
| `evaluate_grounding` | 关闭（`judge=local`） | Ragas / TruLens grounding judge（仅知识性陈述，单独报告） |
| `render` 输出 | 文本 + JSON | Web 分层证据界面（`challengeable` 已预留） |
| `honesty.py` | 文本级披露检查（H1–H5） | LLM 诚实性 judge（同样只读、独立报告） |
| `revise.py` | 确定性改写器 | LLM reviser（`reviser=` 注入） |

## 10. 与项目目标的对齐

| 目标 | 实现方式 |
|------|----------|
| 评估证据质量 | 阶段二来源分级 + 可靠性加权支持度 |
| 可追溯且可质疑的推理 | 每步输出 `cues / criteria / coverage`，全字段可复查 |
| 明示不确定性 | 类型标签 + 置信度 + `boundary_conditions` |
| 拒绝无依据的断言 | `X2` 剔除；Factuality 的 `unsupported` 计 0 |
| 认知准确性优先于满意度 | 对抗"流量驱动"（`D3`、`popularity_bias`），主观意见一律降级标注 |
| 评估证据质量 | 三重校验：证据效力（Stage 2）+ 可审计性（Stage 3）+ 诚实性（H1–H5） |
| 可追溯的修订 | Reviser 改写带 `[C#]` 标记与 `revision_log`，原回答不被篡改 |