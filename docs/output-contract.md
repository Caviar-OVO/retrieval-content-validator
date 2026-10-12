# 输出契约 / Output Contract

**检索内容校验 skill（retrieval-content-validator）** 的对外输出契约，供外部对接方引用。
所有字段与枚举均由 `tests/test_contract.py` 与 `tests/test_contract_doc.py` **交叉校验**，
不会与实际实现漂移。所有产物均为 Pydantic v2 模型，`model_dump_json()` 可直接导出。

- 代码内实现：`rva/`（RVA = Response Validation Agent，本地别名）
- 顶层入口：`rva.validate(question, answer, sources, ...) -> ValidationResult`
- 版本：`0.9.0`

> 每个可校验块前都有一个 ```` ```json ```` 索引围栏（`model:` / `fields:` / `vocabulary:` / `input:`）。
> 测试会据此定位对应实现并逐一比对；围栏与表格是机器可读的契约锚点。

---

## 0. 输入契约

```json
input: rva.validate
```

| 入参 | 必需 | 类型 | 说明 |
|------|------|------|------|
| `question` | 建议必需 | str | **架构级输入锚点**（非流程阶段）。critic 以它为基准抽取 claim；缺失则无法判断相关性 |
| `answer` | 是 | str | LLM 基于检索结果生成的回答；**永不被就地修改** |
| `sources` | 是 | `Source[]` | 检索候选文档 |
| `domain` | 否 | str | 领域，用于可信度加权；默认 `general` |
| `registry` | 否 | dict | 自定义来源分级表（替换内置 `DEFAULT_REGISTRY`） |
| `judge` | 否 | enum | `local` \| `ragas` \| `trulens`；仅控制 grounding 报告 |
| `retriever` | 否 | enum | Stage 2 检索后端：`hash`/`openai`/`auto` |
| `nli` | 否 | enum | Stage 2 蕴含判定后端：`heuristic`/`llm`/`auto` |
| `critic_backend` | 否 | enum | `deterministic`（默认）\| `llm` \| `auto`；仅当 `critic` 未给出时生效。`auto` 仅当**有 key 且有 sources** 时解析为 `llm`，否则安全回退 `deterministic`（严格 source-gating，记 WARNING）；显式 `llm`/`deterministic` 始终优先于 `auto` |
| `critic` | 否 | callable | 自定义 critic 后端（如 `LLMCritic`）；默认确定性规则 critic |
| `revise` | 否 | bool | 是否产出修订文本；默认 False |
| `reviser` | 否 | callable | 自定义修订后端（如 `LLMReviser`）；默认确定性本地改写器 |
| `reviser_backend` | 否 | enum | `local` \| `llm` \| `auto`；仅当 `reviser` 未给出时生效 |
| `recheck` | 否 | bool | 是否运行修订→复查闸门；默认读 `RVA_RECHECK`（1） |
| `recheck_threshold` | 否 | float | 触发再改的分数阈值；默认 0.7 |
| `recheck_max_rounds` | 否 | int | 最大复查轮数；默认 2 |
| `summary` | 否 | bool | 是否附报告级汇总 `summary`（只重述各路信号，**不是**单一总分）；默认 False |

```json
fields: rva.models.Source
```

`id` · `text` · `title` · `url` · `tag` · `domain` · `traffic` · `published`

---

## 1. 顶层结构 `ValidationResult`

```json
model: rva.models.ValidationResult
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `answer` | str | 原始回答（永不被改写） |
| `question` | str \| null | 输入锚点 |
| `domain` | str | 领域 |
| `critic_backend` | str | **报告级追溯**：实际产出 claim + 判决的 critic 后端（`deterministic` \| `llm`）；回退时记为 `deterministic`（降级不静默：`rva.critic` logger 记一条 WARNING） |
| `source_count` | int | 检索来源数 |
| `stage1_stats` | `DecompositionStats` | Stage 1 分类计数 |
| `claims` | `Claim[]` | Stage 1 产物（含 `question_anchor`） |
| `verifications` | `Verification[]` | Stage 2 产物（source 映射所在处） |
| `report` | `Report` | Stage 3 透明分层输出 |
| `factuality` | `FactualityResult` | **报告①** 五级判定（本地、来源锚定） |
| `grounding` | `GroundingResult` \| null | **报告②** 接地性复核（`judge=local` 时为 null） |
| `honesty` | `HonestyResult` | **报告③** 诚实性披露检查 |
| `revision` | `RevisionResult` \| null | **报告④** 修订输出（`revise=False` 时为 null） |
| `summary` | `ValidationSummary` \| null | **可选汇总**：五级分布 / 通过率 + grounding 0–100 + honesty 比例；仅重述、非总分（`summary=False` 时为 null） |

```json
fields: rva.models.DecompositionStats
```

`total` · `fact` · `reasoning` · `opinion` · `inference`

> 四份报告是**并列兄弟字段**，互不嵌套、互不覆盖。

---

## 2. 报告① 五级 verdict — `FactualityResult`

```json
model: rva.models.FactualityResult
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `score` | float | 来源锚定准确率 `(accurate + not_applicable) / total_claims` |
| `accurate` | int | 可靠来源证实 |
| `inaccurate` | int | 可靠来源反驳 |
| `disputed` | int | 可靠来源彼此冲突 |
| `unsupported` | int | 无可靠来源证实（**≠ 为假**） |
| `not_applicable` | int | 主观意见，免于外部核验 |
| `total_claims` | int | 全部 claim |
| `verifiable_claims` | int | 进入外部判定的知识性 claim |
| `applicable_claims` | int | `total_claims - not_applicable` |
| `verdicts` | `ClaimVerdict[]` | 逐 claim 判决 |
| `method` | str | `local_source_anchored`（deterministic critic，默认）\| `llm_critic`（LLM critic） |

> **判决不外包（默认轨）**：默认 deterministic critic 的判决由 `evaluate_factuality(claims, verifications)` 产出，该函数**没有** judge 参数，`method` 恒为 `local_source_anchored`。只有显式 `critic_backend="llm"`（或 `auto` 在「有 key + 有 sources」时选中 LLM 轨）时，五级判决才由 LLM critic 产出（`method="llm_critic"`）——需逐条带 `justification` + 来源 `evidence`，否则**整体回退**规则轨（`critic_backend` 如实记为 `deterministic`，并记一条 WARNING，降级不静默）。

**五级词汇表**（唯一权威定义：`rva.models.Verdict`）

```json
vocabulary: verdicts
```

| verdict | 含义 | 计分 |
|---------|------|------|
| `accurate` | 可靠来源证实 | 1 |
| `inaccurate` | 可靠来源反驳 | 0 |
| `disputed` | 可靠来源彼此冲突 | 0 |
| `unsupported` | 无可靠来源证实 | 0 |
| `not_applicable` | 主观意见，无需外部核验 | 1 |

```json
model: rva.models.ClaimVerdict
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `claim_id` | str | 关联 `claims[].id` |
| `claim_text` | str | 原文 |
| `claim_type` | enum | `fact` \| `reasoning` \| `opinion` \| `inference` |
| `verdict` | enum | 五级之一 |
| `score` | float | 该判决的规范值 0/1（非 judge 置信度） |
| `evidence_domains` | str[] | 支持来源域名 |
| `rationale` | str | 人类可读理由 |

---

## 3. 报告② 双栏：grounding / factuality 分开报告

| 报告 | 锚点 | 问的问题 | 取值 |
|------|------|----------|------|
| `factuality` | **外部可靠来源**（事实效度） | 陈述是否为真？ | 五级 verdict |
| `grounding` | **传入的检索上下文**（内部忠实性） | 陈述是否扎根于给定上下文？ | `grounded` / `ungrounded` / 无判决 |

```json
model: rva.models.GroundingResult
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `method` | str | `external_judge` |
| `judged_claims` | int | 得到判决的 claim 数 |
| `grounded_count` | int | grounded 数 |
| `ungrounded_count` | int | ungrounded 数 |
| `skipped_claims` | int | 非知识性 claim（不消耗 API 调用） |
| `total_claims` | int | 全部 |
| `mean_score` | float \| null | judge 原始分数均值 0..1 |
| `items` | `ClaimGrounding[]` | 逐 claim |

```json
model: rva.models.ClaimGrounding
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `claim_id` | str | 关联 claim |
| `claim_text` | str | 原文 |
| `claim_type` | enum | 四类 claim |
| `grounded` | bool \| null | 判定结果；null = 无判决 |
| `score` | float \| null | judge 原始分数 0..1 |
| `rationale` | str | 理由 |

判定阈值 `GROUNDED_THRESHOLD=0.75`；`None`（无上下文/NaN/后端失败）表示**不下判决**。

**judge 记录契约**（`{claim_id: record}`）：`record` 可为位置序列 `(grounded, score, rationale)`、
带同名字段的映射（dict），或暴露同名属性的对象——三者等价。**其他形状显式 `raise ValueError`**，
不再静默退化为"无判决"（把格式错误伪装成 `None` 等于让坏适配器蒙混过关）；batch judge 的
整体返回也必须是 `{claim_id: record}` 映射。

> **关键不变量**：`grounding` 从不写入 `factuality`；judge 分数**绝不**覆盖五级判决。
> grounding 使用 bool 词汇，五级 verdict 只属于 factuality，两套词汇**刻意不混**。

---

## 4. 分层审计日志 — `Report`

```json
model: rva.models.Report
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `summary` | `ReportSummary` | 决策计数与平均可信度 |
| `items` | `ReportItem[]` | 逐 claim |
| `criteria` | dict | 全部判据定义（见下表） |
| `supplemented_text` | str \| null | 开 `revise` 时=修订文本，否则 null |

```json
fields: rva.models.ReportSummary
```

`total` · `kept` · `demoted` · `dropped` · `contextualized` · `avg_credibility`

```json
model: rva.models.ReportItem
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `claim_id` | str | 关联 claim |
| `text` | str | 原文 |
| `type` | enum | 四类 claim |
| `credibility` | float | 0..1，仅基于**检索到的证据集** |
| `decision` | enum | `keep` / `demote` / `drop` / `contextualize` |
| `decision_label` | str | 决策的可读标签 |
| `criteria` | `Criterion[]` | 触发决策的规则 |
| `boundary_conditions` | str[] | 结论成立/失效的边界 |
| `layers` | `Layers` | 三层证据视图 |
| `popularity_bias` | `PopularityBias` | 传播度偏置 |
| `decomposition` | dict | 分类凭证 + 问题相关性 |
| `challengeable` | bool | 恒 True：**可质疑** |

```json
fields: rva.models.Criterion
```

`id` · `rule`

```json
fields: rva.models.Layers
```

`l1_primary` · `l2_secondary` · `l3_conflicting`

```json
fields: rva.models.PopularityBias
```

`biased` · `ratio` · `note` · `avg_traffic`

`Layers` 三档：`l1_primary`（高可靠一手证据）/ `l2_secondary`（弱/部分证据）/ `l3_conflicting`（冲突来源）——
每项均为 `EvidenceRef`。

### 证据效力检查（per-claim，`rva/efficacy.py`）

对每条 claim 执行四项检查，输出 `source_quality` + `freshness` + `support_score`（外加 `consistency`
与 `popularity_suspect_source_ids`）。全部离线、确定性，且两轨 critic 共用（都经 `verify()`）。

1. **来源权威性评级** — 把绑定候选来源评为 `official`（官方/权威机构）· `professional`（专业媒体/同行评审）·
   `self_media`（自媒体/低可信）· `unverifiable`（不可考）；claim 的总体等级取其**实际参与论证**来源中的最高级。
2. **时效性检查** — 动态变化的事实（政策/价格/事件进展等，或含年份）记录 `checked_at`；无可考日期或过旧则标 `stale`。
3. **一致性比对** — 核对 claim 的数字与限定词：数字在其支持来源中缺失记 `number_mismatch`（篡改/替换嫌疑）；
   限定词强度高于来源记 `qualifier_overreach`（夸大/断章取义嫌疑）。
4. **流量 vs 效度** — 支持集中高流量且低可靠来源时，`popularity_suspect_source_ids` 显式标记「流量偏置嫌疑」，
   并与 `popularity_bias`（`D3`/`H2`）联动；不得因热度默认采信。

**`support_score`** ＝ 支持来源可靠性的均值（基准），再因一致性违规（×0.7）与流量驱动（×0.85）折减：
无违规时与旧值**逐字一致**。

```json
fields: rva.models.SourceQuality
```

`grade` · `counts` · `sources` · `note`

```json
fields: rva.models.SourceGrade
```

`source_id` · `grade` · `reliability` · `tier` · `note`

```json
fields: rva.models.Freshness
```

`checked_at` · `dynamic` · `has_date` · `newest_date` · `stale` · `note`

```json
fields: rva.models.Consistency
```

`consistent` · `number_mismatch` · `qualifier_overreach` · `claim_numbers` · `source_numbers` · `note`

**来源权威性等级词汇表**（唯一权威定义：`rva.models.AuthorityGrade`）

```json
vocabulary: authority_grades
```

| grade | 含义 |
|-------|------|
| `official` | 官方 / 权威机构 |
| `professional` | 专业媒体 / 同行评审 |
| `self_media` | 自媒体 / 低可信 |
| `unverifiable` | 不可考 |

**逐 claim 双源溯源**

```json
fields: rva.models.Claim
```

`id` · `text` · `type` · `confidence` · `cues` · `score_breakdown` · `note` · `verifiable` · `question_anchor`

```json
fields: rva.models.QuestionAnchor
```

`question` · `relevance` · `reactant` · `note`

```json
fields: rva.models.Verification
```

`claim_id` · `claim_text` · `claim_type` · `evidence` · `support_score` · `independent_domains` · `conflict` · `source_quality` · `freshness` · `consistency` · `popularity_suspect_source_ids` · `popularity_bias` · `retrieval_backend` · `nli_backend` · `screened_count` · `question_relevance` · `off_topic`

**source 映射在 `verification.evidence`（`Evidence`）**

```json
fields: rva.models.Evidence
```

`supports` · `partials` · `contradicts` · `screened` · `claim_negation`

| 字段 | 说明 |
|------|------|
| `supports` | 三态判定为支持 |
| `partials` | 中立 / 部分覆盖 |
| `contradicts` | 反驳 |
| `screened` | 被相似度阈值筛掉的候选（**保留**，可复查） |
| `claim_negation` | 命中的否定式（若有） |

```json
fields: rva.models.EvidenceRef
```

`source_id` · `title` · `url` · `domain` · `reliability` · `tier` · `similarity` · `coverage` · `shared_tokens` · `traffic` · `negation` · `stance`

> `similarity` / `coverage` / `reliability` 取值恒为 `[0, 1]`：原始余弦相似度在源头钳到
> `[0, 1]`，`EvidenceRef` 模型层再做一次兜底钳位（反相关的词分布可能算出负值）。
> 越界的浮点被钳位而非抛错，保证报告始终可产出；非数值类型仍会被拒绝。

**判据（criteria）**

| 类别 | 规则 |
|------|------|
| keep | `K1` 有可靠性≥0.7来源支持 · `K2` 非流量驱动 · `K3` 无同等/更高可靠性来源反驳 · `K4` 推理前提得到可靠支持 |
| demote | `D1` 主观意见或无依据推论 · `D2` 仅低可靠来源弱支持 · `D3` 依赖流行度 · `D4` 推理前提缺乏支持 · `D5` 未回应用户问题 |
| drop | `X1` 被更高可靠性来源反驳 · `X2` 可证伪的事实陈述毫无证据 |
| contextualize | `N1` 主观意见不作为既定事实 · `N2` 明确标注其主观属性 |

---

## 5. 报告③ 诚实性凭证 — `HonestyResult`

```json
model: rva.models.HonestyResult
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `score` | float | 适用披露检查的通过比例 0..1 |
| `passed` | bool | `score >= 0.7` 且无 `H1` 硬失败 |
| `flags` | `HonestyFlag[]` | 逐条未通过的披露检查 |
| `checked_claims` | int | 受检 claim 数（意见豁免） |
| `exempt_claims` | int | 豁免（主观意见）数 |
| `method` | str | `local_heuristic` |
| `conflicts_preserved` | bool | Step-4 四查之一：来源冲突被保留而非屏蔽（`H3`，True=诚实） |
| `uncertainty_marked` | bool | Step-4 四查之一：不确定 / 检索范围被显式标注（`H4`） |
| `unsupported_rejected` | bool | Step-4 四查之一：无依据断言被拒斥或加限定（`H1`，硬失败项） |
| `sycophancy_checked` | bool | Step-4 四查之一：流量驱动支持被核验而非迎合（`H2`） |

```json
model: rva.models.HonestyFlag
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `code` | str | `H1`–`H5` |
| `label` | str | 短标签 |
| `detail` | str | 详细说明 |
| `claim_ids` | str[] | 相关 claim |

**披露检查码表**（唯一权威定义：`rva.honesty.HONESTY_CODES`）

```json
vocabulary: honesty_codes
```

| code | 四查字段 | 检查 | 触发条件 |
|------|----------|------|----------|
| `H1` | `unsupported_rejected` | 无依据断言未加限定 | 可验证 claim 无可靠支持且正文无 hedge 词（**硬失败**） |
| `H2` | `sycophancy_checked` | 传播度驱动未坦白 | `popularity_bias.biased` 但回答未提及 |
| `H3` | `conflicts_preserved` | 来源冲突未承认 | `conflict` 为真但回答未提及 |
| `H4` | `uncertainty_marked` | 检索范围未说明 | 回答无"检索/现有资料/来源"等范围声明 |
| `H5` | —（相关性检查，单独报告） | claim 未回应问题 | 有 question 且该知识性 claim 为 off-topic |

`H1`–`H4` 即 Step-4 四查，逐个投影到 `HonestyResult` 上的同名字段（True=诚实）；
`H5` 是问题相关性检查而非披露检查，只经 `flags` 报告。映射表：
`rva.honesty.HONESTY_FOUR_CHECKS`。

---

## 5.1 可选汇总 — `ValidationSummary`

```json
model: rva.models.ValidationSummary
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `factuality_distribution` | dict | 五级判定计数（`accurate`/`inaccurate`/`disputed`/`unsupported`/`not_applicable`） |
| `factuality_pass_rate` | float | 通过率 = `factuality.score`；分布比例，**不是**判定 |
| `grounding_score` | float \| null | grounding judge 的 0–100 分（沿用其自有量纲）；未跑 judge 时为 null |
| `honesty_score` | float | 诚实性披露通过比例 0..1 |
| `note` | str | 声明"这是报告级汇总，不是分数；五级 factuality 才是判定" |

`summary` 刻意**不提供单一融合总分**：那正是五级词汇表要防的错误（把连续分数当判定）。
它只把已分开的三路信号并排重述，供读者一眼看齐，不改变任何一路的结论。

---

## 6. 报告④ 修订输出 — `RevisionResult`

```json
model: rva.models.RevisionResult
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `revised_text` | str | 修订后文本；**改动过的 claim 带 `[C#]` 标记**，未改动的原样通过 |
| `supplemented` | bool | 是否由外部 reviser 后端生成 |
| `changed` | int | 被改动的 claim 数 |
| `revision_log` | `RevisionItem[]` | 逐 claim 修订记录 |
| `method` | str | `local_rewriter` \| `llm_reviser` |
| `diff` | str \| null | 修订前后统一 diff（两种后端都落此凭证） |
| `input_verdicts` | str[] | Reviser 据以改写的输入 verdict 清单 |
| `end_marker` | str | LLM 轨必须输出的编辑终止标记 `---END-OF-EDIT---` |
| `end_marker_present` | bool | LLM 响应是否实际带该标记（本地轨为 False） |
| `fallback` | bool | LLM 后端失败并回退到确定性改写器 |
| `recheck` | `RecheckResult` \| null | 修订→复查闸门报告（未跑闸门时为 null） |

```json
model: rva.models.RevisionItem
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `claim_id` | str | 关联 claim |
| `action` | str | 修订动作（见码表） |
| `original_text` | str | 原文 |
| `revised_text` | str \| null | 改写后；drop 为 null |
| `reason` | str | 理由 |
| `criteria` | str[] | 触发判据 |

**修订动作码表**（唯一权威定义：`rva.revise.REVISION_ACTIONS`）

```json
vocabulary: revision_actions
```

| action | 含义 |
|--------|------|
| `keep` | 未改动，原样保留 |
| `annotate` | 主观意见：保留原句 + `[C#]` + 主观标记 |
| `qualify` | 降级 / off-topic：保留原句 + `[C#]` + 警示 |
| `drop` | 移入 `[Validator supplement]` 块，保留 `[C#]` 与理由 |

> **关键不变量**：原回答永不被就地修改；`revised_text` 只是新增产物。

### 修订→复查闸门 — `RecheckResult` / `RecheckRound`

LLM 修订是**新生成的文本**，可能引入新幻觉（引用忠实 ≠ 正确）。因此每次修订后都对修订文本
**重跑来源锚定的五级事实性判定**；若出现「新增未获证的 claim」（inaccurate / disputed / unsupported）且分数低于阈值，则再改一轮（最多 `max_rounds`），
并保留分数最高的候选，闸门**不会让产出比首轮更差**。

```json
model: rva.models.RecheckResult
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `ran` | bool | 闸门是否运行 |
| `passed` | bool | 末轮是否通过（无新增未获证 claim 或分数达阈） |
| `threshold` | float | 触发再改的分数阈值（默认 0.7） |
| `initial_score` | float | 首轮修订文本的事实性分数 |
| `final_score` | float | 最终保留候选的分数 |
| `max_rounds` | int | 允许的最大轮数（默认 2） |
| `rounds` | `RecheckRound[]` | 逐轮复查记录 |
| `leftover_inaccurate` | str[] | 仍残留的新增未获证 claim（inaccurate / disputed / unsupported） |
| `method` | str | 恒为 `factuality_recheck` |

```json
model: rva.models.RecheckRound
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `round` | int | 轮次（从 1 起） |
| `revised_text` | str | 该轮候选文本 |
| `factuality_score` | float | 该轮事实性分数 |
| `inaccurate` | int | 该轮 Inaccurate claim 数 |
| `new_inaccurate` | str[] | 输入 verdict 未标为未获证、本轮新出现的 claim（inaccurate / disputed / unsupported） |
| `changed` | int | 该轮改动的 claim 数 |
| `method` | str | 该轮后端（`local_rewriter` \| `llm_reviser`） |
| `passed` | bool | 该轮是否通过 |

> **后端双轨**：`local`（默认，离线可审计、零成本）与 `llm`（编辑人设，可回退）。两轨都记录
> `diff` / `input_verdicts` / `end_marker`，凭证闭环不因换后端而断。

---

## 7. 格式选项（CLI）

| 选项 | 输出 |
|------|------|
| （默认） | `report` → `factuality` → `grounding` → `honesty` → `revision` |
| `--factuality` / `--faithfulness` | 报告① |
| `--grounding` | 报告②（需 `--judge ragas/trulens`） |
| `--honesty` | 报告③（诚实性） |
| `--revise` / `--revised-text-only` | 报告④（后者只打印修订文本） |
| `--reviser local\|llm\|auto` | 修订后端：local 确定性（默认）/ llm 编辑人设 / auto |
| `--no-recheck` | 关闭修订→复查闸门 |
| `--recheck-threshold` / `--recheck-rounds` | 复查阈值 / 最大轮数 |
| `--json` | 完整 `ValidationResult` |
| `--summary` | 附报告级汇总（五级分布 / 通过率 + grounding 0–100 + honesty 比例；非总分） |
| `--retriever` / `--nli` | Stage 2 后端：`hash`/`openai`/`auto`、`heuristic`/`llm`/`auto` |

---

## 8. 交叉校验与同步来源

本文档的可校验不变量由 `tests/test_contract_doc.py` 强制，与代码逐项比对：

| 校验项 | 对应实现 |
|--------|----------|
| 字段集合（`model:` / `fields:` 围栏） | `rva/models.py` |
| 五级 verdict 词汇表 | `rva.models.Verdict` |
| honesty 码 `H1`–`H5` | `rva.honesty.HONESTY_CODES` |
| 修订动作码表 | `rva.revise.REVISION_ACTIONS` |
| `validate()` 入参 | `rva.validate` 签名 |

若代码演进导致字段/枚举变化而本文档未同步，上述测试会失败。
行为层面的不变量另见 `tests/test_contract.py`。
