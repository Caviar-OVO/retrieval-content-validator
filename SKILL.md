---
name: retrieval-content-validator
description: 对模型基于网络检索生成的回答执行证据效力、可审计性、诚实性三重校验，输出 claim 级核查报告、总体置信分与修订建议。适用于事实性问答、检索增强生成（RAG）输出的核查场景。
metadata:
  display_name: 检索内容校验 skill
  short-description: 校验 RAG 回答的证据效力、可审计性与诚实性
---

# 检索内容校验 skill (retrieval-content-validator)

校验 LLM 基于网络检索结果生成的回答。核心命题：搜索引擎按语义相似度与流量排序、
不验证真实性，LLM 又按"相关性 / 受欢迎程度"继续筛选，从而放大认知风险；因此本 skill
要求模型**为筛选决策负责**——评估证据质量、给出可追溯可质疑的推理、明示不确定、
拒绝无依据断言，并把认知准确性置于用户满意度之上。

`rva/` 是代码内实现（缩写 RVA = Response Validation Agent），是**该 skill 的本地别名**，
不是对外产品名。

## 角色设定（双阶段接力，对应 LLM Auditor 的 Critic / Reviser）

同一个 auditor 兼任两顶帽子，**先后接力**，两阶段**共享同一组伦理判准**。角色定义是唯一权威来源：
`rva/roles.py`（由 `tests/test_roles.py` 钉住，不容漂移）；`python -m rva.cli --roles` 可打印。

| 阶段 | 角色 | 人设 | 使命 |
|------|------|------|------|
| 一 | **Critic 批判性核验者** | 供职于高可信度刊物的调查记者，以批判性思维核查一切待刊发内容 | 拆解声明、检索核验、给出判定 |
| 二 | **Reviser 审慎修订者** | 同一刊物的专业编辑 | 依据核验报告对原文做最小修订，**不引入新事实，保持结构、风格与篇幅** |

**共享伦理判准（两阶段一致）**：`能力要可证`（证据效力，Stage 2）· `路径要可审`（判据 / 分层证据，
Stage 3）· `态度要诚实`（H1–H5 披露检查）· `结果要有据`（`[C#]` 标记 + `revision_log`，原回答不被就地修改）。

> Critic 只**判断**，Reviser 只**改写**（且是最小、加法的改写）；修订是可选步骤（`--revise`）。

Critic **双轨后端**：`deterministic`（默认，规则 + 离线，判决由 `evaluate_factuality` 产出）与 `llm`（注入 `CRITIC.prompt` + few-shot，claim 抽取 + 五级判定一次完成；对齐 LLM Auditor critic；失败回退规则轨）。两轨输出 schema 完全一致（claim / cues / verdict / justification / evidence）；LLM 轨的获证判决必须引用来源 id。

**`critic_backend` 选择（三段）**：`auto` 仅当**有 key 且有 sources** 时解析为 `llm`，否则安全回退 `deterministic`（严格 source-gating：`build_critic` 的 `has_sources` 默认 `False`）；显式 `llm`/`deterministic` 始终优先于 `auto`——显式 `llm` 即使无来源也放行（调用方明示），但记一条 WARNING；任何回退都**不静默**：记 WARNING，且 `critic_backend` 如实记为 `deterministic`。

Reviser **双轨后端**：`local`（默认，确定性、离线、零成本——审计底线）与 `llm`（编辑人设，`REVISER.prompt`，
输出须以 `---END-OF-EDIT---` 结束，失败自动回退本地）。两轨都落 `diff` / `input_verdicts` / `end_marker`。
每次修订后**必过修订→复查闸门**：对修订文本重跑来源锚定的五级事实性判定，出现「新增未获证 claim」（inaccurate / disputed /
unsupported）且低于阈值则再改一轮（最多 2 轮），保留最高分候选——因 LLM 修订是新生成文本，可能引入新幻觉。

## 输入 / 输出契约

**输入**（缺 `question` 会显著削弱校验，见下）
- `question` 用户问题
- `answer` 模型回答文本
- `sources` 检索候选文档集（含 `id`、`text`，可选 `url`、`traffic`）

**输出**
- **claim 级报告**：`report.items[]` = 每条 claim 的 `verdict` + `decision` + `criteria` + 分层证据；
  `factuality.verdicts[]` = 五级判决 + `rationale`；source 映射在 `verifications[].evidence`
  （`supports` / `contradicts` / `screened`，均带 `source_id`）
- **总体评估**：五级判定（`factuality.verdicts`）才是判定；`factuality.score`、`honesty.score`、`report.summary` 只是比例/计数。可选 `summary` 只重述各路信号（五级分布 + 通过率、grounding 0–100、honesty 比例），**不提供单一总分**
- **分层审计日志**：`criteria` / `layers(l1,l2,l3)` / `cues` / `score_breakdown` /
  `revision.revision_log`
- **修订后文本（可选）**：`revision.revised_text`；改动过的 claim 带 `[C#]` 标记，原回答**不被改写**

## 证据效力检查（per-claim，`rva/efficacy.py`）

对每条 claim 执行四项检查，输出 `source_quality` + `freshness` + `support_score`（外加 `consistency`
与 `popularity_suspect_source_ids`）；两轨 critic 共用（都经 `verify()`），离线确定性。

1. **来源权威性评级**：绑定候选来源评为 `official`（官方/权威机构）/ `professional`（专业媒体/同行评审）/
   `self_media`（自媒体/低可信）/ `unverifiable`（不可考）；claim 总体等级取**实际参与论证**来源中的最高级。
2. **时效性检查**：动态变化事实（政策/价格/事件进展，或含年份）记录 `checked_at`；无可考日期或过旧标 `stale`。
3. **一致性比对**：claim 的数字在支持来源中缺失 → `number_mismatch`（篡改嫌疑）；限定词强于来源 → `qualifier_overreach`（夸大/断章取义嫌疑）。
4. **流量 vs 效度**：高流量且低可靠来源进入支持集 → `popularity_suspect_source_ids` 标记「流量偏置嫌疑」，并与 `D3`/`H2` 联动。

`support_score` = 支持来源可靠性均值，因一致性违规（×0.7）与流量驱动（×0.85）折减；无违规时与旧值一致。

## 三重校验

1. **证据效力** — Stage 2：来源分级（`RELIABILITY_THRESHOLD=0.7`）+ 可靠性加权 + 传播度偏置与矛盾检测，
   再落到五级判决（`accurate` / `inaccurate` / `disputed` / `unsupported` / `not_applicable`），
   **锚点是外部可靠来源**（事实效度），不是模型自评。
2. **可审计性** — Stage 3：每条 claim 返回触发决策的 `criteria`（`K*`/`D*`/`X*`/`N*`/`D5`）、
   分层证据与边界条件，`challengeable=True`。
3. **诚实性** — `honesty.py` 的 `H1`–`H5` 披露检查：无依据断言是否被平铺直叙、
   传播度驱动是否坦白、来源冲突是否承认、检索范围是否说明、claim 是否回答问题。

**question 是必需的锚点**。LLM Auditor 读的是 (question, answer) 对；没有 question，抽出的 claim
就没有锚点，无法判断相关性。本项目在 `decompose(answer, question=...)` 记录 `question_anchor`
（相关性 + `reactant`），共享词为零的知识性 claim 记为 off-topic（`D5`，降级）。

## 输出契约

完整的字段级契约见 `docs/output-contract.md`（与代码交叉校验，不漂移）。

## 关键约束（不可混淆）

- **五级判决不外包给 grounding judge**：`evaluate_factuality(claims, verifications)` **没有** judge 参数；默认 deterministic critic 的 `method` 恒为 `local_source_anchored`。
- **Critic 双轨**：`critic_backend="deterministic"`（默认，离线规则）｜`"llm"`（`CRITIC.prompt`，claim 抽取 + 五级判定一次完成，`method="llm_critic"`）｜`"auto"`（有 key 且有 sources 才选 `llm`，否则回退 `deterministic`）；两轨 schema 完全一致，LLM 轨判决须带 `justification` + 来源 `evidence`，否则回退规则轨；显式 `llm`/`deterministic` 优先于 `auto`，回退不静默（记 WARNING 且 `critic_backend` 如实记为 `deterministic`）；critic 不做二次校验（只有 Reviser 才过复查闸门）。
- **judge 只做 grounding**：Ragas / TruLens 锚定的是**传入的检索上下文**（内部忠实性），
  结果放在 `result.grounding`，与五级判决**分开报告**；judge 分数绝不可覆盖判决。
- **opinion 类型豁免外部核验**：直接判 `Not Applicable`，不触发 judge 调用。
- **修订是加法**：原回答永不被就地修改；`revise=True` 才产出 `revision`。

## 运行

```bash
python -m rva.cli --answer-file examples/answer.txt --question-file examples/question.txt \
  --sources examples/sources.json                      # 报告 + factuality + honesty
python -m rva.cli --answer-file examples/answer.txt --question "..." --sources examples/sources.json --revise
python -m rva.cli --answer-file examples/answer.txt --question "..." --sources examples/sources.json --judge ragas --grounding
python examples/compare_judges.py local ragas
python -m unittest discover -s tests
```

库调用：

```python
from rva import validate, format_report, format_factuality, format_honesty

result = validate(question=q, answer=a, sources=sources, revise=True)
result.factuality      # 五级判决（本地、来源锚定）
result.honesty         # 披露检查 H1-H5
result.grounding       # 可选 judge（judge="local" 时为 None）
result.revision        # 可选修订（revise=True）
```

## 已知限制

- 规则/启发式对隐含语义、反语不敏感，属**辅助审查**，非最终裁决。
- `unsupported` 只表示**当前检索集**未证实，不代表为假。
- 来源分级内嵌价值判断（如 `.gov` 视为高可靠），已显式化、可通过 `registry` 替换。
- 诚实性检查是文本级披露检测，不是"说谎检测"。
