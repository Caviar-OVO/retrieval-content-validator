r"""Streamlit 交互式校验台 for the Response Validation Agent (RVA).

Run:
    .venv/Scripts/streamlit run app/streamlit_app.py
    # or
    python -m streamlit run app/streamlit_app.py

This is a thin UI over `rva.validate(...)`: it renders the four reports the
pipeline already produces (report / factuality / grounding / honesty + the
optional revision), keeping every field traceable back to the models in
`rva/models.py`. No new validation logic lives here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# --- make the package importable when run via `streamlit run` ---------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rva import format_factuality, format_grounding, format_honesty, format_report, validate  # noqa: E402
from rva.config import describe  # noqa: E402
from rva.models import Source, ValidationResult  # noqa: E402
from rva.stages.output import RELIABILITY_THRESHOLD  # noqa: E402
from rva.stages.verify import DEFAULT_REGISTRY  # noqa: E402
import rva.stages.verify as _verify_module  # noqa: E402

#: Bump on every user-visible change; shown in the sidebar.
APP_VERSION = "1.4"

#: A short content fingerprint of the loaded `rva.stages.verify` file. Compare it
#: against the value on GitHub (or a fresh local run): if the running server shows
#: a different fingerprint, it is serving a *stale* copy of the package it
#: imported at start-up. Streamlit caches imported modules, so a browser refresh
#: never re-imports them - the server process (or the cloud deployment) must be
#: restarted/rebuilt for a code change to take effect.
def _source_fingerprint() -> str:
    import hashlib
    try:
        blob = Path(_verify_module.__file__).read_bytes()
        return hashlib.sha256(blob).hexdigest()[:8]
    except Exception:
        return "unknown"


BUILD_STAMP = (
    f"v{APP_VERSION} · registry={len(DEFAULT_REGISTRY)} · "
    f"threshold={RELIABILITY_THRESHOLD} · fp={_source_fingerprint()}"
)

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from import_sources import parse_sources  # noqa: E402

EXAMPLES = ROOT / "examples"

VERDICT_ORDER = ["accurate", "inaccurate", "disputed", "unsupported", "not_applicable"]
VERDICT_META = {
    "accurate":       ("可靠来源证实", "green"),
    "inaccurate":     ("可靠来源反驳", "red"),
    "disputed":       ("来源彼此冲突", "orange"),
    "unsupported":    ("无可靠来源证实（≠ 为假）", "gray"),
    "not_applicable": ("主观意见，免于外部核验", "blue"),
}
DECISION_META = {
    "keep":          ("保留", "green"),
    "demote":        ("降级", "orange"),
    "drop":          ("剔除", "red"),
    "contextualize": ("语境化", "blue"),
}

st.set_page_config(page_title="RVA 校验台", page_icon="🔎", layout="wide")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def load_example_sources() -> list[dict]:
    path = EXAMPLES / "sources.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return [{"id": "S1", "title": "", "url": "", "tag": "", "traffic": 0.5, "text": ""}]


def sources_to_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ("id", "title", "url", "tag", "text"):
        if col not in df:
            df[col] = ""
    if "traffic" not in df:
        df["traffic"] = 0.5
    return df[["id", "title", "url", "tag", "traffic", "text"]].fillna("")


def df_to_sources(df: pd.DataFrame) -> list[Source]:
    out: list[Source] = []
    for _, row in df.iterrows():
        sid = str(row.get("id", "")).strip()
        text = str(row.get("text", "")).strip()
        if not sid and not text:
            continue  # skip blank rows
        item = {"id": sid or f"S{len(out) + 1}", "text": text}
        for key in ("title", "url", "tag"):
            val = str(row.get(key, "")).strip()
            if val:
                item[key] = val
        try:
            item["traffic"] = float(row.get("traffic"))
        except (TypeError, ValueError):
            pass
        out.append(Source(**item))
    return out


def merge_sources(existing: list[dict], incoming: list[dict], mode: str) -> list[dict]:
    """Combine imported records with the current table ('replace' or 'append')."""
    if mode == "替换现有来源":
        return incoming
    used = {str(r.get("id", "")).strip() for r in existing}
    seen: set[str] = set()
    for r in existing:
        seen.add(str(r.get("url", "")).strip().lower() or str(r.get("text", ""))[:120])
    out = list(existing)
    for rec in incoming:
        url = str(rec.get("url", "")).strip().lower()
        dedup = url or str(rec.get("text", ""))[:120]
        if dedup and dedup in seen:
            continue
        base = str(rec.get("id", "")).strip() or f"S{len(out) + 1}"
        sid, i = base, 1
        while sid in used:
            i += 1
            sid = f"{base}_{i}"
        used.add(sid)
        if dedup:
            seen.add(dedup)
        item = dict(rec)
        item["id"] = sid
        out.append(item)
    return out


def render_source_import() -> None:
    """Paste or upload retrieved sources and merge them into the table."""
    with st.expander("📥 导入检索到的来源（粘贴或上传文件）"):
        st.caption(
            "把检索工具的输出、或手抄的搜索结果直接贴进来。自动识别 JSON / JSONL / CSV / TSV、"
            "每行一个 URL、Markdown 链接，以及「标题 + 链接 + 摘要」的文本块；"
            "字段名 `link`/`href`、`snippet`/`content`、`score` 等常见别名会归一化，"
            "缺失的 `id` / `tag` 会自动补全。"
        )
        mode = st.radio("导入方式", ["粘贴文本", "上传文件"], horizontal=True, key="import_mode")
        raw, filename = "", ""
        if mode == "粘贴文本":
            raw = st.text_area(
                "粘贴来源", key="import_paste", height=160,
                placeholder='例：[{"url":"https://www.who.int/...","snippet":"室外空气污染每年约420万人过早死亡。"}]\n'
                            "或每行一个链接，或「标题 + 链接 + 摘要」的文本块。",
            )
        else:
            uploaded = st.file_uploader(
                "选择文件", type=["json", "jsonl", "ndjson", "csv", "tsv", "txt", "md"],
                key="import_file",
            )
            if uploaded is not None:
                filename = uploaded.name
                raw = uploaded.getvalue().decode("utf-8", errors="replace")

        c1, c2 = st.columns([3, 2])
        target = c1.radio("写入方式", ["替换现有来源", "追加到现有来源"], key="import_target")
        submitted = c2.button("解析并导入", type="primary", width="stretch", key="import_submit")

        if submitted:
            if not raw.strip():
                st.session_state["import_msg"] = ("error", "没有可导入的内容：请粘贴文本或选择一个文件。")
            else:
                parsed = parse_sources(raw, filename)
                if not parsed.records:
                    detail = "　".join(parsed.warnings) if parsed.warnings else ""
                    st.session_state["import_msg"] = ("error", f"未能识别任何来源。{detail}")
                else:
                    st.session_state["sources_rows"] = merge_sources(
                        st.session_state["sources_rows"], parsed.records, target
                    )
                    st.session_state.pop("sources_editor", None)
                    note = f"已导入 {len(parsed.records)} 条（识别格式：{parsed.format}）。"
                    if parsed.warnings:
                        note += " " + " ".join(parsed.warnings[:3])
                    st.session_state["import_msg"] = ("success", note)
                    st.rerun()

        msg = st.session_state.pop("import_msg", None)
        if msg:
            kind, text = msg
            (st.success if kind == "success" else st.error)(text)


def badge(text: str, color: str) -> str:
    return f":{color}-background[{text}]"


ONBOARDING_STEPS = [
    ("① 填入问题", "问题（question）是 claim 锚点：critic 以它为基准拆解并判断相关性。"),
    ("② 填入待审回答", "把 LLM 基于检索结果生成的回答（answer）粘进来。"),
    ("③ 提供检索来源", "sources 是核验证据。可点「载入示例」、用「📥 导入检索到的来源」粘贴/上传，或直接编辑表格。"),
    ("④ 运行校验", "点「▶️ 运行校验」，逐 tab 查看五级判定、报告、诚实性与修订。"),
]


def load_example_into_form() -> None:
    """Fill the answer + sources from examples/ (used by guide and button)."""
    answer_file = EXAMPLES / "answer.txt"
    if answer_file.exists():
        st.session_state["answer_text"] = answer_file.read_text(encoding="utf-8").strip()
    st.session_state["sources_rows"] = load_example_sources()
    st.session_state.pop("sources_editor", None)  # force the table to re-render


def render_onboarding() -> None:
    """A first-run, dismissible guide with a one-click 'try the example' button."""
    with st.container(border=True):
        st.markdown("### 👋 新手引导 · 4 步跑通回答校验")
        st.caption(
            "这个校验台审查 LLM 基于网络检索生成的回答：拆解 claim → 检索核验 → "
            "五级事实性判定 → 诚实性检查 →（可选）修订。所有后端默认离线，点一下即可运行。"
        )
        cols = st.columns(len(ONBOARDING_STEPS))
        for col, (title, desc) in zip(cols, ONBOARDING_STEPS):
            with col:
                st.markdown(f"**{title}**")
                st.caption(desc)

        st.markdown("---")
        c1, c2, _ = st.columns([1.2, 1.2, 2])
        if c1.button("🚀 一键体验示例", type="primary", width="stretch"):
            load_example_into_form()
            st.session_state["show_onboarding"] = False
            st.rerun()
        if c2.button("跳过，直接开始", width="stretch"):
            st.session_state["show_onboarding"] = False
            st.rerun()

        with st.expander("概念速览：critic 双轨 / 五级判定 / 四个报告"):
            st.markdown(
                "- **critic 双轨**：`deterministic`（默认，离线规则）｜`llm`（人设 critic，失败回退）；"
                "`auto` 需「有 key 且有 sources」才走 LLM 轨。\n"
                "- **五级判定**（报告①）：`accurate` / `inaccurate` / `disputed` / `unsupported` / "
                "`not_applicable`——锚点是**外部可靠来源**，`unsupported` ≠ 为假。\n"
                "- **四个报告并列**：① 事实性 ② claim 报告（判据/分层证据/边界条件）③ 诚实性 H1–H5 "
                "④ 接地性 grounding（judge）；外加可选的修订 + 复查闸门。\n"
                "- **修订是加法**：原回答永不被就地修改，改动过的 claim 带 `[C#]` 标记可回溯。"
            )


def verdict_badge(v: str) -> str:
    label, color = VERDICT_META.get(v, (v, "gray"))
    return f":{color}[**{v}**] — {label}"


def decision_badge(d: str) -> str:
    label, color = DECISION_META.get(d, (d, "gray"))
    return f":{color}[{label} · `{d}`]"


# --------------------------------------------------------------------------
# sidebar: configuration
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ 配置")
st.sidebar.caption("默认全部离线可跑；带 key 才启用 LLM / judge 轨。")

# First visit: show the guide automatically.
if "show_onboarding" not in st.session_state:
    st.session_state["show_onboarding"] = True

with st.sidebar:
    if st.button("❓ 新手引导", width="stretch"):
        st.session_state["show_onboarding"] = True
        st.rerun()
    cfg_info = describe()
    st.caption(f"API key：{cfg_info['openai_api_key']}   模型：{cfg_info['judge_model']}")
    st.caption(f"🧩 版本：{BUILD_STAMP}")

    st.markdown("**Critic（阶段一）**")
    critic_backend = st.selectbox(
        "critic_backend",
        ["deterministic", "auto", "llm"],
        index=0,
        help="deterministic=离线规则（默认）｜auto=有 key 且有 sources 才走 llm｜llm=强制人设 critic（失败回退）",
    )

    st.markdown("**Reviser（阶段二，可选）**")
    revise = st.toggle("开启修订 revise", value=True)
    reviser_backend = st.selectbox(
        "reviser_backend", ["local", "auto", "llm"], index=0,
        help="local=确定性离线改写｜llm=编辑人设（失败回退本地）",
    )
    recheck = st.toggle("修订→复查闸门", value=True)
    rcol1, rcol2 = st.columns(2)
    recheck_threshold = rcol1.number_input("阈值", 0.0, 1.0, 0.7, 0.05)
    recheck_max_rounds = rcol2.number_input("最大轮数", 1, 5, 2, 1)

    st.markdown("**Stage 2 后端**")
    retriever = st.selectbox("retriever", ["hash", "openai", "auto"], index=0,
                             help="hash=离线向量｜openai=真实 embedding")
    nli = st.selectbox("nli", ["heuristic", "llm", "auto"], index=0,
                       help="heuristic=离线蕴含｜llm=LLM judge")

    st.markdown("**Grounding judge（独立报告）**")
    judge = st.selectbox("judge", ["local", "ragas", "trulens"], index=0,
                         help="local=关闭｜ragas/trulens 需 key + 网络；绝不改写五级判决")

    st.markdown("**其他**")
    domain = st.text_input("domain", value="general")
    summary = st.toggle("附报告级 summary", value=True)


# --------------------------------------------------------------------------
# main: input
# --------------------------------------------------------------------------
st.title("🔎 检索内容校验台")
st.caption(
    "审查 LLM 基于网络检索生成的回答：拆解 claim → 向量检索 + NLI 核验 → 五级事实性判定 → "
    "诚实性披露检查 →（可选）修订与复查闸门。四个报告并列、互不覆盖。"
)

if st.session_state.get("show_onboarding"):
    render_onboarding()

col_a, col_b = st.columns([3, 2])

if "question_text" not in st.session_state:
    qt = EXAMPLES / "question.txt"
    st.session_state["question_text"] = qt.read_text(encoding="utf-8").strip() if qt.exists() else ""
if "answer_text" not in st.session_state:
    st.session_state["answer_text"] = ""

with col_a:
    question = st.text_area(
        "问题（claim 锚点）",
        key="question_text",
        height=80,
        help="critic 以此为基准抽取并锚定 claim；缺失会削弱相关性检查。",
    )
with col_b:
    st.markdown("**加载示例**")
    st.caption("examples/ 下的问答与来源，一键填入。")
    if st.button("📥 载入示例回答 + 来源", width="stretch"):
        load_example_into_form()
        st.rerun()

answer = st.text_area(
    "待审回答（answer）",
    key="answer_text",
    height=140,
    placeholder="把 LLM 的回答粘贴到这里……",
)

st.markdown("**检索到的来源（sources）**")
st.caption("每行一个来源：`id` 必填且唯一；`text` 为正文；`traffic` 0–1 表示热度。")
if "sources_rows" not in st.session_state:
    st.session_state["sources_rows"] = load_example_sources()

render_source_import()

df = sources_to_df(st.session_state["sources_rows"])
edited = st.data_editor(
    df, num_rows="dynamic", width='stretch', hide_index=True,
    column_config={
        "id": st.column_config.TextColumn("id", width="small", required=True),
        "title": st.column_config.TextColumn("title", width="medium"),
        "url": st.column_config.TextColumn("url", width="medium"),
        "tag": st.column_config.TextColumn("tag", width="small"),
        "traffic": st.column_config.NumberColumn("traffic", min_value=0.0, max_value=1.0, step=0.05),
        "text": st.column_config.TextColumn("text", width="large"),
    },
    key="sources_editor",
)

with st.expander("高级：直接编辑 sources JSON"):
    json_text = st.text_area(
        "sources JSON", value=json.dumps(edited.fillna("").to_dict("records"), ensure_ascii=False, indent=2),
        height=200, label_visibility="collapsed",
    )

run = st.button("▶️ 运行校验", type="primary", width='stretch')


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def do_validate() -> ValidationResult:
    try:
        rows = json.loads(json_text)
        srcs = [Source(**r) for r in rows]
    except Exception:
        srcs = df_to_sources(edited)

    return validate(
        answer=answer,
        question=question or None,
        sources=srcs,
        domain=domain or "general",
        judge=judge,
        retriever=retriever,
        nli=nli,
        critic_backend=critic_backend,
        revise=revise,
        reviser_backend=reviser_backend,
        recheck=recheck,
        recheck_threshold=recheck_threshold,
        recheck_max_rounds=int(recheck_max_rounds),
        summary=summary,
    )


if run:
    if not answer.strip():
        st.error("请先填入待审回答（answer）。")
    else:
        with st.spinner("校验中……"):
            try:
                st.session_state["result"] = do_validate()
                st.session_state["error"] = None
            except Exception as exc:  # surface, never swallow
                st.session_state["error"] = f"{type(exc).__name__}: {exc}"
                st.session_state["result"] = None

if st.session_state.get("error"):
    st.error(f"校验失败：{st.session_state['error']}")

result: ValidationResult | None = st.session_state.get("result")
if result is None:
    st.info("填入问题与回答（或点「载入示例」），然后点「运行校验」。")
    st.stop()


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------
st.divider()

# --- headline metrics ------------------------------------------------------
f = result.factuality
backend_label = "LLM critic" if result.critic_backend == "llm" else "deterministic（离线规则）"
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("critic 后端", result.critic_backend, help=backend_label)
m2.metric("事实性通过率", f"{f.score:.2f}", f"{f.accurate + f.not_applicable}/{f.total_claims}")
m3.metric("诚实性", f"{result.honesty.score:.2f}", "通过" if result.honesty.passed else "未通过")
m4.metric("接地性 grounding", "—" if not result.grounding or result.grounding.mean_score is None
          else f"{result.grounding.mean_score * 100:.0f}/100")
m5.metric("来源数", result.source_count)

if result.critic_backend != critic_backend and critic_backend in ("llm", "auto"):
    st.warning(
        f"请求 critic_backend=`{critic_backend}`，实际产出判定的是 `{result.critic_backend}`"
        f"（回退不静默，已记 WARNING）。"
    )

tab_fact, tab_report, tab_honesty, tab_grounding, tab_revision, tab_summary, tab_raw = st.tabs(
    ["① 五级事实性", "② claim 报告", "③ 诚实性", "④ 接地性", "⑤ 修订", "汇总", "原始 JSON"]
)

# --- ① factuality ----------------------------------------------------------
with tab_fact:
    dist_cols = st.columns(5)
    for col, v in zip(dist_cols, VERDICT_ORDER):
        label, _ = VERDICT_META[v]
        col.metric(v, getattr(f, v), help=label)

    rows = []
    for cv in f.verdicts:
        rows.append({
            "claim": cv.claim_id,
            "type": cv.claim_type.value,
            "verdict": cv.verdict.value,
            "score": cv.score,
            "rationale": cv.rationale,
            "证据域": ", ".join(cv.evidence_domains) or "—",
            "text": cv.claim_text,
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    for cv in f.verdicts:
        with st.expander(f"{cv.claim_id} · {cv.verdict.value} — {cv.claim_text[:60]}"):
            st.markdown(verdict_badge(cv.verdict.value))
            st.markdown(f"**理由**：{cv.rationale or '—'}")
            st.markdown(f"**类型**：`{cv.claim_type.value}`　**分数**：{cv.score}")
            st.markdown(f"**证据域**：{', '.join(cv.evidence_domains) or '—'}")

    st.caption(f"method = `{f.method}`（来源锚定：只有可靠来源能判 accurate/inaccurate；无来源证实 ≠ 为假）")

# --- ② claim report --------------------------------------------------------
with tab_report:
    for item in result.report.items:
        head = f"{item.claim_id} · {item.type.value} · 可信度 {item.credibility:.2f}"
        with st.expander(head):
            st.markdown(decision_badge(item.decision.value) + f"  —  {item.decision_label}")
            st.markdown(f"**claim**：{item.text}")
            if item.criteria:
                st.markdown("**判据**：" + " ".join(f"`{c.id}`" for c in item.criteria))
                for c in item.criteria:
                    st.caption(f"{c.id}: {c.rule}")
            if item.boundary_conditions:
                st.markdown("**边界条件**：")
                for b in item.boundary_conditions:
                    st.caption(f"· {b}")

            layer_defs = [
                ("l1_primary", "L1 主要证据（可靠性达标）"),
                ("l2_secondary", "L2 次要证据（部分相关）"),
                ("l3_conflicting", "L3 冲突证据"),
            ]
            for key, title in layer_defs:
                refs = getattr(item.layers, key)
                if not refs:
                    continue
                st.markdown(f"**{title}**")
                st.dataframe(
                    pd.DataFrame([{
                        "source_id": r.source_id, "domain": r.domain, "tier": r.tier,
                        "reliability": r.reliability, "coverage": round(r.coverage, 2),
                        "stance": r.stance, "title": r.title, "url": r.url,
                    } for r in refs]),
                    width='stretch', hide_index=True,
                )

# --- ③ honesty -------------------------------------------------------------
with tab_honesty:
    h = result.honesty
    st.metric("诚实性得分", f"{h.score:.2f}", "通过" if h.passed else "未通过")
    checks = [
        ("unsupported_rejected", "H1 无依据断言是否被拒绝/标注"),
        ("sycophancy_checked", "H2 传播度驱动是否坦白（反从众）"),
        ("conflicts_preserved", "H3 来源冲突是否被承认"),
        ("uncertainty_marked", "H4 不确定/检索范围是否说明"),
    ]
    hc = st.columns(4)
    for col, (attr, label) in zip(hc, checks):
        val = getattr(h, attr)
        col.metric(label.split(" ")[0], "✅" if val else "❌", help=label)
    st.caption("H5（claim 是否回答问题）在 question_anchor / off-topic 中单独报告。")
    if h.flags:
        st.markdown("**命中项 flags**：")
        for fl in h.flags:
            st.caption(f"· {fl}")
    st.text(format_honesty(h))

# --- ④ grounding -----------------------------------------------------------
with tab_grounding:
    g = result.grounding
    if g is None:
        st.info("未运行 grounding judge（judge=`local`）。选 `ragas`/`trulens` 且配置 key 后重跑。")
    else:
        gc = st.columns(4)
        gc[0].metric("method", g.method)
        gc[1].metric("grounded", g.grounded_count)
        gc[2].metric("ungrounded", g.ungrounded_count)
        gc[3].metric("mean", "—" if g.mean_score is None else round(g.mean_score, 2))
        st.dataframe(
            pd.DataFrame([{
                "claim": i.claim_id, "type": i.claim_type.value, "grounded": i.grounded,
                "score": i.score, "rationale": i.rationale, "text": i.claim_text,
            } for i in g.items]),
            width='stretch', hide_index=True,
        )
        st.caption("Grounding 锚定的是**传入的检索上下文**（内部忠实性），与五级判决分开报告、绝不覆盖。")

# --- ⑤ revision ------------------------------------------------------------
with tab_revision:
    r = result.revision
    if r is None:
        st.info("未开启修订（revise=False）。")
    else:
        if r.fallback:
            st.warning("LLM reviser 失败，已回退本地确定性改写器。")
        left, right = st.columns(2)
        left.markdown("**原回答（永不被就地修改）**")
        left.text(result.answer)
        right.markdown("**修订文本 revised_text**")
        right.text(r.revised_text)

        st.markdown(f"改动 claim 数：**{r.changed}**　·　method = `{r.method}`　·　"
                    f"终止标记 `{r.end_marker}` 出现：{r.end_marker_present}")
        if r.revision_log:
            st.dataframe(
                pd.DataFrame([{
                    "claim": i.claim_id, "action": i.action, "criteria": ", ".join(i.criteria),
                    "reason": i.reason, "revised_text": i.revised_text,
                } for i in r.revision_log]),
                width='stretch', hide_index=True,
            )
        if r.diff:
            st.markdown("**修订 diff**")
            st.code(r.diff, language="diff")
        st.markdown("**Reviser 据以改写的输入 verdict**")
        st.code("\n".join(r.input_verdicts) or "—", language="text")

        if r.recheck is not None:
            rc = r.recheck
            st.markdown("---")
            st.markdown("**修订→复查闸门**（对修订文本重跑来源锚定判定）")
            cc = st.columns(4)
            cc[0].metric("通过", "✅" if rc.passed else "❌")
            cc[1].metric("首轮分", rc.initial_score)
            cc[2].metric("最终分", rc.final_score)
            cc[3].metric("轮数", len(rc.rounds))
            st.dataframe(
                pd.DataFrame([{
                    "round": rr.round, "factuality_score": rr.factuality_score,
                    "inaccurate": rr.inaccurate, "new_未获证": len(rr.new_inaccurate),
                    "changed": rr.changed, "passed": rr.passed, "method": rr.method,
                } for rr in rc.rounds]),
                width='stretch', hide_index=True,
            )
            if rc.leftover_inaccurate:
                st.warning("仍残留新增未获证 claim：" + "；".join(rc.leftover_inaccurate))

# --- summary ---------------------------------------------------------------
with tab_summary:
    s = result.summary
    if s is None:
        st.info("未附 summary（summary=False）。")
    else:
        st.text(s.text)
        st.caption("summary 只重述各路信号（五级分布 + 通过率、grounding 0–100、honesty 比例），**不是单一总分**。")

# --- raw -------------------------------------------------------------------
with tab_raw:
    payload = result.model_dump_json(indent=2)
    st.download_button("⬇️ 下载 ValidationResult.json", payload, file_name="ValidationResult.json",
                       mime="application/json")
    st.code(payload, language="json")


# --------------------------------------------------------------------------
# plain-text reports (parity with the CLI)
# --------------------------------------------------------------------------
with st.expander("纯文本报告（与 CLI 输出一致）"):
    st.text(format_report(result.report))
    st.text(format_factuality(result.factuality))
    if result.grounding is not None:
        st.text(format_grounding(result.grounding))
    if result.honesty is not None:
        st.text(format_honesty(result.honesty))
