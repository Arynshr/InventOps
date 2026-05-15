"""
dashboard/app.py — InventOps Model Visibility Dashboard
Run: streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make sure project root is importable when running from any cwd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dashboard.queries as Q

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH = os.getenv("INVENTOPS_DB", str(ROOT / "metrics" / "inventops.db"))

PALETTE = {
    "bg":       "#0d1117",
    "surface":  "#161b22",
    "border":   "#30363d",
    "accent":   "#58a6ff",
    "green":    "#3fb950",
    "red":      "#f85149",
    "orange":   "#d29922",
    "purple":   "#bc8cff",
    "text":     "#e6edf3",
    "subtext":  "#8b949e",
}

ACTION_COLORS = {
    "order":    PALETTE["green"],
    "transfer": PALETTE["accent"],
    "hold":     PALETTE["orange"],
}

TASKS = ["easy", "medium", "hard"]

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="InventOps · Model Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

      html, body, [class*="css"] {{
          font-family: 'Inter', sans-serif;
          background-color: {PALETTE["bg"]};
          color: {PALETTE["text"]};
      }}

      /* Sidebar */
      [data-testid="stSidebar"] {{
          background-color: {PALETTE["surface"]};
          border-right: 1px solid {PALETTE["border"]};
      }}

      /* Metric cards */
      [data-testid="metric-container"] {{
          background: {PALETTE["surface"]};
          border: 1px solid {PALETTE["border"]};
          border-radius: 12px;
          padding: 16px 20px;
          transition: border-color 0.2s ease;
      }}
      [data-testid="metric-container"]:hover {{
          border-color: {PALETTE["accent"]};
      }}

      /* Tabs */
      [data-testid="stTabs"] button {{
          font-weight: 500;
          color: {PALETTE["subtext"]};
      }}
      [data-testid="stTabs"] button[aria-selected="true"] {{
          color: {PALETTE["accent"]};
          border-bottom: 2px solid {PALETTE["accent"]};
      }}

      /* Plotly backgrounds */
      .js-plotly-plot .plotly {{
          border-radius: 12px;
      }}

      /* Header gradient */
      .dash-header {{
          background: linear-gradient(135deg, #1a2233 0%, #0d1117 60%);
          border: 1px solid {PALETTE["border"]};
          border-radius: 16px;
          padding: 24px 32px;
          margin-bottom: 24px;
      }}
      .dash-header h1 {{
          margin: 0;
          font-size: 2rem;
          font-weight: 700;
          background: linear-gradient(90deg, {PALETTE["accent"]}, {PALETTE["purple"]});
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
      }}
      .dash-header p {{
          margin: 4px 0 0;
          color: {PALETTE["subtext"]};
          font-size: 0.9rem;
      }}

      /* Empty state */
      .empty-state {{
          background: {PALETTE["surface"]};
          border: 1px dashed {PALETTE["border"]};
          border-radius: 12px;
          padding: 48px;
          text-align: center;
          color: {PALETTE["subtext"]};
      }}

      /* Divider */
      hr {{ border-color: {PALETTE["border"]}; margin: 20px 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _chart_layout(title: str = "", height: int = 360) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=PALETTE["text"], size=14)),
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["surface"],
        font=dict(color=PALETTE["text"], family="Inter"),
        height=height,
        margin=dict(l=16, r=16, t=40, b=16),
        xaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
        yaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
    )


def _empty(msg: str = "Run an episode first to populate this chart."):
    st.markdown(
        f'<div class="empty-state">📭 {msg}</div>',
        unsafe_allow_html=True,
    )


def _db_exists() -> bool:
    return Path(DB_PATH).exists()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📦 InventOps")
    _c_sub = PALETTE["subtext"]
    st.markdown(f"<span style='color:{_c_sub};font-size:0.8rem;'>Model Visibility Dashboard</span>", unsafe_allow_html=True)
    st.divider()

    st.markdown("**Database**")
    st.code(DB_PATH, language=None)
    if _db_exists():
        size_kb = Path(DB_PATH).stat().st_size / 1024
        _c_green = PALETTE["green"]
        st.markdown(f"<span style='color:{_c_green};'>● Connected</span> · `{size_kb:.1f} KB`", unsafe_allow_html=True)
    else:
        _c_orange = PALETTE["orange"]
        st.markdown(f"<span style='color:{_c_orange};'>⚠ No data yet</span>", unsafe_allow_html=True)
        st.caption("Run `python inference.py` or `python evaluate.py` to populate.")

    st.divider()
    st.markdown("**Filter by Task**")
    selected_task = st.selectbox("Task", ["All"] + TASKS, label_visibility="collapsed")
    task_filter = None if selected_task == "All" else selected_task

    st.markdown("**Auto-refresh**")
    auto_refresh = st.toggle("Refresh every 10s", value=False)
    if auto_refresh:
        import time
        st.caption("Page will auto-refresh.")
        time.sleep(10)
        st.rerun()

    st.divider()
    if st.button("🔄 Refresh now", use_container_width=True):
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="dash-header">
        <h1>📦 InventOps Dashboard</h1>
        <p>Supply chain RL · Agent performance · RLVR loop · Inference health</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_overview, tab_episodes, tab_rlvr, tab_inference = st.tabs([
    "🏠 Overview", "📈 Episodes", "🔁 RLVR Loop", "⚡ Inference"
])


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 1 · Overview                                           ║
# ╚══════════════════════════════════════════════════════════════╝

with tab_overview:
    if not _db_exists():
        _empty("No database found. Run `python inference.py` or `python evaluate.py --seeds 5` to generate data.")
    else:
        kpis = Q.summary_kpis(DB_PATH)

        # KPI row
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total Episodes", kpis["total_episodes"])
        col2.metric("Success Rate", f"{kpis['success_rate']}%")
        col3.metric("Mean Score", f"{kpis['mean_score']:.3f}")
        col4.metric("Total Steps", kpis["total_steps"])
        col5.metric("RLVR Rounds", kpis["rlvr_rounds"])
        col6.metric("Parse Errors", kpis["parse_errors"])

        st.divider()

        col_a, col_b = st.columns([3, 2])

        with col_a:
            df_task = Q.score_by_task(DB_PATH)
            if df_task.empty:
                _empty("Score data not available.")
            else:
                fig = px.bar(
                    df_task,
                    x="task_id", y="mean_score",
                    color="agent",
                    barmode="group",
                    error_y=None,
                    color_discrete_map={
                        "llm": PALETTE["accent"],
                        "hold-only": PALETTE["subtext"],
                        "random": PALETTE["orange"],
                        "groq-llm": PALETTE["green"],
                        "groq-optimized": PALETTE["purple"],
                    },
                    labels={"mean_score": "Mean Score", "task_id": "Task", "agent": "Agent"},
                )
                fig.update_layout(**_chart_layout("Mean Score by Task & Agent", 360))
                fig.update_traces(marker_line_width=0)
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            recent = Q.recent_runs(DB_PATH, limit=10)
            if recent.empty:
                _empty()
            else:
                st.markdown("**Recent Runs**")
                st.dataframe(
                    recent[["task_id", "agent", "score", "success", "steps", "recorded_at"]],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "score": st.column_config.NumberColumn(format="%.3f"),
                        "success": st.column_config.CheckboxColumn(),
                    },
                )


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 2 · Episodes                                           ║
# ╚══════════════════════════════════════════════════════════════╝

with tab_episodes:
    if not _db_exists():
        _empty()
    else:
        run_ids = Q.available_run_ids(DB_PATH)
        seeds = Q.available_seeds(DB_PATH, task_id=task_filter)

        col_r, col_s = st.columns([2, 1])
        with col_r:
            chosen_run = st.selectbox(
                "Run ID", ["All"] + run_ids,
                help="Select a specific inference/evaluate run"
            )
        with col_s:
            chosen_seed = st.selectbox(
                "Seed", ["All"] + [str(s) for s in seeds]
            )

        run_filter = None if chosen_run == "All" else chosen_run
        seed_filter = None if chosen_seed == "All" else int(chosen_seed)

        steps_df = Q.reward_curve(DB_PATH, task_id=task_filter, seed=seed_filter, run_id=run_filter)

        # ── Reward curve ──────────────────────────────────────────────────────
        st.markdown("#### Reward per Step")
        if steps_df.empty:
            _empty()
        else:
            fig = go.Figure()
            # Group by seed to draw individual traces
            for seed_val, grp in steps_df.groupby("seed"):
                fig.add_trace(go.Scatter(
                    x=grp["step"], y=grp["reward"],
                    mode="lines",
                    name=f"seed={seed_val}",
                    line=dict(width=1.5),
                    opacity=0.8,
                ))
            # Rolling mean overlay
            if len(steps_df) > 5:
                mean_by_step = steps_df.groupby("step")["reward"].mean().reset_index()
                fig.add_trace(go.Scatter(
                    x=mean_by_step["step"], y=mean_by_step["reward"],
                    mode="lines", name="mean",
                    line=dict(color=PALETTE["accent"], width=2.5, dash="dot"),
                ))
            fig.update_layout(**_chart_layout("", 340))
            fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["border"])
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        col_pie, col_bar, col_fail = st.columns(3)

        with col_pie:
            st.markdown("#### Action Distribution")
            act_df = Q.action_distribution(DB_PATH, task_id=task_filter)
            if act_df.empty:
                _empty()
            else:
                fig = px.pie(
                    act_df, names="action_type", values="count",
                    color="action_type",
                    color_discrete_map=ACTION_COLORS,
                    hole=0.45,
                )
                fig.update_layout(**_chart_layout("", 300))
                fig.update_traces(textfont_color=PALETTE["text"])
                st.plotly_chart(fig, use_container_width=True)

        with col_bar:
            st.markdown("#### Reward Components")
            rb_df = Q.reward_breakdown_agg(DB_PATH, task_id=task_filter)
            if rb_df.empty or rb_df.isnull().all().all():
                _empty("Reward breakdown not in data\n(requires env to return reward_breakdown in info dict).")
            else:
                melted = rb_df.T.reset_index()
                melted.columns = ["component", "avg_value"]
                melted = melted.dropna()
                fig = px.bar(
                    melted, x="avg_value", y="component",
                    orientation="h",
                    color="avg_value",
                    color_continuous_scale=["#f85149", "#d29922", "#3fb950"],
                )
                fig.update_layout(**_chart_layout("", 300))
                fig.update_coloraxes(showscale=False)
                st.plotly_chart(fig, use_container_width=True)

        with col_fail:
            st.markdown("#### Failure Reasons")
            fail_df = Q.failure_reasons(DB_PATH, task_id=task_filter)
            if fail_df.empty:
                st.markdown(
                    f"<div class='empty-state' style='padding:24px;'>✅ No failures recorded</div>",
                    unsafe_allow_html=True,
                )
            else:
                fig = px.bar(
                    fail_df.head(10), x="occurrences", y="failure_reason",
                    orientation="h",
                    color_discrete_sequence=[PALETTE["red"]],
                )
                fig.update_layout(**_chart_layout("", 300))
                st.plotly_chart(fig, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 3 · RLVR Loop                                         ║
# ╚══════════════════════════════════════════════════════════════╝

with tab_rlvr:
    if not _db_exists():
        _empty()
    else:
        rlvr_df = Q.rlvr_progression(DB_PATH, task_id=task_filter)

        if rlvr_df.empty:
            _empty("No RLVR runs found. Run `python rlvr/prompt_optimizer.py` to populate.")
        else:
            st.markdown("#### Score Progression per Round")

            fig = go.Figure()
            for run_id_val, grp in rlvr_df.groupby("run_id"):
                label = run_id_val[:16]
                fig.add_trace(go.Scatter(
                    x=grp["round_num"], y=grp["mean_score"],
                    mode="lines+markers",
                    name=f"run {label}",
                    line=dict(width=2),
                    marker=dict(size=8),
                ))
                # Error band (min/max)
                fig.add_trace(go.Scatter(
                    x=list(grp["round_num"]) + list(grp["round_num"])[::-1],
                    y=list(grp["max_score"]) + list(grp["min_score"])[::-1],
                    fill="toself",
                    fillcolor="rgba(88,166,255,0.08)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    name=f"{label} band",
                ))

            fig.update_layout(**_chart_layout("Mean Score vs Prompt Round", 400))
            fig.update_xaxes(title="Prompt Round", dtick=1)
            fig.update_yaxes(title="Score", range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            col_tbl, col_err = st.columns([3, 2])

            with col_tbl:
                st.markdown("#### Round Summary Table")
                show_cols = ["run_id", "task_id", "round_num", "mean_score", "min_score", "max_score", "recorded_at"]
                st.dataframe(
                    rlvr_df[show_cols].sort_values(["run_id", "round_num"]),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "mean_score": st.column_config.NumberColumn(format="%.4f"),
                        "min_score":  st.column_config.NumberColumn(format="%.4f"),
                        "max_score":  st.column_config.NumberColumn(format="%.4f"),
                    },
                )

            with col_err:
                st.markdown("#### Failure Types (RLVR)")
                fail_df = Q.rlvr_failure_summary(DB_PATH)
                if fail_df.empty:
                    st.markdown("<div class='empty-state' style='padding:24px;'>✅ No failures</div>", unsafe_allow_html=True)
                else:
                    fig = px.bar(
                        fail_df.head(10), x="count", y="failure_reason",
                        orientation="h",
                        color_discrete_sequence=[PALETTE["orange"]],
                    )
                    fig.update_layout(**_chart_layout("", 300))
                    st.plotly_chart(fig, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 4 · Inference Health                                   ║
# ╚══════════════════════════════════════════════════════════════╝

with tab_inference:
    if not _db_exists():
        _empty()
    else:
        col_lat, col_err = st.columns(2)

        with col_lat:
            st.markdown("#### LLM Latency Distribution")
            lat_df = Q.llm_latency_hist(DB_PATH)
            if lat_df.empty:
                _empty("No latency data. Run `python inference.py` (not evaluate.py) to collect LLM timing.")
            else:
                p50 = lat_df["llm_latency_ms"].median()
                p95 = lat_df["llm_latency_ms"].quantile(0.95)
                p99 = lat_df["llm_latency_ms"].quantile(0.99)

                m1, m2, m3 = st.columns(3)
                m1.metric("p50", f"{p50:.0f} ms")
                m2.metric("p95", f"{p95:.0f} ms")
                m3.metric("p99", f"{p99:.0f} ms")

                fig = px.histogram(
                    lat_df, x="llm_latency_ms",
                    nbins=40,
                    color_discrete_sequence=[PALETTE["accent"]],
                    labels={"llm_latency_ms": "Latency (ms)"},
                )
                fig.update_layout(**_chart_layout("", 280))
                fig.add_vline(x=p50, line_dash="dash", line_color=PALETTE["green"],
                              annotation_text="p50", annotation_font_color=PALETTE["green"])
                fig.add_vline(x=p95, line_dash="dash", line_color=PALETTE["orange"],
                              annotation_text="p95", annotation_font_color=PALETTE["orange"])
                st.plotly_chart(fig, use_container_width=True)

        with col_err:
            st.markdown("#### Parse Error Rate Over Time")
            err_df = Q.parse_error_rate_over_time(DB_PATH)
            if err_df.empty:
                _empty()
            else:
                err_df["error_rate"] = err_df["failures"] / err_df["total_steps"].clip(lower=1)
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=err_df["minute"], y=err_df["total_steps"],
                    name="Total Steps",
                    marker_color=PALETTE["border"],
                ))
                fig.add_trace(go.Scatter(
                    x=err_df["minute"], y=err_df["error_rate"],
                    name="Error Rate",
                    yaxis="y2",
                    line=dict(color=PALETTE["red"], width=2),
                    mode="lines+markers",
                ))
                fig.update_layout(
                    **_chart_layout("", 280),
                    yaxis2=dict(
                        overlaying="y", side="right",
                        title="Error Rate",
                        gridcolor="transparent",
                        tickformat=".0%",
                        range=[0, 1],
                    ),
                    legend=dict(orientation="h", y=-0.15),
                )
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("#### Raw Step Log")
        raw_df = Q.reward_curve(DB_PATH, task_id=task_filter)
        if not raw_df.empty:
            st.dataframe(
                raw_df[["run_id", "task_id", "seed", "step", "action_type",
                        "reward", "failure_reason", "llm_latency_ms"]].tail(200),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "reward": st.column_config.NumberColumn(format="%.3f"),
                    "llm_latency_ms": st.column_config.NumberColumn("Latency (ms)", format="%.0f"),
                },
            )
