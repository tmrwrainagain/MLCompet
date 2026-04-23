"""
Module B — Streamlit Interactive Dashboard.

Launch:
    streamlit run dashboard.py

Features:
  - Three access levels: admin / teacher / student
  - Auto-refresh (TTL-based cache, 60 s)
  - Analytics panels covering all six required metrics
  - Clustering visualisation with quality metrics
  - Full data table (admin only)
"""

import importlib.util as _ilu
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

_b_cfg_path = Path(__file__).parent / "config.py"
_b_cfg_spec = _ilu.spec_from_file_location("_module_b_dashboard_config", _b_cfg_path)
if _b_cfg_spec is None or _b_cfg_spec.loader is None:
    raise ImportError(f"Cannot load Module B config from {_b_cfg_path}")
_b_cfg = _ilu.module_from_spec(_b_cfg_spec)
_b_cfg_spec.loader.exec_module(_b_cfg)

DATABASE_URL = _b_cfg.DATABASE_URL
USERS = _b_cfg.USERS
ROLE_PAGES = _b_cfg.ROLE_PAGES
LESSON_TYPE_LABELS = _b_cfg.LESSON_TYPE_LABELS
DASHBOARD_REFRESH_SECONDS = _b_cfg.DASHBOARD_REFRESH_SECONDS

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Аналитика учебных материалов",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_session():
    defaults = {
        "authenticated": False,
        "username":      "",
        "role":          "",
        "name":          "",
        "last_refresh":  0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Authentication
# ─────────────────────────────────────────────────────────────────────────────

def _login_form():
    st.markdown("## 🔐 Вход в систему")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Логин")
            password = st.text_input("Пароль", type="password")
            submitted = st.form_submit_button("Войти", use_container_width=True)
        if submitted:
            user = USERS.get(username)
            if user and user["password"] == password:
                st.session_state["authenticated"] = True
                st.session_state["username"]      = username
                st.session_state["role"]          = user["role"]
                st.session_state["name"]          = user["name"]
                st.rerun()
            else:
                st.error("Неверный логин или пароль.")


def _can_access(page: str) -> bool:
    role  = st.session_state.get("role", "")
    pages = ROLE_PAGES.get(role, [])
    return page in pages


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (cached with TTL)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=DASHBOARD_REFRESH_SECONDS)
def load_materials() -> pd.DataFrame:
    import psycopg2
    import psycopg2.extras
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT m.*,
                   COALESCE(mf_wc.feature_value, 0) AS word_count,
                   COALESCE(mf_sl.feature_value, 0) AS avg_sentence_length,
                   COALESCE(mf_mc.feature_value, 0) AS media_count,
                   COALESCE(mf_hi.feature_value, 0) AS has_images,
                   COALESCE(mf_hv.feature_value, 0) AS has_videos,
                   COALESCE(mf_hq.feature_value, 0) AS has_questions
            FROM materials m
            LEFT JOIN material_features mf_wc ON mf_wc.material_id = m.id AND mf_wc.feature_name = 'word_count'
            LEFT JOIN material_features mf_sl ON mf_sl.material_id = m.id AND mf_sl.feature_name = 'avg_sentence_length'
            LEFT JOIN material_features mf_mc ON mf_mc.material_id = m.id AND mf_mc.feature_name = 'media_count'
            LEFT JOIN material_features mf_hi ON mf_hi.material_id = m.id AND mf_hi.feature_name = 'has_images'
            LEFT JOIN material_features mf_hv ON mf_hv.material_id = m.id AND mf_hv.feature_name = 'has_videos'
            LEFT JOIN material_features mf_hq ON mf_hq.material_id = m.id AND mf_hq.feature_name = 'has_questions'
            ORDER BY m.id
        """)
        rows = cur.fetchall()
        conn.close()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
    except Exception as e:
        st.error(f"Ошибка подключения к БД: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=DASHBOARD_REFRESH_SECONDS)
def load_compliance_summary() -> pd.DataFrame:
    import psycopg2
    import psycopg2.extras
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT m.subject, m.is_generated,
                   mr.category, mr.requirement,
                   AVG(mc.score)  AS avg_score,
                   COUNT(mc.id)   AS sample_count,
                   SUM(CASE WHEN mc.is_met THEN 1 ELSE 0 END)::REAL / NULLIF(COUNT(mc.id),0) AS met_ratio
            FROM methodology_compliance mc
            JOIN materials              m  ON m.id  = mc.material_id
            JOIN methodology_requirements mr ON mr.id = mc.requirement_id
            GROUP BY m.subject, m.is_generated, mr.category, mr.requirement
            ORDER BY m.subject, mr.category, mr.requirement
        """)
        rows = cur.fetchall()
        conn.close()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _sidebar(df: pd.DataFrame) -> str:
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state['name']}")
        st.caption(f"Роль: **{st.session_state['role']}**")
        st.divider()

        pages_map = {
            "overview":     "📊 Обзор",
            "analytics":    "📈 Аналитика материалов",
            "requirements": "✅ Методические требования",
            "clustering":   "🔵 Кластеризация",
            "data":         "🗄️ Данные",
        }

        allowed = ROLE_PAGES.get(st.session_state["role"], [])
        options = [pages_map[p] for p in pages_map if p in allowed]
        selected_label = st.radio("Навигация", options)
        page = [k for k, v in pages_map.items() if v == selected_label][0]

        st.divider()
        st.caption(f"Материалов в БД: **{len(df)}**")
        st.caption(f"Обновление: каждые {DASHBOARD_REFRESH_SECONDS} сек.")
        if st.button("🔄 Обновить сейчас"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        if st.button("🚪 Выйти"):
            for k in ["authenticated", "username", "role", "name"]:
                st.session_state[k] = "" if k != "authenticated" else False
            st.rerun()

    return page


# ─────────────────────────────────────────────────────────────────────────────
# Page: Overview
# ─────────────────────────────────────────────────────────────────────────────

def page_overview(df: pd.DataFrame):
    st.title("📊 Обзор набора учебных материалов")
    st.caption(f"Данные обновлены: {pd.Timestamp.now().strftime('%H:%M:%S')}")

    # KPI row
    total    = len(df)
    n_gen    = int(df["is_generated"].sum())   if "is_generated"    in df.columns else 0
    n_orig   = total - n_gen
    n_subj   = df["subject"].nunique()          if "subject"         in df.columns else 0
    avg_sc   = round(df["compliance_score"].dropna().mean(), 2) if "compliance_score" in df.columns else 0
    n_appr   = int((df["moderation_status"] == "approved").sum()) if "moderation_status" in df.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Материалов всего",   total)
    c2.metric("Исходных",           n_orig)
    c3.metric("Сгенерированных",    n_gen)
    c4.metric("Предметов",          n_subj)
    c5.metric("Ср. оценка (0–10)",  avg_sc)

    st.divider()

    col1, col2 = st.columns(2)

    # Distribution by subject
    with col1:
        if "subject" in df.columns and not df["subject"].dropna().empty:
            subj_cnt = df["subject"].value_counts().reset_index()
            subj_cnt.columns = ["Предмет", "Кол-во"]
            fig = px.bar(subj_cnt, x="Кол-во", y="Предмет", orientation="h",
                         title="Материалы по предметам",
                         color="Кол-во", color_continuous_scale="Blues")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Distribution by lesson type
    with col2:
        if "lesson_type" in df.columns and not df["lesson_type"].dropna().empty:
            lt_cnt = df["lesson_type"].map(
                lambda x: LESSON_TYPE_LABELS.get(x, x)
            ).value_counts().reset_index()
            lt_cnt.columns = ["Тип занятия", "Кол-во"]
            fig = px.pie(lt_cnt, names="Тип занятия", values="Кол-во",
                         title="Распределение по типам занятий",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

    # Moderation status
    if "moderation_status" in df.columns:
        mod_cnt = df["moderation_status"].value_counts().reset_index()
        mod_cnt.columns = ["Статус", "Кол-во"]
        color_map = {"approved": "#4CAF50", "rejected": "#F44336",
                     "needs_revision": "#FF9800", "pending": "#9E9E9E", "error": "#795548"}
        fig = px.bar(mod_cnt, x="Статус", y="Кол-во",
                     title="Статусы модерации",
                     color="Статус", color_discrete_map=color_map)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Analytics — all 6 required metrics
# ─────────────────────────────────────────────────────────────────────────────

def page_analytics(df: pd.DataFrame, comp_df: pd.DataFrame):
    st.title("📈 Аналитика учебных материалов")

    # ── Metric 1: Topic coverage per subject vs average ──────────────────
    st.header("1. Уровень покрытия тем по предметам (относительно среднего)")
    if "subject" in df.columns and "topic" in df.columns:
        cov = (
            df.groupby("subject")["topic"]
              .nunique()
              .reset_index(name="unique_topics")
        )
        avg_cov = cov["unique_topics"].mean()
        cov["vs_avg_%"] = ((cov["unique_topics"] - avg_cov) / avg_cov * 100).round(1)
        cov["color"]    = cov["vs_avg_%"].apply(lambda x: "above" if x >= 0 else "below")

        fig = px.bar(
            cov.sort_values("vs_avg_%"),
            x="vs_avg_%", y="subject", orientation="h",
            color="color",
            color_discrete_map={"above": "#4CAF50", "below": "#F44336"},
            title=f"Отклонение покрытия тем от среднего ({avg_cov:.1f} тем/предмет)",
            labels={"vs_avg_%": "Отклонение от среднего, %", "subject": "Предмет"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cov[["subject", "unique_topics", "vs_avg_%"]].rename(
            columns={"subject": "Предмет", "unique_topics": "Уникальных тем",
                     "vs_avg_%": "Отклонение от среднего, %"}
        ), use_container_width=True)
    else:
        st.info("Данные о предметах или темах отсутствуют.")

    st.divider()

    # ── Metric 2: Share of generated materials ────────────────────────────
    st.header("2. Доля автоматически сгенерированных материалов")
    if "is_generated" in df.columns and "subject" in df.columns:
        gen_by_subj = (
            df.groupby("subject")
              .apply(lambda g: pd.Series({
                  "total":     len(g),
                  "generated": int(g["is_generated"].sum()),
              }))
              .reset_index()
        )
        gen_by_subj["gen_pct"] = (gen_by_subj["generated"] / gen_by_subj["total"] * 100).round(1)

        overall_pct = round(df["is_generated"].mean() * 100, 1)
        st.metric("По всем предметам", f"{overall_pct} %")

        fig = px.bar(
            gen_by_subj.sort_values("gen_pct", ascending=False),
            x="subject", y="gen_pct",
            color="gen_pct", color_continuous_scale="Oranges",
            title="Доля сгенерированных материалов по предметам, %",
            labels={"subject": "Предмет", "gen_pct": "Доля, %"},
        )
        fig.add_hline(y=overall_pct, line_dash="dash",
                      annotation_text=f"Среднее: {overall_pct}%",
                      annotation_position="top right")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Нет данных о генерации.")

    st.divider()

    # ── Metric 3: Distribution by lesson type ─────────────────────────────
    st.header("3. Распределение по типам занятий")
    if "lesson_type" in df.columns and "subject" in df.columns:
        df_lt = df.copy()
        df_lt["lesson_label"] = df_lt["lesson_type"].map(
            lambda x: LESSON_TYPE_LABELS.get(x, x)
        )

        tab_all, tab_by_subj = st.tabs(["Все предметы", "По предметам"])

        with tab_all:
            lt_all = df_lt["lesson_label"].value_counts().reset_index()
            lt_all.columns = ["Тип занятия", "Кол-во"]
            fig = px.pie(lt_all, names="Тип занятия", values="Кол-во",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)

        with tab_by_subj:
            pivot = (
                df_lt.groupby(["subject", "lesson_label"])
                     .size()
                     .reset_index(name="count")
            )
            fig = px.bar(
                pivot, x="subject", y="count", color="lesson_label",
                barmode="stack",
                title="Типы занятий по предметам",
                labels={"subject": "Предмет", "count": "Кол-во",
                        "lesson_label": "Тип занятия"},
            )
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Данные о типах занятий отсутствуют.")

    st.divider()

    # ── Metric 4: TOP requirements (per subject vs average) ───────────────
    st.header("4–5. ТОП выполняемых / невыполняемых требований по предметам")
    if not comp_df.empty:
        global_avg = comp_df.groupby("requirement")["met_ratio"].mean().reset_index()
        global_avg.columns = ["requirement", "global_avg"]

        subj_req = (
            comp_df.groupby(["subject", "requirement"])["met_ratio"]
              .mean()
              .reset_index()
              .merge(global_avg, on="requirement")
        )
        subj_req["vs_avg"] = (subj_req["met_ratio"] - subj_req["global_avg"]).round(3)

        subjects = sorted(subj_req["subject"].dropna().unique().tolist())
        sel_subj = st.selectbox("Предмет", subjects, key="top_req_subj")

        sub = subj_req[subj_req["subject"] == sel_subj].sort_values("vs_avg", ascending=False)
        top5    = sub.head(5)[["requirement", "met_ratio", "vs_avg"]]
        bottom5 = sub.tail(5)[["requirement", "met_ratio", "vs_avg"]]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**✅ Топ-5 наиболее выполняемых**")
            st.dataframe(
                top5.rename(columns={"requirement": "Требование",
                                     "met_ratio": "Выполнение",
                                     "vs_avg": "vs Среднее"}),
                use_container_width=True,
            )
        with c2:
            st.markdown("**❌ Топ-5 наименее выполняемых**")
            st.dataframe(
                bottom5.rename(columns={"requirement": "Требование",
                                        "met_ratio": "Выполнение",
                                        "vs_avg": "vs Среднее"}),
                use_container_width=True,
            )

        # Waterfall chart
        plot_df = pd.concat([top5, bottom5]).drop_duplicates()
        fig = px.bar(
            plot_df, x="vs_avg", y="requirement", orientation="h",
            color="vs_avg",
            color_continuous_scale="RdYlGn",
            title=f"Отклонение выполнения требований от среднего — «{sel_subj}»",
            labels={"vs_avg": "Отклонение от среднего", "requirement": "Требование"},
        )
        fig.add_vline(x=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Данные о выполнении требований отсутствуют. Запустите модерацию в Модуле А.")

    st.divider()

    # ── Metric 6: Original vs Generated requirements ──────────────────────
    st.header("6. Требования: исходные vs сгенерированные материалы")
    if not comp_df.empty and "is_generated" in comp_df.columns:
        comp_pivot = (
            comp_df.groupby(["requirement", "is_generated"])["met_ratio"]
              .mean()
              .reset_index()
        )
        comp_pivot["тип"] = comp_pivot["is_generated"].map(
            {True: "Сгенерированный", False: "Исходный"}
        )
        fig = px.bar(
            comp_pivot, x="met_ratio", y="requirement",
            color="тип", barmode="group", orientation="h",
            title="Выполнение требований: исходные vs сгенерированные",
            labels={"met_ratio": "Доля выполнения", "requirement": "Требование"},
            color_discrete_map={"Исходный": "#1565C0", "Сгенерированный": "#E65100"},
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Нет данных для сравнения.")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Methodology requirements
# ─────────────────────────────────────────────────────────────────────────────

def page_requirements(comp_df: pd.DataFrame):
    st.title("✅ Методические требования")

    if comp_df.empty:
        st.warning("Данные о методических требованиях отсутствуют. "
                   "Запустите модерацию через Модуль А.")
        return

    # Coverage by category
    st.header("Обеспеченность требований по категориям")
    cat_agg = (
        comp_df.groupby("category")["met_ratio"]
          .mean()
          .reset_index()
          .sort_values("met_ratio", ascending=False)
    )
    cat_agg["met_pct"] = (cat_agg["met_ratio"] * 100).round(1)

    fig = px.bar(
        cat_agg, x="met_pct", y="category", orientation="h",
        color="met_pct", color_continuous_scale="Greens",
        title="Средний % выполнения требований по категориям",
        labels={"met_pct": "% выполнения", "category": "Категория"},
        range_x=[0, 100],
    )
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap: subject × category
    st.subheader("Тепловая карта: предмет × категория")
    if "subject" in comp_df.columns:
        heat = (
            comp_df.groupby(["subject", "category"])["met_ratio"]
              .mean()
              .unstack(fill_value=0)
              .mul(100)
              .round(1)
        )
        if not heat.empty:
            fig = px.imshow(
                heat,
                color_continuous_scale="RdYlGn",
                zmin=0, zmax=100,
                aspect="auto",
                title="% выполнения требований (предмет × категория)",
                labels={"x": "Категория", "y": "Предмет", "color": "%"},
            )
            st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.subheader("Детальная таблица требований")
    detail = (
        comp_df.groupby(["category", "requirement"])["met_ratio"]
          .mean()
          .reset_index()
          .sort_values(["category", "met_ratio"], ascending=[True, False])
    )
    detail["Выполнение, %"] = (detail["met_ratio"] * 100).round(1)
    st.dataframe(
        detail[["category", "requirement", "Выполнение, %"]].rename(
            columns={"category": "Категория", "requirement": "Требование"}
        ),
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page: Clustering
# ─────────────────────────────────────────────────────────────────────────────

def page_clustering(df: pd.DataFrame):
    st.title("🔵 Кластеризация учебных материалов")

    cluster_defs = [
        ("parallel_cluster",   "parallel_cluster_label",   "Параллельное изучение",
         "Материалы из независимых последовательностей, которые можно изучать одновременно."),
        ("sequential_cluster", "sequential_cluster_label", "Последовательное изучение",
         "Материалы, образующие логическую цепочку (в рамках одной или нескольких дисциплин)."),
        ("complexity_cluster", "difficulty_label",         "Сложность освоения",
         "Группировка по сложности: Базовый / Средний / Продвинутый."),
    ]

    has_clusters = any(col in df.columns for col, *_ in cluster_defs)

    if not has_clusters:
        st.warning("Кластеры ещё не вычислены. Запустите `python main.py --cluster` в Модуле Б.")
        if st.button("▶ Запустить кластеризацию сейчас"):
            with st.spinner("Кластеризация..."):
                try:
                    from clustering import run_clustering
                    run_clustering()
                    st.cache_data.clear()
                    st.success("Готово! Перезагрузите страницу.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка: {e}")
        return

    # 2-D projection
    coords = None
    try:
        from clustering import compute_2d_projection
        coords = compute_2d_projection(df, method="tsne")
    except Exception:
        pass

    for col, lbl_col, name, description in cluster_defs:
        if col not in df.columns:
            continue

        st.subheader(f"🔹 {name}")
        st.caption(description)

        labels    = df[col].dropna()
        lbl_names = df[lbl_col].fillna(f"Кластер") if lbl_col in df.columns else labels.astype(str)

        # Cluster sizes
        size_df = (
            df.groupby(lbl_col if lbl_col in df.columns else col)
              .size()
              .reset_index(name="Кол-во")
        )
        size_df.columns = ["Кластер", "Кол-во"]

        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(size_df, use_container_width=True)

        with c2:
            # Scatter or bar based on availability
            if coords is not None and getattr(coords, "ndim", 0) == 2 and coords.shape[1] >= 2:
                n_points = min(len(df), len(coords))
                plot_source = df.iloc[:n_points].reset_index(drop=True).copy()
                plot_df = pd.DataFrame({
                    "x": coords[:n_points, 0],
                    "y": coords[:n_points, 1],
                    "Кластер": (
                        lbl_names.iloc[:n_points].reset_index(drop=True)
                        if hasattr(lbl_names, "iloc")
                        else lbl_names[:n_points]
                    ),
                    "topic": plot_source["topic"].fillna("") if "topic" in plot_source.columns else "",
                    "subject": plot_source["subject"].fillna("") if "subject" in plot_source.columns else "",
                })
                if n_points < len(df):
                    st.caption(
                        f"Визуализация построена по {n_points} из {len(df)} материалов: "
                        "для части записей 2-D проекция недоступна."
                    )
                fig = px.scatter(
                    plot_df, x="x", y="y", color="Кластер",
                    hover_data=["topic", "subject"],
                    title=f"t-SNE проекция — {name}",
                )
                fig.update_traces(marker=dict(size=9, opacity=0.8))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.bar(size_df, x="Кластер", y="Кол-во",
                             color="Кластер",
                             title=f"Размеры кластеров — {name}")
                st.plotly_chart(fig, use_container_width=True)

        # Quality metrics for this clustering
        st.markdown("**Метрики качества:**")
        try:
            from quality_metrics import _combined_matrix, compute_metrics
            X = _combined_matrix(df)
            if X is not None:
                raw_labels = df[col].fillna(-1).astype(int).values
                m = compute_metrics(X, raw_labels)
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Силуэт",           f"{m['silhouette']:.4f}"         if not np.isnan(m['silhouette'])         else "—")
                mc2.metric("Дэвис-Болдин",      f"{m['davies_bouldin']:.4f}"     if not np.isnan(m['davies_bouldin'])     else "—")
                mc3.metric("Калинский-Харабаш", f"{m['calinski_harabasz']:.2f}"  if not np.isnan(m['calinski_harabasz'])  else "—")
        except Exception:
            pass

        st.divider()

    # Method comparison
    st.subheader("Сравнение методов кластеризации")
    try:
        from quality_metrics import compare_methods
        cmp_df = compare_methods(df)
        if not cmp_df.empty:
            st.dataframe(cmp_df, use_container_width=True)
            st.markdown(
                "**Вывод:** K-Means выбран для параллельного и сложностного кластеров "
                "(эффективен на TF-IDF пространствах), агломеративная кластеризация — "
                "для последовательного (сохраняет иерархию тем)."
            )
    except Exception as e:
        st.warning(f"Не удалось вычислить сравнение: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Data (admin only)
# ─────────────────────────────────────────────────────────────────────────────

def page_data(df: pd.DataFrame):
    st.title("🗄️ Таблица материалов")

    col1, col2, col3 = st.columns(3)
    with col1:
        subjects = ["Все"] + sorted(df["subject"].dropna().unique().tolist()) \
            if "subject" in df.columns else ["Все"]
        sel_subj = st.selectbox("Предмет", subjects)
    with col2:
        statuses = ["Все"] + sorted(df["moderation_status"].dropna().unique().tolist()) \
            if "moderation_status" in df.columns else ["Все"]
        sel_status = st.selectbox("Статус модерации", statuses)
    with col3:
        gen_opts = {"Все": None, "Исходные": False, "Сгенерированные": True}
        sel_gen  = st.selectbox("Тип", list(gen_opts.keys()))

    filtered = df.copy()
    if sel_subj != "Все" and "subject" in filtered.columns:
        filtered = filtered[filtered["subject"] == sel_subj]
    if sel_status != "Все" and "moderation_status" in filtered.columns:
        filtered = filtered[filtered["moderation_status"] == sel_status]
    if gen_opts[sel_gen] is not None and "is_generated" in filtered.columns:
        filtered = filtered[filtered["is_generated"] == gen_opts[sel_gen]]

    st.caption(f"Отображено: {len(filtered)} из {len(df)} записей")

    display_cols = [c for c in [
        "id", "subject", "topic", "lesson_type", "moderation_status",
        "compliance_score", "is_generated", "is_compliant",
        "has_previous", "has_next", "difficulty_label",
    ] if c in filtered.columns]

    st.dataframe(filtered[display_cols], use_container_width=True)

    if st.checkbox("Показать полный текст выбранного материала"):
        mid = st.number_input("ID материала", min_value=1, step=1)
        row = filtered[filtered["id"] == mid]
        if not row.empty and "text_content" in row.columns:
            st.text_area("Текст", row.iloc[0]["text_content"][:4000], height=300)


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _init_session()

    if not st.session_state["authenticated"]:
        _login_form()
        return

    df      = load_materials()
    comp_df = load_compliance_summary()

    page = _sidebar(df)

    if page == "overview":
        page_overview(df)
    elif page == "analytics" and _can_access("analytics"):
        page_analytics(df, comp_df)
    elif page == "requirements" and _can_access("requirements"):
        page_requirements(comp_df)
    elif page == "clustering" and _can_access("clustering"):
        page_clustering(df)
    elif page == "data" and _can_access("data"):
        page_data(df)
    else:
        st.warning("У вас нет доступа к этому разделу.")


if __name__ == "__main__":
    main()
