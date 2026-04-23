"""FastAPI application for Module G.

Loads pre-trained sklearn models at startup via lifespan.
Validates all input with Pydantic v2.
Never retrains models — only calls .predict() / .transform().
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import DATABASE_URL, LESSON_TYPE_LABELS, METHODOLOGY_REQUIREMENTS, METHODOLOGICAL_GUIDELINES, MODEL_FAST
from .predictor import predictor
from .schemas import (
    AssessRequest, AssessResponse, RequirementResult,
    ClusterRequest, ParallelResponse, ParallelGroup,
    SequentialResponse, SequentialItem,
    ComplexityRequest, ComplexityResponse,
    TimeMaterialRequest, TimeMaterialResponse, TimeBreakdown,
    TimeSubjectRequest, TimeSubjectResponse,
    TimeSubjectsRequest, TimeSubjectsResponse,
    TrajectoryRequest, TrajectoryResponse, TrajectoryStep,
    HealthResponse,
)


# ── DB helpers ────────────────────────────────────────────────────────────────

def _db_conn():
    import psycopg2
    import psycopg2.extras
    return psycopg2.connect(DATABASE_URL)


def _get_material(mid: int) -> Optional[Dict]:
    try:
        conn = _db_conn()
        cur  = conn.cursor(cursor_factory=__import__("psycopg2.extras", fromlist=["RealDictCursor"]).RealDictCursor)
        cur.execute("SELECT * FROM materials WHERE id = %s", (mid,))
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None


def _get_compliance(mid: int) -> List[Dict]:
    try:
        import psycopg2.extras
        conn = _db_conn()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT mc.is_met, mc.score, mc.notes, mr.category, mr.requirement
            FROM methodology_compliance mc
            JOIN methodology_requirements mr ON mr.id = mc.requirement_id
            WHERE mc.material_id = %s ORDER BY mr.category, mr.requirement
        """, (mid,))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def _get_materials_by_subject(subject: str) -> List[Dict]:
    try:
        import psycopg2.extras
        conn = _db_conn()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT m.*,
                   COALESCE(f_wc.feature_value, 500) AS word_count_f,
                   COALESCE(f_hq.feature_value, 0)   AS has_questions_f,
                   COALESCE(f_hv.feature_value, 0)   AS has_videos_f
            FROM materials m
            LEFT JOIN material_features f_wc ON f_wc.material_id = m.id AND f_wc.feature_name = 'word_count'
            LEFT JOIN material_features f_hq ON f_hq.material_id = m.id AND f_hq.feature_name = 'has_questions'
            LEFT JOIN material_features f_hv ON f_hv.material_id = m.id AND f_hv.feature_name = 'has_videos'
            WHERE LOWER(m.subject) = LOWER(%s) AND m.text_content IS NOT NULL
            ORDER BY m.id
        """, (subject,))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


# ── LLM assessment ────────────────────────────────────────────────────────────

def _assess_with_llm(text: str, subject: str, topic: str, lesson_type: str) -> Dict:
    from .llm_client import generate_text, extract_json_object, extract_json_array

    req_list = "\n".join(
        f"{i+1}. [{r['category']}] {r['requirement']}: {r['description']}"
        for i, r in enumerate(METHODOLOGY_REQUIREMENTS)
    )

    prompt_overall = f"""Вы — опытный методист. Оцените учебный материал.

МЕТОДИЧЕСКИЕ РЕКОМЕНДАЦИИ:{METHODOLOGICAL_GUIDELINES}

МАТЕРИАЛ:
Предмет: {subject or 'не указан'}
Тема: {topic or 'не указана'}
Тип занятия: {lesson_type or 'не указан'}
Текст (первые 2500 символов): {text[:2500]}

Верните ТОЛЬКО JSON:
{{
  "compliance_score": <число 0-10>,
  "is_compliant": <true/false>,
  "moderation_status": "<approved|rejected|needs_revision>",
  "moderation_notes": "<общее заключение до 200 символов>",
  "strengths": ["<достоинство1>", "<достоинство2>"],
  "weaknesses": ["<недостаток1>", "<недостаток2>"],
  "recommendations": "<рекомендации до 200 символов>"
}}"""

    overall = extract_json_object(generate_text(prompt_overall, MODEL_FAST))

    prompt_reqs = f"""Оцените учебный материал по каждому требованию.

Предмет: {subject or 'н/д'}, Тема: {topic or 'н/д'}
Текст: {text[:2000]}

ТРЕБОВАНИЯ:
{req_list}

Верните ТОЛЬКО JSON-массив (ровно {len(METHODOLOGY_REQUIREMENTS)} объектов):
[
  {{"index": 0, "is_met": true, "score": 8.5, "notes": "..."}},
  ...
]"""

    req_results = extract_json_array(generate_text(prompt_reqs, MODEL_FAST))

    requirements = []
    for i, req in enumerate(METHODOLOGY_REQUIREMENTS):
        res = next((r for r in req_results if r.get("index") == i), {})
        requirements.append({
            "category":    req["category"],
            "requirement": req["requirement"],
            "is_met":      bool(res.get("is_met", False)),
            "score":       float(res.get("score", 0.0)),
            "notes":       str(res.get("notes", "")),
        })

    return {
        "overall_score":    float(overall.get("compliance_score", 0.0)),
        "is_compliant":     bool(overall.get("is_compliant", False)),
        "moderation_status": str(overall.get("moderation_status", "needs_revision")),
        "moderation_notes": str(overall.get("moderation_notes", "")),
        "strengths":        list(overall.get("strengths", [])),
        "weaknesses":       list(overall.get("weaknesses", [])),
        "recommendations":  str(overall.get("recommendations", "")),
        "requirements":     requirements,
    }


# ── App factory ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    yield


app = FastAPI(
    title="Educational Materials API — Module G",
    version="1.0.0",
    description="API для анализа учебных материалов: оценка, кластеризация, время, траектория",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return {"status": "ok", "models_loaded": predictor.loaded, "version": "1.0.0"}


@app.post("/api/v1/assess", response_model=AssessResponse, tags=["Assessment"])
def assess(req: AssessRequest):
    """Оценить учебный материал по методическим требованиям."""
    subject = req.subject
    topic   = req.topic
    lesson_type = req.lesson_type or "other"
    cached  = False

    if req.material_id:
        mat = _get_material(req.material_id)
        if mat is None:
            raise HTTPException(404, f"Материал {req.material_id} не найден")

        subject     = subject or mat.get("subject")
        topic       = topic   or mat.get("topic")
        lesson_type = lesson_type or mat.get("lesson_type") or "other"

        cached_reqs = _get_compliance(req.material_id)
        if cached_reqs and mat.get("compliance_score") is not None:
            requirements = [
                RequirementResult(
                    category=r["category"], requirement=r["requirement"],
                    is_met=bool(r["is_met"]), score=float(r["score"] or 0),
                    notes=str(r["notes"] or ""),
                )
                for r in cached_reqs
            ]
            return AssessResponse(
                material_id=req.material_id, subject=subject, topic=topic,
                overall_score=float(mat["compliance_score"]),
                is_compliant=bool(mat.get("is_compliant", False)),
                moderation_status=str(mat.get("moderation_status", "unknown")),
                moderation_notes=str(mat.get("moderation_notes", "")),
                strengths=[], weaknesses=[], recommendations="",
                requirements=requirements, cached=True,
            )
        text = mat.get("text_content") or ""
    else:
        text = req.text or ""

    if not text.strip():
        raise HTTPException(400, "Текст материала пуст")

    result = _assess_with_llm(text, subject or "", topic or "", lesson_type)
    requirements = [RequirementResult(**r) for r in result["requirements"]]

    return AssessResponse(
        material_id=req.material_id, subject=subject, topic=topic,
        overall_score=result["overall_score"],
        is_compliant=result["is_compliant"],
        moderation_status=result["moderation_status"],
        moderation_notes=result["moderation_notes"],
        strengths=result["strengths"],
        weaknesses=result["weaknesses"],
        recommendations=result["recommendations"],
        requirements=requirements,
        cached=cached,
    )


@app.post("/api/v1/parallel", response_model=ParallelResponse, tags=["Clustering"])
def parallel(req: ClusterRequest):
    """Определить группы материалов для параллельного изучения."""
    texts  = [m.text for m in req.materials]
    labels = predictor.predict_parallel(texts)

    groups_map: Dict[int, List[int]] = {}
    for i, lbl in enumerate(labels):
        groups_map.setdefault(lbl, []).append(i)

    phase_names = {0: "Тематический поток А", 1: "Тематический поток Б",
                   2: "Тематический поток В", 3: "Тематический поток Г"}

    groups = [
        ParallelGroup(
            group_id=gid,
            label=phase_names.get(gid, f"Группа {gid}"),
            titles=[req.materials[i].title or f"Материал {i+1}" for i in idxs],
        )
        for gid, idxs in sorted(groups_map.items())
    ]

    n_groups = len(groups)
    rec = (
        f"Материалы разделены на {n_groups} независимых групп — "
        "материалы из разных групп можно изучать одновременно."
        if n_groups > 1 else
        "Все материалы образуют одну тематическую группу."
    )
    return ParallelResponse(groups=groups, total_groups=n_groups, recommendation=rec)


@app.post("/api/v1/sequential", response_model=SequentialResponse, tags=["Clustering"])
def sequential(req: ClusterRequest):
    """Определить порядок последовательного изучения материалов."""
    texts  = [m.text for m in req.materials]
    phases = predictor.predict_sequential(texts)

    phase_names = {0: "Ориентация (Знание)", 1: "Понимание (Компрехенсия)",
                   2: "Применение", 3: "Синтез и Оценка"}

    indexed = sorted(enumerate(phases), key=lambda x: x[1])
    order = [
        SequentialItem(
            position=pos + 1,
            phase=phases[orig_i],
            phase_name=phase_names.get(phases[orig_i], f"Фаза {phases[orig_i]}"),
            title=req.materials[orig_i].title or f"Материал {orig_i + 1}",
            subject=req.materials[orig_i].subject,
        )
        for pos, (orig_i, _) in enumerate(indexed)
    ]

    return SequentialResponse(
        order=order,
        explanation=(
            "Материалы упорядочены по фазам когнитивного освоения согласно таксономии Блума: "
            "от базового ознакомления к синтезу и критическому анализу."
        ),
    )


@app.post("/api/v1/complexity", response_model=ComplexityResponse, tags=["Clustering"])
def complexity(req: ComplexityRequest):
    """Определить уровень сложности освоения учебного материала."""
    label, word_count, avg_sent = predictor.predict_complexity_from_text(req.text)

    explanations = {
        "Базовый":     "Материал доступен без специальной подготовки. Небольшой объём, простые предложения.",
        "Средний":     "Материал требует базовой подготовки. Умеренный объём и сложность изложения.",
        "Продвинутый": "Материал предназначен для углублённого изучения. Большой объём, сложный язык.",
    }
    scores = {"Базовый": 0.25, "Средний": 0.55, "Продвинутый": 0.85}

    return ComplexityResponse(
        level=label,
        score=scores.get(label, 0.5),
        word_count=word_count,
        avg_sentence_length=round(avg_sent, 1),
        explanation=explanations.get(label, ""),
    )


@app.post("/api/v1/time/material", response_model=TimeMaterialResponse, tags=["Time"])
def time_material(req: TimeMaterialRequest):
    """Оценить время освоения одного учебного материала."""
    title       = None
    lesson_type = req.lesson_type or "lecture"
    has_q       = req.has_questions
    has_v       = req.has_videos
    diff_label  = req.difficulty_label

    if req.material_id:
        mat = _get_material(req.material_id)
        if mat is None:
            raise HTTPException(404, f"Материал {req.material_id} не найден")
        title       = mat.get("topic")
        lesson_type = mat.get("lesson_type") or lesson_type
        diff_label  = diff_label or mat.get("difficulty_label")
        text        = mat.get("text_content") or ""
        word_count  = len(text.split())
    else:
        word_count = len((req.text or "").split())

    if not diff_label:
        diff_label, _, _ = predictor.predict_complexity_from_text(req.text or "")

    est = predictor.estimate_minutes(word_count, lesson_type, diff_label, has_q, has_v)
    total = est["total"]
    h, m  = divmod(total, 60)

    return TimeMaterialResponse(
        title=title,
        lesson_type=LESSON_TYPE_LABELS.get(lesson_type, lesson_type),
        difficulty_label=diff_label,
        word_count=word_count,
        estimated_minutes=total,
        breakdown=TimeBreakdown(**est["breakdown"]),
        human_readable=f"~{total} мин" if h == 0 else f"~{h} ч {m} мин",
    )


@app.post("/api/v1/time/subject", response_model=TimeSubjectResponse, tags=["Time"])
def time_subject(req: TimeSubjectRequest):
    """Оценить время освоения всех материалов предмета."""
    mats = _get_materials_by_subject(req.subject)
    if not mats:
        raise HTTPException(404, f"Материалы по предмету '{req.subject}' не найдены")

    total = 0
    by_lt: Dict[str, int] = {}
    by_diff: Dict[str, int] = {}

    for mat in mats:
        wc  = int(mat.get("word_count_f") or len((mat.get("text_content") or "").split()))
        lt  = mat.get("lesson_type") or "other"
        dl  = mat.get("difficulty_label") or "Средний"
        hq  = bool(mat.get("has_questions_f"))
        hv  = bool(mat.get("has_videos_f"))
        est = predictor.estimate_minutes(wc, lt, dl, hq, hv)["total"]
        total += est
        by_lt[LESSON_TYPE_LABELS.get(lt, lt)]   = by_lt.get(LESSON_TYPE_LABELS.get(lt, lt), 0) + est
        by_diff[dl] = by_diff.get(dl, 0) + est

    h, m = divmod(total, 60)
    return TimeSubjectResponse(
        subject=req.subject, material_count=len(mats), total_minutes=total,
        human_readable=f"~{total} мин" if h == 0 else f"~{h} ч {m} мин",
        by_lesson_type=by_lt, by_difficulty=by_diff,
    )


@app.post("/api/v1/time/subjects", response_model=TimeSubjectsResponse, tags=["Time"])
def time_subjects(req: TimeSubjectsRequest):
    """Оценить время освоения набора предметов."""
    results = []
    for subj in req.subjects:
        mats = _get_materials_by_subject(subj)
        subj_total = 0
        for mat in mats:
            wc  = int(mat.get("word_count_f") or len((mat.get("text_content") or "").split()))
            lt  = mat.get("lesson_type") or "other"
            dl  = mat.get("difficulty_label") or "Средний"
            hq  = bool(mat.get("has_questions_f"))
            hv  = bool(mat.get("has_videos_f"))
            subj_total += predictor.estimate_minutes(wc, lt, dl, hq, hv)["total"]
        results.append({"subject": subj, "total_minutes": subj_total, "material_count": len(mats)})

    grand_total   = sum(r["total_minutes"] for r in results)
    parallel_max  = max((r["total_minutes"] for r in results), default=0)

    def fmt(m):
        h, mn = divmod(m, 60)
        return f"~{m} мин" if h == 0 else f"~{h} ч {mn} мин"

    return TimeSubjectsResponse(
        subjects=results,
        grand_total_minutes=grand_total,
        parallel_total_minutes=parallel_max,
        human_readable_sequential=fmt(grand_total),
        human_readable_parallel=fmt(parallel_max) + " (при параллельном изучении)",
    )


@app.post("/api/v1/trajectory", response_model=TrajectoryResponse, tags=["Trajectory"])
def trajectory(req: TrajectoryRequest):
    """Построить индивидуальную траекторию изучения материалов."""
    mats = _get_materials_by_subject(req.subject)
    if not mats:
        raise HTTPException(404, f"Материалы по предмету '{req.subject}' не найдены")

    for mat in mats:
        wc = int(mat.get("word_count_f") or len((mat.get("text_content") or "").split()))
        mat["word_count"]   = wc
        mat["has_questions"] = bool(mat.get("has_questions_f"))

    result = predictor.build_trajectory(
        materials=mats,
        difficulty_level=req.difficulty_level,
        available_hours_per_week=req.available_hours_per_week,
        learning_style=req.learning_style,
        target_topics=req.target_topics,
        exclude_ids=req.exclude_material_ids,
    )

    steps = [
        TrajectoryStep(step=i + 1, **{k: v for k, v in s.items() if k != "phase" and k != "phase_name"})
        for i, s in enumerate(result["trajectory"])
    ]

    return TrajectoryResponse(
        trajectory=steps,
        total_steps=result["total_steps"],
        total_estimated_minutes=result["total_estimated_minutes"],
        total_weeks=result["total_weeks"],
        weekly_plan={k: v for k, v in result["weekly_plan"].items()},
        parameters_used={
            "subject":                  req.subject,
            "difficulty_level":         req.difficulty_level,
            "available_hours_per_week": req.available_hours_per_week,
            "learning_style":           req.learning_style,
        },
    )
