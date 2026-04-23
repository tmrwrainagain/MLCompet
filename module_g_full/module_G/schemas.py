"""Pydantic v2 request/response schemas for Module G API."""
from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator


# ── /assess ──────────────────────────────────────────────────────────────────

class AssessRequest(BaseModel):
    material_id: Optional[int] = Field(None, gt=0, description="ID материала в БД")
    text: Optional[str] = Field(None, min_length=10, description="Текст материала напрямую")
    subject: Optional[str] = None
    topic: Optional[str] = None
    lesson_type: Optional[Literal["lecture","seminar","practice","lab","self_study","test","other"]] = None

    @model_validator(mode="after")
    def check_source(self) -> "AssessRequest":
        if self.material_id is None and self.text is None:
            raise ValueError("Укажите material_id или text")
        return self


class RequirementResult(BaseModel):
    category: str
    requirement: str
    is_met: bool
    score: float
    notes: str


class AssessResponse(BaseModel):
    material_id: Optional[int]
    subject: Optional[str]
    topic: Optional[str]
    overall_score: float
    is_compliant: bool
    moderation_status: str
    moderation_notes: str
    strengths: list[str]
    weaknesses: list[str]
    recommendations: str
    requirements: list[RequirementResult]
    cached: bool


# ── /parallel, /sequential ───────────────────────────────────────────────────

class MaterialText(BaseModel):
    text: str = Field(..., min_length=5)
    title: Optional[str] = None
    subject: Optional[str] = None


class ClusterRequest(BaseModel):
    materials: list[MaterialText] = Field(..., min_length=2, description="Минимум 2 материала")


class ParallelGroup(BaseModel):
    group_id: int
    label: str
    titles: list[str]


class ParallelResponse(BaseModel):
    groups: list[ParallelGroup]
    total_groups: int
    recommendation: str


class SequentialItem(BaseModel):
    position: int
    phase: int
    phase_name: str
    title: str
    subject: Optional[str]


class SequentialResponse(BaseModel):
    order: list[SequentialItem]
    explanation: str


# ── /complexity ───────────────────────────────────────────────────────────────

class ComplexityRequest(BaseModel):
    text: str = Field(..., min_length=10)
    title: Optional[str] = None


class ComplexityResponse(BaseModel):
    level: str
    score: float
    word_count: int
    avg_sentence_length: float
    explanation: str


# ── /time/* ───────────────────────────────────────────────────────────────────

class TimeMaterialRequest(BaseModel):
    material_id: Optional[int] = Field(None, gt=0)
    text: Optional[str] = Field(None, min_length=5)
    lesson_type: Optional[str] = "lecture"
    difficulty_label: Optional[Literal["Базовый","Средний","Продвинутый"]] = None
    has_questions: bool = False
    has_videos: bool = False

    @model_validator(mode="after")
    def check_source(self) -> "TimeMaterialRequest":
        if self.material_id is None and self.text is None:
            raise ValueError("Укажите material_id или text")
        return self


class TimeBreakdown(BaseModel):
    reading_minutes: float
    complexity_overhead_minutes: float
    practice_minutes: float
    video_minutes: float


class TimeMaterialResponse(BaseModel):
    title: Optional[str]
    lesson_type: str
    difficulty_label: str
    word_count: int
    estimated_minutes: int
    breakdown: TimeBreakdown
    human_readable: str


class TimeSubjectRequest(BaseModel):
    subject: str = Field(..., min_length=1)


class TimeSubjectResponse(BaseModel):
    subject: str
    material_count: int
    total_minutes: int
    human_readable: str
    by_lesson_type: dict[str, int]
    by_difficulty: dict[str, int]


class TimeSubjectsRequest(BaseModel):
    subjects: list[str] = Field(..., min_length=1)
    parallel_study: bool = False


class TimeSubjectsResponse(BaseModel):
    subjects: list[dict[str, Any]]
    grand_total_minutes: int
    parallel_total_minutes: int
    human_readable_sequential: str
    human_readable_parallel: str


# ── /trajectory ───────────────────────────────────────────────────────────────

class TrajectoryRequest(BaseModel):
    subject: str
    difficulty_level: Literal["Базовый","Средний","Продвинутый"]
    available_hours_per_week: float = Field(..., gt=0, le=80)
    learning_style: Literal["sequential","mixed"] = "sequential"
    exclude_material_ids: list[int] = []
    target_topics: list[str] = []


class TrajectoryStep(BaseModel):
    step: int
    material_id: int
    topic: Optional[str]
    difficulty_label: Optional[str]
    estimated_minutes: int
    week: int
    rationale: str


class TrajectoryResponse(BaseModel):
    trajectory: list[TrajectoryStep]
    total_steps: int
    total_estimated_minutes: int
    total_weeks: int
    weekly_plan: dict[str, list[int]]
    parameters_used: dict[str, Any]


# ── /health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str = "1.0.0"
