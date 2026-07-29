"""Pydantic contracts shared by constrained local generation backends."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StrictOutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SolverProposal(StrictOutputModel):
    analysis_summary: list[str] = Field(min_length=1)
    python_code: str = Field(min_length=1)


class EvaluatorPostcheck(StrictOutputModel):
    accepted: bool
    verified_answer: str = Field(min_length=1)
    answer_type: str = Field(min_length=1)
    substitution_passed: bool
    difficulty_acceptable: bool
    estimated_difficulty: int = Field(ge=1, le=10)
    reason: str = Field(min_length=1)


class SemanticAnswerJudgment(StrictOutputModel):
    semantically_equivalent: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    format_only_difference: bool


class TestTakerOutput(StrictOutputModel):
    reasoning_summary: list[str] = Field(min_length=1, max_length=8)
    final_answer: str = Field(min_length=1)
    answer_type: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
