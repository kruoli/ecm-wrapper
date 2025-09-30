from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

class EffortLevel(BaseModel):
    b1: int
    curves: int

class ECMWorkSummary(BaseModel):
    total_attempts: int
    total_curves: int
    effort_by_level: List[EffortLevel]
    last_attempt: Optional[datetime]

class CompositeStats(BaseModel):
    composite: str = Field(..., description="The composite number (original form)")
    current_composite: str = Field(..., description="Current composite being factored")
    digit_length: int = Field(..., description="Decimal digit length")
    has_snfs_form: bool = Field(..., description="Whether number has SNFS polynomial form")
    snfs_difficulty: Optional[int] = Field(None, description="GNFS-equivalent digit count for SNFS numbers")
    target_t_level: Optional[float] = Field(..., description="Target t-level")
    current_t_level: Optional[float] = Field(..., description="Current t-level achieved")
    priority: int = Field(..., description="Priority level")
    status: Literal["composite", "prime", "fully_factored"] = Field(..., description="Current status")
    factors_found: List[str] = Field(default_factory=list, description="Known factors")
    ecm_work: ECMWorkSummary = Field(..., description="Summary of ECM work done")
    projects: List[str] = Field(default_factory=list, description="Associated projects")

class CompositeResponse(BaseModel):
    id: int
    number: str
    current_composite: str
    digit_length: int
    has_snfs_form: bool
    snfs_difficulty: Optional[int]
    target_t_level: Optional[float]
    current_t_level: Optional[float]
    priority: int
    is_prime: Optional[bool]
    is_fully_factored: bool
    created_at: datetime
    updated_at: datetime