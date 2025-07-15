from typing import List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class EmployeeInfo(BaseModel):
    employee_name: str
    title: str = ""
    department: str = ""
    company_name: str = ""
    additional_context: str = ""


class EmailResult(BaseModel):
    email: str
    confidence: float = Field(ge=0.0, le=1.0)
    pattern_type: str = ""  # "public", "pattern", "guess"
    source: str = ""
    domain: str = ""
    pattern_frequency: int = 0  # How often this pattern appears

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if "@" not in v:
            raise ValueError("Invalid email format")
        return v.lower().strip()


class EmailPattern(BaseModel):
    pattern: str  # e.g., "first.last", "first_last", "flast"
    frequency: int = 0
    confidence: float = Field(ge=0.0, le=1.0)
    examples: List[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    title: str = ""
    link: str
    snippet: str = ""


class LLMResponse(BaseModel):
    success: bool = True
    data: Dict[str, Any] | None = None
    error: str | None = None
    provider: str
