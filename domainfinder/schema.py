from typing import List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class CompanyInfo(BaseModel):
    company_name: str
    website: str = ""
    likely_email_domains: List[str] = Field(default_factory=list)
    description: str = ""
    sub_mail_domains: List[str] = Field(default_factory=list)

    @field_validator("likely_email_domains")
    @classmethod
    def validate_domains(cls, v):
        return [domain.strip().lower() for domain in v if domain.strip()]


class DomainResult(BaseModel):
    domain: str
    email_count: int = 0
    source_count: int = 0
    confidence: float = Field(ge=0.0, le=1.0)
    from_llm: bool = False
    mx_valid: bool = False
    sub_mail_domains: List[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    title: str = ""
    link: str
    snippet: str = ""


class LLMResponse(BaseModel):
    success: bool = True
    data: Dict[str, Any] | None = None
    error: str | None = None
    provider: str
