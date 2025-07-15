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
