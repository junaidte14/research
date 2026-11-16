from pydantic import BaseModel, ConfigDict, Field

class CalculatorInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid", 
        strict=True
    )
    
    expression: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "A mathematical expression, for example "
            "(125 * 8) + 450"
        )
    )

class DateTimeInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True
    )
    
class TextAnalyzerInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True
    )

    text: str = Field(
        description="The text to analyze."
    )
