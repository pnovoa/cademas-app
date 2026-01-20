from pydantic import BaseModel, Field, model_validator
from typing import List, Literal, Optional


class ADMMetadata(BaseModel):
    """
    Metadata describing an Automated Decision Component (ADM).
    """

    adm_id: str = Field(..., description="Unique identifier of the ADM")

    department: str = Field(
        ..., description="Organizational unit represented by this ADM"
    )

    model_type: Literal["h2o_mojo", "sklearn"] = Field(
        ..., description="Underlying model technology"
    )

    target_class: str = Field(
        ..., description="Class of interest (e.g., 'Attrition=Yes')"
    )

    features: List[str] = Field(
        ..., description="List of feature names consumed by the ADM"
    )

    performance: dict = Field(
        ..., description="Performance metrics (e.g., AUC, accuracy)"
    )

    description: Optional[str] = Field(
        None, description="Human-readable description of the ADM"
    )


class MembershipFunction(BaseModel):
    type: Literal["triangular", "trapezoidal"]
    parameters: List[float]

    @model_validator(mode="after")
    def check_parameters(self):
        if self.type == "triangular" and len(self.parameters) != 3:
            raise ValueError("Triangular membership requires 3 parameters (a, b, c)")
        if self.type == "trapezoidal" and len(self.parameters) != 4:
            raise ValueError("Trapezoidal membership requires 4 parameters (a, b, c, d)")
        return self


class ContextFactor(BaseModel):
    name: str
    variable: str
    membership: MembershipFunction


class ContextDefinition(BaseModel):
    context_id: str
    aggregation_operator: Literal["min", "product"]
    factors: List[ContextFactor]