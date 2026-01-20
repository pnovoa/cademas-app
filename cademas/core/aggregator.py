from typing import List
from cademas.core.decision_component import DecisionComponent


def aggregate_risk(case: dict, components: list[DecisionComponent]) -> float:
    return sum(
        dc.weight * dc.predict(case)
        for dc in components
    )