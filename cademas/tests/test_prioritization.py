import pytest

from cademas.validation.schemas import ADMMetadata
from cademas.core.decision_component import DecisionComponent
from cademas.core.aggregator import aggregate_risk
from cademas.context.evaluate import evaluate_context
from cademas.core.priorization import prioritization_index

from cademas.tests.mocks import DummyModel


@pytest.fixture
def decision_components():
    adm1 = ADMMetadata(
        adm_id="HR_ADM",
        department="Human Resources",
        model_type="sklearn",
        target_class="Attrition=Yes",
        features=["Age", "YearsAtCompany"],
        performance={"auc": 0.78},
    )

    adm2 = ADMMetadata(
        adm_id="FIN_ADM",
        department="Finance",
        model_type="sklearn",
        target_class="Attrition=Yes",
        features=["MonthlyIncome"],
        performance={"auc": 0.74},
    )

    dc1 = DecisionComponent(
        metadata=adm1,
        model=DummyModel(0.8),   # riesgo alto
        weight=0.6,
    )

    dc2 = DecisionComponent(
        metadata=adm2,
        model=DummyModel(0.4),   # riesgo medio
        weight=0.4,
    )

    return [dc1, dc2]


@pytest.fixture
def employee():
    return {
        "Age": 45,
        "YearsAtCompany": 10,
        "MonthlyIncome": 2500,
    }

@pytest.fixture
def context_value():
    return 0.7


def test_prioritization_index(decision_components, employee, context_value):
    risk = aggregate_risk(employee, decision_components)

    # 0.6*0.8 + 0.4*0.4 = 0.64
    assert abs(risk - 0.64) < 1e-6

    lambda_policy = 0.7

    pi = lambda_policy * risk + (1 - lambda_policy) * context_value

    assert 0.0 <= pi <= 1.0

def test_full_prioritization_pipeline(
    decision_components, case, context_definition
):
    pi = prioritization_index(
        case=case,
        adms=decision_components,
        context_def=context_definition,
        lambda_policy=0.7
    )

    assert isinstance(pi, float)
    assert 0.0 <= pi <= 1.0