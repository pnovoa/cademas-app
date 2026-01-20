from cademas.core.aggregator import aggregate_risk
from cademas.context.evaluate import evaluate_context


def prioritization_index(
    case: dict,
    adms: list,
    context_def,
    lambda_policy: float
) -> dict:
    """
    Computes cooperative prioritization index π(e_i)
    """
    if not 0.0 <= lambda_policy <= 1.0:
        raise ValueError("lambda_policy must be in [0,1]")

    R = aggregate_risk(case, adms)
    C = evaluate_context(case, context_def)

    pi = lambda_policy * R + (1 - lambda_policy) * C

    return {
        "risk": R,
        "context": C,
        "priority": pi
    }