from cademas.context.memberships import triangular, trapezoidal


def evaluate_context(employee, context_def):
    values = []

    for factor in context_def.factors:
        x = employee[factor.variable]
        mf = factor.membership

        if mf.type == "triangular":
            mu = triangular(x, *mf.parameters)
        elif mf.type == "trapezoidal":
            mu = trapezoidal(x, *mf.parameters)
        else:
            raise ValueError("Unknown membership function")

        values.append(mu)

    if context_def.aggregation_operator == "min":
        return min(values)
    elif context_def.aggregation_operator == "product":
        result = 1.0
        for v in values:
            result *= v
        return result
    return None