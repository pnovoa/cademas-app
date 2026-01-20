import json
from cademas.validation.schemas import ContextDefinition
from cademas.context.evaluate import evaluate_context

if __name__ == "__main__":
    employee = {
    "JobLevel": 4,
    "TotalWorkingYears": 7,
    "YearsSinceLastPromotion": 5
    }

    with open("../configs/context_example.json") as f:
        context = ContextDefinition(**json.load(f))

    score = evaluate_context(employee, context)
    print("Context score:", score)