from cademas.validation.schemas import ADMMetadata
from cademas.models.base_model import BaseModel

class DecisionComponent:
    """
    Executable ADM: metadata + predictive model
    """

    def __init__(self, metadata: ADMMetadata, model: BaseModel, weight: float):
        self.metadata = metadata
        self.model = model
        self.weight = weight

    def predict(self, employee_features: dict) -> float:
        """
        Returns P(target_class | employee)
        """
        X = {f: employee_features[f] for f in self.metadata.features}
        proba = self.model.predict_proba(X)

        if not 0.0 <= proba <= 1.0:
            raise ValueError("Invalid probability output")

        return proba