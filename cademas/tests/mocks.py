from cademas.models.base_model import BaseModel

class DummyModel(BaseModel):
    def __init__(self, value: float):
        self.value = value

    def predict_proba(self, X):
        return self.value