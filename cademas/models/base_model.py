from abc import ABC, abstractmethod

class BaseModel(ABC):

    @abstractmethod
    def predict_proba(self, X):
        pass