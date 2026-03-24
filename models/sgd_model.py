from sklearn.linear_model import SGDClassifier

from models.base import BaseModel


class SGDModel(BaseModel):
    def __init__(self) -> None:
        super().__init__("SGD")

    def build_estimator(self):
        return SGDClassifier(
            random_state=0,
            class_weight="balanced",
            max_iter=1000,
            tol=1e-3,
        )
