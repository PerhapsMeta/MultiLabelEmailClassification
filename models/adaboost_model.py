from sklearn.ensemble import AdaBoostClassifier

from models.base import BaseModel


class AdaBoostModel(BaseModel):
    def __init__(self) -> None:
        super().__init__("AdaBoost")

    def build_estimator(self):
        return AdaBoostClassifier(n_estimators=100, random_state=0)
