from sklearn.ensemble import HistGradientBoostingClassifier

from models.base import BaseModel


class HistGradientBoostingModel(BaseModel):
    def __init__(self) -> None:
        super().__init__("HistGradientBoosting")

    def build_estimator(self):
        return HistGradientBoostingClassifier(random_state=0)
