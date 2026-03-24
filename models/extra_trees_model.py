from sklearn.ensemble import ExtraTreesClassifier

from models.base import BaseModel


class ExtraTreesModel(BaseModel):
    def __init__(self) -> None:
        super().__init__("ExtraTrees")

    def build_estimator(self):
        return ExtraTreesClassifier(
            n_estimators=100,
            min_samples_leaf=10,
            random_state=0,
        )
