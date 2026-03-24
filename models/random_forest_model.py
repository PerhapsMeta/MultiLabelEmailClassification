from sklearn.ensemble import RandomForestClassifier

from models.base import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self) -> None:
        super().__init__("RandomForest")

    def build_estimator(self):
        # MODIFY: Rebuild a fresh estimator for each chained task to keep model runs independent.
        return RandomForestClassifier(
            n_estimators=1000,
            random_state=0,
            class_weight="balanced_subsample",
        )
