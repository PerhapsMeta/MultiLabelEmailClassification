from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from models.base import BaseModel


class VotingModel(BaseModel):
    def __init__(self) -> None:
        super().__init__("Voting")

    def build_estimator(self):
        return VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(random_state=0, max_iter=1000)),
                ("rf", RandomForestClassifier(n_estimators=50, random_state=0)),
                ("gnb", GaussianNB()),
            ],
            voting="hard",
        )
