from models.adaboost_model import AdaBoostModel
from models.extra_trees_model import ExtraTreesModel
from models.hist_gb_model import HistGradientBoostingModel
from models.random_forest_model import RandomForestModel
from models.sgd_model import SGDModel
from models.voting_model import VotingModel


MODEL_REGISTRY = {
    "RandomForest": RandomForestModel,
    "HistGradientBoosting": HistGradientBoostingModel,
    "SGD": SGDModel,
    "AdaBoost": AdaBoostModel,
    "Voting": VotingModel,
    "ExtraTrees": ExtraTreesModel,
}

__all__ = ["MODEL_REGISTRY"]
