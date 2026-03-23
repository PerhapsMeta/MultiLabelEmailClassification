from model.SGD import SGD
from model.randomforest import RandomForest
from model.adaboost import AdaBoost
from model.voting import Voting
from model.hist_gb import Hist_GB
from model.random_trees_ensembling import RandomTreesEmbedding


def print_model_header(model_name: str) -> None:
    print(f"\n--- {model_name} ---")


def model_predict(data, df, name):
    results = []
    print_model_header("RandomForest")
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)

    print_model_header("Hist_GB")
    model = Hist_GB("Hist_GB", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    res = model.print_results(data)
    results.append(res)

    print_model_header("SGD")
    model = SGD("SGD", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)

    print_model_header("AdaBoost")
    model = AdaBoost("AdaBoost", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)

    print_model_header("Voting")
    model = Voting("Voting", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)

    print_model_header("RandomTreesEmbedding")
    model = RandomTreesEmbedding("RandomTreesEmbedding", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)


def model_evaluate(model, data):
    model.print_results(data)
