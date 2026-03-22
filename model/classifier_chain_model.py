
"""Classifier Chain model for multi-label classification"""
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression

class ClassifierChainModel:

    def __init__(self):
        base = LogisticRegression(max_iter=1000)
        self.model = ClassifierChain(base)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
