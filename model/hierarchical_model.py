
"""Hierarchical classification model"""
from sklearn.ensemble import RandomForestClassifier

class HierarchicalModel:

    def __init__(self):
        self.level1 = RandomForestClassifier()
        self.level2 = RandomForestClassifier()

    def train(self, X, y_level1, y_level2):
        self.level1.fit(X, y_level1)
        self.level2.fit(X, y_level2)

    def predict(self, X):
        l1 = self.level1.predict(X)
        l2 = self.level2.predict(X)
        return l1, l2
