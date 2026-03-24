from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.estimator = self.build_estimator()

    @abstractmethod
    def build_estimator(self) -> Any:
        raise NotImplementedError

    def train(self, dataset) -> None:
        self.estimator = self.build_estimator()
        self.estimator.fit(dataset.X_train, dataset.y_train)

    def predict(self, X_test):
        return self.estimator.predict(X_test)

    def print_results(self, level_title: str, report_text: str) -> None:
        print(f"\n--- {self.model_name} | {level_title} ---")
        print(report_text)
