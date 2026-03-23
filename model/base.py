from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


def format_metrics_report(y_true, y_pred) -> str:
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)

    return "\n".join(
        [
            "Intro: precision = predicted labels that were correct, recall = true labels that were found, f1 = balance between precision and recall, support = number of test samples.",
            f"Summary: accuracy={accuracy:.2f} | macro_f1={macro_f1:.2f} | weighted_f1={weighted_f1:.2f}",
            report,
        ]
    )


class BaseModel(ABC):
    def __init__(self) -> None:
        ...


    @abstractmethod
    def train(self) -> None:
        """
        Train the model using ML Models for Multi-class and mult-label classification.
        :params: df is essential, others are model specific
        :return: classifier
        """
        ...

    @abstractmethod
    def predict(self) -> int:
        """

        """
        ...

    #
    @abstractmethod
    def data_transform(self) -> None:
        return

    # def build(self, values) -> BaseModel:
    def build(self, values={}):
        values = values if isinstance(values, dict) else utils.string2any(values)
        self.__dict__.update(self.defaults)
        self.__dict__.update(values)
        return self
