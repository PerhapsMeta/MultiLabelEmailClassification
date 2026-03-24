from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from Config import AppConfig, Config


@dataclass
class LevelResult:
    model_name: str
    level_name: str
    accuracy: float
    chained_prefix_score: float
    macro_f1: float
    weighted_f1: float
    report: str

    def to_console_text(self, config: AppConfig) -> str:
        # //MODIFY: Print both the standard accuracy and the prefix-aware Chained score.
        return "\n".join(
            [
                (
                    f"Summary: level={config.level_display_name(self.level_name)} "
    
                    f"| chained_score={self.chained_prefix_score:.2f}"
                ),
            ]
        )

@dataclass
class ModelResult:
    model_name: str
    level_results: Dict[str, LevelResult]
    chained_score: float


@dataclass
class ResultBundle:
    model_results: List[ModelResult]


def evaluate_predictions(model_name: str, level_name: str, y_true, y_pred) -> LevelResult:
    return LevelResult(
        model_name=model_name,
        level_name=level_name,
        accuracy=accuracy_score(y_true, y_pred),
        chained_prefix_score=0.0,
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        weighted_f1=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        report=classification_report(y_true, y_pred, zero_division=0),
    )


def compute_chained_scores(
    dataset_bundle,
    predictions_by_level: Dict[str, np.ndarray],
) -> tuple[Dict[str, float], float]:
    # MODIFY: Score each level with prefix dependency and average the level scores for the final Chained score.
    # the algorithm of accuracy in document
    shared_mask = np.ones(len(dataset_bundle.test_idx), dtype=bool)
    level_scores: Dict[str, float] = {}

    for level_name in Config.CHAIN_LEVELS:
        if level_name not in predictions_by_level:
            break
        y_true = dataset_bundle.get_level(level_name).y_test
        y_pred = predictions_by_level[level_name]
        shared_mask &= y_true == y_pred
        level_scores[level_name] = float(shared_mask.mean())

    overall_score = float(np.mean(list(level_scores.values()))) if level_scores else 0.0
    return level_scores, overall_score


def export_result_bundle(result_bundle: ResultBundle, export_path: Path, config: AppConfig) -> None:
    rows = []
    for model_result in result_bundle.model_results:
        for level_name, level_result in model_result.level_results.items():
            rows.append(
                {
                    "model_name": model_result.model_name,
                    "level_name": config.level_display_name(level_name),
                    "chained_score": level_result.chained_prefix_score,
                    "model_chained_score": model_result.chained_score,
                }
            )
    pd.DataFrame(rows).to_csv(export_path, index=False)
