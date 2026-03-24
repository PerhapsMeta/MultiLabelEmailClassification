from __future__ import annotations

import numpy as np

from Config import AppConfig, Config
from models import MODEL_REGISTRY
from modelling.results import (
    ModelResult,
    ResultBundle,
    compute_chained_scores,
    evaluate_predictions,
    export_result_bundle,
)


class ModelRegistry:
    def create(self, model_name: str):
        if model_name not in MODEL_REGISTRY:
            raise KeyError(f"Unsupported model: {model_name}")
        return MODEL_REGISTRY[model_name]()


class ModelRunner:
    def __init__(self, config: AppConfig, registry: ModelRegistry | None = None) -> None:
        self.config = config
        self.registry = registry or ModelRegistry()

    def _print_model_header(self, model_name: str) -> None:
        print("\n" + "=" * 80)
        print(f"Model: {model_name}")
        print("=" * 80)

    def run(self, dataset_bundle) -> ResultBundle:
        # //MODIFY: Run each model across the chained targets and score it with prefix-aware Chained rules.
        model_results = []
        for model_name in self.config.enabled_models:
            self._print_model_header(model_name)
            level_results = {}
            predictions_by_level = {}
            models_by_level = {}

            for level_name in Config.CHAIN_LEVELS:
                dataset = dataset_bundle.get_level(level_name)
                if np.unique(dataset.y_train).size < 2:
                    print(
                        f"\n--- {model_name} | {self.config.level_display_name(level_name)} ---"
                    )
                    print("Skipped: this level does not have enough distinct classes for training.")
                    continue

                model = self.registry.create(model_name)
                model.train(dataset)
                y_pred = model.predict(dataset.X_test)
                models_by_level[level_name] = model
                predictions_by_level[level_name] = y_pred

                level_result = evaluate_predictions(model_name, level_name, dataset.y_test, y_pred)
                level_results[level_name] = level_result

            chained_level_scores, chained_score = compute_chained_scores(
                dataset_bundle,
                predictions_by_level,
            )

            for level_name in Config.CHAIN_LEVELS:
                if level_name not in level_results:
                    continue
                level_results[level_name].chained_prefix_score = chained_level_scores.get(
                    level_name,
                    0.0,
                )
                models_by_level[level_name].print_results(
                    self.config.level_display_name(level_name),
                    level_results[level_name].to_console_text(self.config),
                )

            print(f"\nChained score (prefix-based): {chained_score:.2f}")
            model_results.append(
                ModelResult(
                    model_name=model_name,
                    level_results=level_results,
                    chained_score=chained_score,
                )
            )

        result_bundle = ResultBundle(model_results=model_results)
        export_result_bundle(result_bundle, self.config.results_export_path, self.config)
        return result_bundle
