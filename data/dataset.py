from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from Config import AppConfig, Config


@dataclass
class PreparedData:
    raw_df: pd.DataFrame
    clean_df: pd.DataFrame
    text_series: pd.Series
    label_df: pd.DataFrame


@dataclass
class SplitDataset:
    level_name: str
    X_train: Any
    X_test: Any
    y_train: np.ndarray
    y_test: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    classes: np.ndarray


@dataclass
class DatasetBundle:
    filtered_df: pd.DataFrame
    label_df: pd.DataFrame
    vectorizer: TfidfVectorizer
    levels: Dict[str, SplitDataset]
    train_idx: np.ndarray
    test_idx: np.ndarray

    def get_level(self, level_name: str) -> SplitDataset:
        return self.levels[level_name]


def _filter_small_classes(prepared_data: PreparedData, config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_level_counts = prepared_data.label_df[Config.LABEL_L4].value_counts()
    valid_labels = final_level_counts[final_level_counts >= config.min_samples_per_class].index
    keep_mask = prepared_data.label_df[Config.LABEL_L4].isin(valid_labels)

    filtered_df = prepared_data.clean_df.loc[keep_mask].reset_index(drop=True)
    filtered_labels = prepared_data.label_df.loc[keep_mask].reset_index(drop=True)

    if filtered_df.empty:
        raise ValueError("No rows remain after filtering small Level 3 classes.")

    if filtered_labels[Config.LABEL_L4].nunique() < 2:
        raise ValueError("At least two final chained classes are required for modelling.")

    return filtered_df, filtered_labels


def _resolve_test_size(label_df: pd.DataFrame, config: AppConfig) -> float:
    class_count = label_df[Config.LABEL_L4].nunique()
    min_ratio = class_count / len(label_df)
    test_size = max(config.test_size, min_ratio)
    if test_size >= 1.0:
        raise ValueError("The computed test size is invalid for the available data.")
    return test_size


def build_dataset_bundle(prepared_data: PreparedData, config: AppConfig) -> DatasetBundle:
    # MODIFY: Keep one shared split for all chained targets so multi-level evaluation stays aligned.
    filtered_df, filtered_labels = _filter_small_classes(prepared_data, config)
    indices = np.arange(len(filtered_df))
    test_size = _resolve_test_size(filtered_labels, config)

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=config.random_state,
        stratify=filtered_labels[Config.LABEL_L4],
    )

    train_text = filtered_df.iloc[train_idx][Config.TEXT]
    test_text = filtered_df.iloc[test_idx][Config.TEXT]

    vectorizer = TfidfVectorizer(
        max_features=config.max_features,
        min_df=config.min_df,
        max_df=config.max_df,
    )

    # MODIFY: Convert TF-IDF features to dense arrays so all retained sklearn models can share the same input format.
    X_train = vectorizer.fit_transform(train_text).toarray()
    X_test = vectorizer.transform(test_text).toarray()

    levels: Dict[str, SplitDataset] = {}
    for level_name in Config.CHAIN_LEVELS:
        y_train = filtered_labels.iloc[train_idx][level_name].to_numpy()
        y_test = filtered_labels.iloc[test_idx][level_name].to_numpy()
        levels[level_name] = SplitDataset(
            level_name=level_name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            train_idx=train_idx,
            test_idx=test_idx,
            classes=np.sort(filtered_labels[level_name].unique()),
        )

    return DatasetBundle(
        filtered_df=filtered_df,
        label_df=filtered_labels,
        vectorizer=vectorizer,
        levels=levels,
        train_idx=train_idx,
        test_idx=test_idx,
    )
