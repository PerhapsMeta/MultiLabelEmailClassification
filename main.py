import sys

# MODIFY: Disable Python bytecode cache generation before importing project modules.
sys.dont_write_bytecode = True

from Config import AppConfig, Config
from data import build_dataset_bundle
from modelling import ModelRunner
from preprocessing import prepare_data


def print_pipeline_summary(prepared_data, dataset_bundle, config: AppConfig) -> None:
    # MODIFY: Summarize the shared chained dataset once because Type 1 grouping is intentionally removed.
    print("\n" + "=" * 80)
    print("Chained Multi-Label Classification Pipeline")
    print("The pipeline cleans the ticket text, builds chained labels, creates one shared split, and evaluates each model across all hierarchy levels.")
    print(f"Raw samples: {len(prepared_data.raw_df)}")
    print(f"Prepared samples with Type 2 labels: {len(prepared_data.clean_df)}")
    print(f"Samples kept after Level 3 class filtering: {len(dataset_bundle.filtered_df)}")
    print(f"Train samples: {len(dataset_bundle.train_idx)} | Test samples: {len(dataset_bundle.test_idx)}")
    for level_name in Config.CHAIN_LEVELS:
        class_count = dataset_bundle.label_df[level_name].nunique()
        print(f"{config.level_display_name(level_name)} classes: {class_count}")
    print("=" * 80)


def main() -> None:
    config = AppConfig()
    prepared_data = prepare_data(config)
    dataset_bundle = build_dataset_bundle(prepared_data, config)
    print_pipeline_summary(prepared_data, dataset_bundle, config)
    runner = ModelRunner(config)
    runner.run(dataset_bundle)


if __name__ == "__main__":
    main()
