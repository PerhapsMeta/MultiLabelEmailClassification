from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


BASE_DIR = Path(__file__).resolve().parent


class Config:
    #  Use canonical column names so preprocessing and modelling share one schema.
    TICKET_ID = "ticket_id"
    INTERACTION_ID = "interaction_id"
    INTERACTION_DATE = "interaction_date"
    MAILBOX = "mailbox"
    TICKET_SUMMARY = "ticket_summary"
    INTERACTION_CONTENT = "interaction_content"
    #  Define type column names as constants to avoid hardcoding in the pipeline and ensure consistency.
    TEXT = "text"
    TYPE1 = "type1"
    TYPE2 = "type2"
    TYPE3 = "type3"
    TYPE4 = "type4"

    #  Define label names for each classification level to avoid hardcoding in the pipeline.
    LABEL_L2 = "label_l2"
    LABEL_L3 = "label_l3"
    LABEL_L4 = "label_l4"
    CHAIN_LEVELS = (LABEL_L2, LABEL_L3, LABEL_L4)


@dataclass(frozen=True)
class AppConfig:
    # Centralize pipeline settings so the controller only orchestrates the workflow.
    base_dir: Path = BASE_DIR
    raw_data_dir: Path = BASE_DIR / "data"
    #  Define input files as a tuple to ensure immutability and clear expected inputs.
    input_files: Tuple[str, ...] = ("AppGallery.csv", "Purchasing.csv")
    cleaned_export_path: Path = BASE_DIR / "cleaned_tickets.csv"
    results_export_path: Path = BASE_DIR / "results_summary.csv"
    #  Define constants for preprocessing and modelling to avoid magic numbers and strings in the code.
    none_label: str = "NONE"
    chain_separator: str = "__"
    min_samples_per_class: int = 3
    test_size: float = 0.2
    random_state: int = 0
    max_features: int = 2000
    min_df: int = 4
    max_df: float = 0.9
    #  List enabled models in a tuple to ensure immutability and clear model selection.
    enabled_models: Tuple[str, ...] = (
        "RandomForest",
        "HistGradientBoosting",
        "SGD",
        "AdaBoost",
        "Voting",
        "ExtraTrees",
    )
    column_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "Ticket id": Config.TICKET_ID,
            "Interaction id": Config.INTERACTION_ID,
            "Interaction date": Config.INTERACTION_DATE,
            "Mailbox": Config.MAILBOX,
            "Ticket Summary": Config.TICKET_SUMMARY,
            "Interaction content": Config.INTERACTION_CONTENT,
            "Type 1": Config.TYPE1,
            "Type 2": Config.TYPE2,
            "Type 3": Config.TYPE3,
            "Type 4": Config.TYPE4,
        }
    )

    def level_display_name(self, level_name: str) -> str:
        level_names = {
            Config.LABEL_L2: "Level 1 (Type 2)",
            Config.LABEL_L3: "Level 2 (Type 2 + Type 3)",
            Config.LABEL_L4: "Level 3 (Type 2 + Type 3 + Type 4)",
        }
        return level_names[level_name]
