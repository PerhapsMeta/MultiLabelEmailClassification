from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from Config import AppConfig, Config
from data.dataset import PreparedData


def load_raw_data(config: AppConfig) -> pd.DataFrame:
    frames = []
    for file_name in config.input_files:
        file_path = config.raw_data_dir / file_name
        frames.append(pd.read_csv(file_path, skipinitialspace=True))
    raw_df = pd.concat(frames, ignore_index=True)
    return raw_df


def standardize_columns(raw_df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    standardized_df = raw_df.rename(columns=config.column_mapping).copy()
    required_columns = [
        Config.TICKET_ID,
        Config.TICKET_SUMMARY,
        Config.INTERACTION_CONTENT,
        Config.TYPE1,
        Config.TYPE2,
        Config.TYPE3,
        Config.TYPE4,
    ]
    standardized_df = standardized_df[required_columns]

    for text_column in [Config.TICKET_SUMMARY, Config.INTERACTION_CONTENT]:
        standardized_df[text_column] = standardized_df[text_column].fillna("").astype(str)

    for label_column in [Config.TYPE1, Config.TYPE2, Config.TYPE3, Config.TYPE4]:
        standardized_df[label_column] = standardized_df[label_column].fillna("").astype(str).str.strip()

    standardized_df = standardized_df.loc[standardized_df[Config.TYPE2] != ""].reset_index(drop=True)
    return standardized_df


def _build_customer_template_pattern() -> str:
    customer_templates = {
        "english": [
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team\,?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is a company incorporated under the laws of Ireland with its headquarters in Dublin, Ireland\.?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is the provider of Huawei Mobile Services to Huawei and Honor device owners in (?:Europe|\*\*\*\*\*\(LOC\)), Canada, Australia, New Zealand and other countries\.?",
        ],
        "german": [
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Kundenservice\,?",
            r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist eine Gesellschaft nach irischem Recht mit Sitz in Dublin, Irland\.?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist der Anbieter von Huawei Mobile Services fur Huawei- und Honor-Geratebesitzer in Europa, Kanada, Australien, Neuseeland und anderen Landern\.?",
        ],
        "french": [
            r"L'equipe d'assistance a la clientele d'Aspiegel\,?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE est une societe de droit irlandais dont le siege est a Dublin, en Irlande\.?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE est le fournisseur de services mobiles Huawei aux proprietaires d'appareils Huawei et Honor en Europe, au Canada, en Australie, en Nouvelle-Zelande et dans d'autres pays\.?",
        ],
        "spanish": [
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Soporte Servicio al Cliente\,?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) es una sociedad constituida en virtud de la legislacion de Irlanda con su sede en Dublin, Irlanda\.?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE es el proveedor de servicios moviles de Huawei a los propietarios de dispositivos de Huawei y Honor en Europa, Canada, Australia, Nueva Zelanda y otros paises\.?",
        ],
        "italian": [
            r"Il tuo team ad (?:Aspiegel|\*\*\*\*\*\(PERSON\))\,?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE e una societa costituita secondo le leggi irlandesi con sede a Dublino, Irlanda\.?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE e il fornitore di servizi mobili Huawei per i proprietari di dispositivi Huawei e Honor in Europa, Canada, Australia, Nuova Zelanda e altri paesi\.?",
        ],
        "portuguese": [
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team\,?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE e uma empresa constituida segundo as leis da Irlanda, com sede em Dublin, Irlanda\.?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE e o provedor de Huawei Mobile Services para Huawei e Honor proprietarios de dispositivos na Europa, Canada, Australia, Nova Zelandia e outros paises\.?",
        ],
    }
    return "|".join(f"({pattern})" for pattern in sum(customer_templates.values(), []))


def deduplicate_interaction_content(df: pd.DataFrame) -> pd.DataFrame:
    # MODIFY: Remove repeated email fragments at the ticket level before feature generation.
    deduplicated_df = df.copy()
    deduplicated_df["deduplicated_content"] = ""

    customer_pattern = _build_customer_template_pattern()
    split_pattern = "|".join(
        [
            r"(From\s?:\s?xxxxx@xxxx.com Sent\s?:.{30,70}Subject\s?:)",
            r"(On.{30,60}wrote:)",
            r"(Re\s?:|RE\s?:)",
            r"(\*\*\*\*\*\(PERSON\) Support issue submit)",
            r"(\s?\*\*\*\*\*\(PHONE\))*$",
        ]
    )

    for ticket_id, ticket_rows in deduplicated_df.groupby(Config.TICKET_ID):
        seen_fragments: set[str] = set()
        ticket_output = []

        for content in ticket_rows[Config.INTERACTION_CONTENT].tolist():
            split_content = re.split(split_pattern, content)
            split_content = [fragment for fragment in split_content if fragment is not None]
            split_content = [re.sub(split_pattern, "", fragment.strip()) for fragment in split_content]
            split_content = [re.sub(customer_pattern, "", fragment.strip()) for fragment in split_content]

            current_fragments = []
            for fragment in split_content:
                if fragment and fragment not in seen_fragments:
                    seen_fragments.add(fragment)
                    current_fragments.append(fragment)

            ticket_output.append(" ".join(current_fragments))

        deduplicated_df.loc[
            deduplicated_df[Config.TICKET_ID] == ticket_id,
            "deduplicated_content",
        ] = ticket_output

    deduplicated_df[Config.INTERACTION_CONTENT] = deduplicated_df["deduplicated_content"]
    deduplicated_df = deduplicated_df.drop(columns=["deduplicated_content"])
    return deduplicated_df


def remove_text_noise(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()
    summary_noise = (
        r"(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|"
        r"(\[|\])|(aspiegel support issue submit)|(null)|(nan)|"
        r"((bonus place my )?support.pt 自动回复:)"
    )

    cleaned_df[Config.TICKET_SUMMARY] = (
        cleaned_df[Config.TICKET_SUMMARY]
        .str.lower()
        .replace(summary_noise, " ", regex=True)
        .replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    cleaned_df[Config.INTERACTION_CONTENT] = cleaned_df[Config.INTERACTION_CONTENT].str.lower()
    interaction_noise = [
        r"(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)",
        r"(january|february|march|april|may|june|july|august|september|october|november|december)",
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        r"\d{2}(:|.)\d{2}",
        r"(xxxxx@xxxx\.com)|(\*{5}\([a-z]+\))",
        r"dear ((customer)|(user))",
        r"dear",
        r"(hello)|(hallo)|(hi )|(hi there)",
        r"good morning",
        r"thank you for your patience ((during (our)? investigation)|(and cooperation))?",
        r"thank you for contacting us",
        r"thank you for your availability",
        r"thank you for providing us this information",
        r"thank you for contacting",
        r"thank you for reaching us (back)?",
        r"thank you for patience",
        r"thank you for (your)? reply",
        r"thank you for (your)? response",
        r"thank you for (your)? cooperation",
        r"thank you for providing us with more information",
        r"thank you very kindly",
        r"thank you( very much)?",
        r"i would like to follow up on the case you raised on the date",
        r"i will do my very best to assist you",
        r"in order to give you the best solution",
        r"could you please clarify your request with following information:",
        r"in this matter",
        r"we hope you(( are)|('re)) doing ((fine)|(well))",
        r"i would like to follow up on the case you raised on",
        r"we apologize for the inconvenience",
        r"sent from my huawei (cell )?phone",
        r"original message",
        r"customer support team",
        r"(aspiegel )?se is a company incorporated under the laws of ireland with its headquarters in dublin, ireland.",
        r"(aspiegel )?se is the provider of huawei mobile services to huawei and honor device owners in",
        r"canada, australia, new zealand and other countries",
        r"\d+",
        r"[^0-9a-zA-Z]+",
        r"(\s|^).(\s|$)",
    ]

    for pattern in interaction_noise:
        cleaned_df[Config.INTERACTION_CONTENT] = cleaned_df[Config.INTERACTION_CONTENT].replace(
            pattern,
            " ",
            regex=True,
        )

    cleaned_df[Config.INTERACTION_CONTENT] = (
        cleaned_df[Config.INTERACTION_CONTENT].replace(r"\s+", " ", regex=True).str.strip()
    )
    return cleaned_df


def normalize_label_value(value: str, config: AppConfig) -> str:
    normalized_value = "" if pd.isna(value) else str(value).strip()
    return normalized_value or config.none_label


def build_chain_labels(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    # //MODIFY: Build Chained target labels as chained combinations to preserve label dependencies.
    label_df = df[[Config.TYPE2, Config.TYPE3, Config.TYPE4]].copy()
    label_df[Config.TYPE2] = label_df[Config.TYPE2].map(lambda value: normalize_label_value(value, config))
    label_df[Config.TYPE3] = label_df[Config.TYPE3].map(lambda value: normalize_label_value(value, config))
    label_df[Config.TYPE4] = label_df[Config.TYPE4].map(lambda value: normalize_label_value(value, config))

    label_df[Config.LABEL_L2] = label_df[Config.TYPE2]
    label_df[Config.LABEL_L3] = (
        label_df[Config.TYPE2] + config.chain_separator + label_df[Config.TYPE3]
    )
    label_df[Config.LABEL_L4] = (
        label_df[Config.TYPE2]
        + config.chain_separator
        + label_df[Config.TYPE3]
        + config.chain_separator
        + label_df[Config.TYPE4]
    )
    return label_df[[Config.LABEL_L2, Config.LABEL_L3, Config.LABEL_L4]]


def export_cleaned_data(clean_df: pd.DataFrame, label_df: pd.DataFrame, export_path: Path) -> None:
    export_df = clean_df.copy()
    for label_column in Config.CHAIN_LEVELS:
        export_df[label_column] = label_df[label_column]
    export_df.to_csv(export_path, index=False)


def prepare_data(config: AppConfig) -> PreparedData:
    raw_df = load_raw_data(config)
    clean_df = standardize_columns(raw_df, config)
    clean_df = deduplicate_interaction_content(clean_df)
    clean_df = remove_text_noise(clean_df)
    clean_df[Config.TEXT] = (
        clean_df[Config.TICKET_SUMMARY].fillna("")
        + " "
        + clean_df[Config.INTERACTION_CONTENT].fillna("")
    ).str.strip()

    label_df = build_chain_labels(clean_df, config)
    export_cleaned_data(clean_df, label_df, config.cleaned_export_path)
    return PreparedData(
        raw_df=raw_df,
        clean_df=clean_df,
        text_series=clean_df[Config.TEXT],
        label_df=label_df,
    )
