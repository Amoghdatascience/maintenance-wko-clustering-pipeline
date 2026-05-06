"""
Aviation Maintenance Workorder Clustering Pipeline

This module converts raw aviation maintenance workorders into structured
cluster labels using a hybrid pipeline that combines:

1. Rule-based classification (seed rules + strict token rules)
2. Machine learning (Logistic Regression with TF-IDF features)
3. Semantic similarity (cosine similarity against cluster descriptions)
4. Hierarchical prediction (Parent -> Child clusters)

Outputs
-------
- datasets/dataset.csv: final clustered workorder records

Author: Amogh Naik
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import hstack, spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"

INPUT_FILE = DATASET_DIR / "Valid_Problems_Workorder.xlsx"
OUTPUT_DATASET = DATASET_DIR / "dataset.csv"
RANDOM_STATE = 42

# Abbreviation expansion is essential because aviation maintenance text
# contains a large amount of shorthand, abbreviations, and compressed terms.
# Expanding these terms improves:
# - rule-based matching precision
# - TF-IDF feature quality
# - semantic similarity consistency
abbr_map: Dict[str, str] = {
    r"\ba/c\b": "aircraft",
    r"\bacft\b": "aircraft",
    r"\bmx\b": "maintenance",
    r"\binsp\b": "inspection",
    r"\biaw\b": "in accordance with",
    r"\bw/o\b": "work order",
    r"\bwo\b": "work order",
    r"\bpd\b": "problem document",
    r"\boil pres\b": "oil pressure",
    r"\boil prs\b": "oil pressure",
    r"\bspk\b": "spark",
    r"\bplg\b": "plug",
    r"\bbaf\b": "baffle",
    r"\brgh\b": "rough",
    r"\bexh\b": "exhaust",
    r"\bad\b": "airworthiness directive",
    r"\bsb\b": "service bulletin",
    r"\bmel\b": "minimum equipment list",
    r"\bpoh\b": "pilot operating handbook",
    r"\bw&b\b": "weight and balance",
    r"\bl/h\b": "left hand",
    r"\br/h\b": "right hand",
    r"\bo/b\b": "outboard",
    r"\bi/b\b": "inboard",
    r"\bfwd\b": "forward",
    r"\baft\b": "aft",
    r"\bext\b": "external",
    r"\bint\b": "internal",
    r"\bvert\b": "vertical",
    r"\bhorz\b": "horizontal",
    r"\bmlg\b": "main landing gear",
    r"\bnlg\b": "nose landing gear",
    r"\bbrake assy\b": "brake assembly",
    r"\bcaliper\b": "brake caliper",
    r"\bcotterpin\b": "cotter pin",
    r"\bshimmy dampner\b": "shimmy damper",
    r"\beng\b": "engine",
    r"\bprop\b": "propeller",
    r"\bcyl\b": "cylinder",
    r"\bcyls\b": "cylinders",
    r"\bmag\b": "magneto",
    r"\bmags\b": "magnetos",
    r"\brpm\b": "revolutions per minute",
    r"\bpsi\b": "pressure",
    r"\boil psi\b": "oil pressure",
    r"\bcht\b": "cylinder head temperature",
    r"\begt\b": "exhaust gas temperature",
    r"\balt\b": "alternator",
    r"\bbatt\b": "battery",
    r"\bstby\b": "standby",
    r"\bfuel inj\b": "fuel injector",
    r"\binj\b": "injector",
    r"\bavgas\b": "aviation gasoline",
    r"\b100ll\b": "100 low lead aviation gasoline",
    r"\belt\b": "emergency locator transmitter",
    r"\bstab\b": "stabilizer",
    r"\bstatic wick\b": "static wick",
    r"\bbaffle\b": "baffle",
    r"\bcowl\b": "cowling",
    r"\bptt\b": "push to talk",
    r"\bxpdr\b": "transponder",
    r"\bahrs\b": "attitude and heading reference system",
    r"\bvor\b": "very high frequency omnidirectional range",
    r"\boat\b": "outside air temperature",
    r"\bmfd\b": "multi function display",
    r"\bgps\b": "global positioning system",
    r"\bco det\b": "carbon monoxide detector",
    r"\bfod\b": "foreign object debris",
    r"\brivnut\b": "rivet nut",
    r"\bnutplate\b": "nut plate",
    r"\b\vfeb": "maximum flap extended speed",
    r"\bgpu\b": "ground power unit",
}

# Child-cluster descriptions are used for:
# 1. human-readable cluster naming
# 2. semantic similarity fallback when ML confidence is weak
cluster_desc: Dict[str, str] = {
    "c_0": "inspection routine scheduled inspection",
    "c_1": "aircraft start issue hard start no start starter no crank",
    "c_2": "baffle bolt",
    "c_3": "baffle bracket",
    "c_4": "baffle crack damage loose missing",
    "c_5": "baffle mount",
    "c_6": "baffle plug",
    "c_7": "baffle rivet",
    "c_8": "baffle screw",
    "c_9": "baffle seal",
    "c_10": "baffle spring",
    "c_11": "baffle tie rod",
    "c_12": "cowling cowl damage loose",
    "c_13": "cylinder compression low compression",
    "c_14": "cylinder crack failure",
    "c_15": "exhaust valve stuck valve",
    "c_16": "cylinder head temperature",
    "c_17": "pushrod tube",
    "c_18": "drain line tube",
    "c_19": "carburetor carb",
    "c_20": "crankcase crankshaft firewall",
    "c_21": "engine failure fire power loss engine quit",
    "c_22": "engine idle rpm issue",
    "c_23": "engine repair reinstall clean remove replace engine",
    "c_24": "engine run rough rough running misfire",
    "c_25": "engine seal tube bolt loose",
    "c_26": "propeller overspeed",
    "c_27": "induction leak induction system",
    "c_28": "intake gasket",
    "c_29": "intake tube boot seal",
    "c_30": "magneto ignition mag drop",
    "c_31": "mixture adjust mixture control",
    "c_32": "oil cooler",
    "c_33": "oil dipstick tube filler tube",
    "c_34": "oil leak oil pressure oil temperature",
    "c_35": "oil return line",
    "c_36": "pilot reported in flight noticed",
    "c_37": "rocker cover valve cover",
    "c_38": "sniffler valve",
    "c_39": "spark plug plug fouled",
    "c_40": "battery issue low voltage weak battery charging issue",
    "c_41": "external start issue external power ground power unit start",
    "c_42": "flight control aileron elevator rudder flap spoiler trim",
    "c_43": (
        "appearance cleaning paint wash dirty exterior clean fuselage "
        "surface finish cosmetic exterior surface"
    ),
    "c_44": "landing gear tire",
    "c_45": "exhaust gas temperature",
    "c_46": "propeller damage",
}

cluster_name_map: Dict[str, str] = cluster_desc.copy()

# Parent descriptions support the hierarchical stage by first predicting
# broader systems before predicting specific child clusters.
parent_desc: Dict[str, str] = {
    "inspection": "inspection routine scheduled inspection check",
    "start_system": (
        "aircraft start starter hard start no start no crank "
        "external start ground power"
    ),
    "baffle": "baffle bolt bracket crack mount plug rivet screw seal spring tie rod",
    "cowling": "cowling cowl damage loose missing",
    "cylinder_Exhaust": (
        "cylinder compression crack exhaust valve temperature " "push rod pushrod tube"
    ),
    "engine_general": (
        "engine rough idle repair crankcase carburetor " "engine failure power loss"
    ),
    "propeller": "propeller overspeed propeller damage",
    "induction_intake": "induction intake gasket boot seal leak",
    "ignition": "magneto ignition spark plug mag drop fouled plug",
    "fuel_control": "mixture mixture control fuel adjust",
    "oil_system": (
        "oil cooler dipstick filler tube oil leak oil pressure "
        "oil temperature return line"
    ),
    "pilot_reported": "pilot reported in flight pilot noticed",
    "valve_cover": "rocker cover valve cover sniffler valve",
    "electrical": "battery voltage charging alternator electrical",
    "appearance_cleaning": (
        "paint wash dirty exterior clean fuselage surface finish "
        "cosmetic exterior appearance cleaning"
    ),
    "landing_gear_tire": "landing gear tire",
}

cluster_to_parent: Dict[str, str] = {
    "c_0": "inspection",
    "c_1": "start_system",
    "c_41": "start_system",
    "c_2": "baffle",
    "c_3": "baffle",
    "c_4": "baffle",
    "c_5": "baffle",
    "c_6": "baffle",
    "c_7": "baffle",
    "c_8": "baffle",
    "c_9": "baffle",
    "c_10": "baffle",
    "c_11": "baffle",
    "c_12": "cowling",
    "c_13": "cylinder_Exhaust",
    "c_14": "cylinder_Exhaust",
    "c_15": "cylinder_Exhaust",
    "c_16": "cylinder_Exhaust",
    "c_17": "cylinder_Exhaust",
    "c_18": "engine_general",
    "c_19": "engine_general",
    "c_20": "engine_general",
    "c_21": "engine_general",
    "c_22": "engine_general",
    "c_23": "engine_general",
    "c_24": "engine_general",
    "c_25": "engine_general",
    "c_26": "propeller",
    "c_27": "induction_intake",
    "c_28": "induction_intake",
    "c_29": "induction_intake",
    "c_30": "ignition",
    "c_39": "ignition",
    "c_31": "fuel_control",
    "c_32": "oil_system",
    "c_33": "oil_system",
    "c_34": "oil_system",
    "c_35": "oil_system",
    "c_36": "pilot_reported",
    "c_37": "valve_cover",
    "c_38": "valve_cover",
    "c_40": "electrical",
    "c_42": "flight_control",
    "c_43": "appearance_cleaning",
    "c_44": "landing_gear_tire",
    "c_45": "cylinder_Exhaust",
    "c_46": "propeller",
}


def build_parent_to_children_map(
    cluster_parent_map: dict[str, str],
) -> dict[str, list[str]]:
    """
    Build a reverse mapping from parent cluster to child clusters.

    Purpose in pipeline:
    The hierarchical child-classification stage operates parent by parent.
    This helper creates the structure required to retrieve all child clusters
    that belong to a predicted parent system.

    Parameters:
    cluster_parent_map (dict[str, str]): Mapping from child cluster ID to parent cluster.

    Returns:
    dict[str, list[str]]: Mapping from parent cluster to the list of child clusters.
    """
    parent_to_children: Dict[str, List[str]] = {}
    for child, parent in cluster_parent_map.items():
        parent_to_children.setdefault(parent, []).append(child)
    return parent_to_children


PARENT_TO_CHILDREN: Dict[str, List[str]] = build_parent_to_children_map(
    cluster_to_parent
)

# Seed rules intentionally prioritize precision over recall.
# They provide high-confidence weak labels that bootstrap the ML stages.
seed_rules: Dict[str, List[str]] = {
    "c_41": [
        "external start",
        "external power",
        "ground power",
        "ground power unit",
        "start cart",
    ],
    "c_1": [
        "aircraft start",
        "starter issue",
        "hard start",
        "no start",
        "would not start",
        "starting issue",
        "no crank",
    ],
    "c_12": ["cowling", "cowl"],
    "c_13": ["compression", "low compression"],
    "c_14": ["cylinder crack", "cylinder failure", "cracked cylinder"],
    "c_15": ["exhaust valve", "stuck valve"],
    "c_16": ["cylinder head temperature", "cht"],
    "c_17": ["push rod", "pushrod", "push tube"],
    "c_18": ["drain line", "drain tube"],
    "c_19": ["carburetor", "carb"],
    "c_20": ["crankcase", "crankshaft", "firewall"],
    "c_21": ["engine failure", "engine fire", "power loss", "engine quit"],
    "c_22": ["idle", "rpm issue", "low rpm", "high rpm"],
    "c_23": [
        "engine repair",
        "reinstall engine",
        "clean engine",
        "remove replace engine",
    ],
    "c_24": ["engine rough", "run rough", "rough running", "misfire", "stumble"],
    "c_25": ["engine seal", "engine tube", "engine bolt"],
    "c_26": ["propeller overspeed"],
    "c_27": ["induction", "induction leak", "induction system"],
    "c_28": ["intake gasket"],
    "c_29": ["intake tube", "intake boot", "intake seal"],
    "c_30": ["magneto", "mag drop", "ignition"],
    "c_31": ["mixture", "mixture control", "adjust mixture"],
    "c_32": ["oil cooler"],
    "c_33": ["dipstick", "filler tube", "oil tube"],
    "c_34": ["oil leak", "oil pressure", "oil temperature"],
    "c_35": ["oil return line", "oil return"],
    "c_36": ["pilot reported", "in flight", "pilot noticed"],
    "c_37": ["rocker cover", "valve cover"],
    "c_38": ["sniffler valve", "sniffler"],
    "c_39": ["spark plug", "plug fouled", "fouled plug"],
    "c_40": [
        "battery",
        "dead battery",
        "low battery",
        "battery low",
        "low voltage",
        "battery voltage",
        "weak battery",
        "discharged battery",
        "charging issue",
        "would not crank",
    ],
    "c_0": [
        "inspection due",
        "insp due",
        "annual inspection",
        "100 hour inspection",
        "100hr inspection",
        "biennial insp due",
        "static system altimeter biennial",
        "next due",
        "c w ad",
        "no defects noted",
        "no defects found",
    ],
    "c_42": ["aileron", "elevator", "rudder", "flap", "spoiler", "trim", "strut"],
    "c_43": ["paint", "wash", "dirty exterior", "cosmetic"],
    "c_45": ["exhaust gas temperature", "egt"],
    "c_46": ["prop damage", "propeller damage", "damaged propeller"],
}

# Baffle records often contain overlapping terms. These stricter token rules
# require all tokens to be present, which reduces ambiguous matches.
baffle_and_rules: Dict[str, List[List[str]]] = {
    "c_2": [["baffle", "bolt"]],
    "c_3": [["baffle", "bracket"]],
    "c_5": [["baffle", "mount"]],
    "c_6": [["baffle", "plug"]],
    "c_7": [["baffle", "rivet"]],
    "c_8": [["baffle", "screw"]],
    "c_9": [["baffle", "seal"]],
    "c_10": [["baffle", "spring"]],
    "c_11": [["baffle", "tie", "rod"], ["baffle", "tie rod"]],
    "c_4": [
        ["baffle", "crack"],
        ["baffle", "loose"],
        ["baffle", "damage"],
        ["baffle", "missing"],
    ],
}


def normalize_series(series: pd.Series) -> pd.Series:
    """
    Normalize maintenance text for downstream rule-based and ML processing.

    Purpose in pipeline:
    This function standardizes problem and action text so that rule matching,
    TF-IDF feature extraction, and semantic similarity all operate on the same
    cleaned representation.

    Key design decisions:
    - Lowercasing removes case sensitivity.
    - Abbreviation expansion improves both keyword matching and sparse text features.
    - Punctuation removal and whitespace normalization produce a stable token space.

    Parameters:
    series (pd.Series): Raw text series to normalize.

    Returns:
    pd.Series: Normalized text series.
    """
    normalized = series.fillna("").astype(str).str.lower()
    normalized = normalized.str.replace(r"[/\-]", " ", regex=True)

    # Expand domain abbreviations before removing punctuation so that
    # aviation shorthand contributes meaningful tokens.
    for pattern, replacement in abbr_map.items():
        normalized = normalized.str.replace(pattern, replacement, regex=True)

    normalized = normalized.str.replace(r"[^a-z0-9\s]", " ", regex=True)
    normalized = normalized.str.replace(r"\s+", " ", regex=True).str.strip()
    return normalized


def load_input_data(path: str | Path) -> pd.DataFrame:
    """
    Load the input workbook and construct normalized text fields.

    Purpose in pipeline:
    This function prepares the shared text representation used by both
    rule-based and ML stages.

    Parameters:
    path (str | Path): Path to the Excel input file.

    Returns:
    pd.DataFrame: Input data with normalized problem, action, and combined text.
    """
    print("Loading dataset")
    df = pd.read_excel(path)
    print(f"Total records: {len(df)}")

    df["problem_norm"] = normalize_series(df["PROBLEM"])
    df["action_norm"] = normalize_series(df["ACTION"])
    df["text_norm"] = (df["problem_norm"] + " " + df["action_norm"]).str.strip()

    print("Text normalization complete")
    return df


def contains_all_tokens(text_series: pd.Series, tokens: list[str]) -> pd.Series:
    """
    Check whether each text row contains all required tokens.

    Purpose in pipeline:
    This helper supports strict rule matching where multiple tokens must
    co-occur before a label is assigned.

    Parameters:
    text_series (pd.Series): Normalized text to inspect.
    tokens (list[str]): Tokens that must all be present.

    Returns:
    pd.Series: Boolean mask identifying rows containing all tokens.
    """
    mask = pd.Series(True, index=text_series.index)
    for token in tokens:
        mask &= text_series.str.contains(
            rf"\b{re.escape(token)}\b", regex=True, na=False
        )
    return mask


def apply_seed_rules(
    text_series: pd.Series,
    rule_dict: dict[str, list[str]],
    and_rule_dict: Optional[dict[str, list[list[str]]]] = None,
) -> pd.Series:
    """
    Apply high-precision seed rules to generate initial cluster labels.

    Purpose in pipeline:
    This is the weak-supervision stage. It produces reliable seed labels
    that are later used to train the ML classifiers.

    Key design decisions:
    - "First matching rule wins" prevents later rules from overwriting an
      earlier high-confidence assignment.
    - Strict AND-token rules are evaluated before simple keyword rules because
      they are more precise.

    Assumptions:
    - Input text has already been normalized.
    - Seed rules prioritize precision over recall.

    Parameters:
    text_series (pd.Series): Normalized maintenance text.
    rule_dict (dict[str, list[str]]): Simple keyword-based rules.
    and_rule_dict (Optional[dict[str, list[list[str]]]]): Strict multi-token rules.

    Returns:
    pd.Series: Initial cluster labels, with "UNKNOWN" for unmatched rows.
    """
    labels = pd.Series("UNKNOWN", index=text_series.index, dtype=object)

    if and_rule_dict is not None:
        for label, token_groups in and_rule_dict.items():
            for tokens in token_groups:
                mask = contains_all_tokens(text_series, tokens)
                assign_mask = mask & (labels == "UNKNOWN")
                labels.loc[assign_mask] = label

    for label, patterns in rule_dict.items():
        for pattern in patterns:
            mask = text_series.str.contains(re.escape(pattern), regex=True, na=False)
            assign_mask = mask & (labels == "UNKNOWN")
            labels.loc[assign_mask] = label

    return labels


def get_all_matches(
    text_series: pd.Series,
    rule_dict: dict[str, list[str]],
    and_rule_dict: Optional[dict[str, list[list[str]]]] = None,
) -> dict[int, list[str]]:
    """
    Collect all rule matches for each record for diagnostic analysis.

    Purpose in pipeline:
    This function is used to inspect ambiguity in the rule-based stage by
    determining whether records matched one, multiple, or no rules.

    Parameters:
    text_series (pd.Series): Normalized maintenance text.
    rule_dict (dict[str, list[str]]): Simple keyword-based rules.
    and_rule_dict (Optional[dict[str, list[list[str]]]]): Strict multi-token rules.

    Returns:
    dict[int, list[str]]: Mapping from row index to all matched labels.
    """
    matches: Dict[int, List[str]] = {idx: [] for idx in text_series.index}

    if and_rule_dict:
        for label, token_groups in and_rule_dict.items():
            for tokens in token_groups:
                mask = pd.Series(True, index=text_series.index)
                for token in tokens:
                    mask &= text_series.str.contains(
                        rf"\b{re.escape(token)}\b", regex=True, na=False
                    )
                for idx in text_series[mask].index:
                    matches[idx].append(label)

    for label, patterns in rule_dict.items():
        for pattern in patterns:
            mask = text_series.str.contains(re.escape(pattern), regex=True, na=False)
            for idx in text_series[mask].index:
                matches[idx].append(label)

    return matches


def apply_temperature_split_override(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split temperature-related records into separate CHT and EGT clusters.

    Purpose in pipeline:
    Some records mention only cylinder head temperature, while others mention
    only exhaust gas temperature. This override makes that split explicit so
    EGT-only records are not absorbed into c_16.

    Key design decisions:
    - CHT-only records are assigned to c_16.
    - EGT-only records are assigned to c_45.
    - Records mentioning both are left unchanged here and can follow the
      existing pipeline behavior.

    Parameters:
    df (pd.DataFrame): Dataset containing text_norm and seed_cluster columns.

    Returns:
    pd.DataFrame: Dataset with temperature-specific seed overrides applied.
    """
    df = df.copy()

    cht_mask = df["text_norm"].str.contains(
        r"\bcylinder head temperature\b",
        regex=True,
        na=False,
    )
    egt_mask = df["text_norm"].str.contains(
        r"\bexhaust gas temperature\b",
        regex=True,
        na=False,
    )

    # Assign only pure CHT mentions to c_16.
    df.loc[
        cht_mask & ~egt_mask & (df["seed_cluster"] == "UNKNOWN"),
        "seed_cluster",
    ] = "c_16"

    # Assign only pure EGT mentions to c_45.
    df.loc[
        egt_mask & ~cht_mask & (df["seed_cluster"] == "UNKNOWN"),
        "seed_cluster",
    ] = "c_45"

    return df


def apply_prop_split_override(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split propeller-related records into separate overspeed and damage clusters.

    Purpose in pipeline:
    Propeller overspeed and propeller damage describe different fault types.
    This override separates pure damage cases from overspeed cases before the
    rest of the pipeline continues.

    Key design decisions:
    - Overspeed-only records are assigned to c_26.
    - Damage-only records are assigned to c_46.
    - Records mentioning both are left unchanged here and can follow the
      existing pipeline behavior.

    Parameters:
    df (pd.DataFrame): Dataset containing text_norm and seed_cluster columns.

    Returns:
    pd.DataFrame: Dataset with propeller-specific seed overrides applied.
    """
    df = df.copy()

    overspeed_mask = df["text_norm"].str.contains(
        r"\bpropeller overspeed\b|\boverspeed\b",
        regex=True,
        na=False,
    )
    damage_mask = df["text_norm"].str.contains(
        r"\bprop damage\b|\bpropeller damage\b|\bdamaged propeller\b",
        regex=True,
        na=False,
    )

    # Assign only pure overspeed mentions to c_26.
    df.loc[
        overspeed_mask & ~damage_mask & (df["seed_cluster"] == "UNKNOWN"),
        "seed_cluster",
    ] = "c_26"

    # Assign only pure propeller damage mentions to c_46.
    df.loc[
        damage_mask & ~overspeed_mask & (df["seed_cluster"] == "UNKNOWN"),
        "seed_cluster",
    ] = "c_46"

    return df


def classify_c0_type(text: str) -> str:
    """
    Classify inspection-related text into grounded c_0 subtypes.

    Purpose in pipeline:
    The inspection cluster is prone to false positives because words such as
    "inspection" can appear in non-inspection contexts. This function separates
    true inspection records from likely misfits.

    Key design decisions:
    - True inspection patterns focus on due/compliance/admin phrasing.
    - Panel and repair language are treated as likely misfits.

    Parameters:
    text (str): Normalized combined maintenance text.

    Returns:
    str: Inspection subtype label.
    """
    text = str(text)

    if re.search(
        (
            r"\b("
            r"inspection due|insp due|annual inspection|100 hour inspection|100hr inspection|"
            r"biennial insp due|completed inspection|next due|"
            r"no defects found|no defects noted|"
            r"c w ad|airworthiness directive|service bulletin|"
            r"static system.*altimeter.*biennial|altimeter.*biennial|"
            r"inspection completed"
            r")\b"
        ),
        text,
    ):
        return "true_inspection"

    if re.search(r"\binsp time\b|\binspector time\b", text):
        return "inspection_admin"

    if re.search(r"\binspection panel\b", text):
        return "inspection_panel_misfit"

    if re.search(
        (
            r"\b("
            r"tire|tube|leak|failed|friction test|missing|screw|tighten|"
            r"replaced|replace|installed|install|removed|remove|"
            r"worn|crack|cracked|loose|damaged|damage|flat"
            r")\b"
        ),
        text,
    ):
        return "misfit_recluster"

    return "review_needed"


def apply_inspection_override(df: pd.DataFrame) -> pd.DataFrame:
    """
    Refine inspection assignments after seed labeling.

    Purpose in pipeline:
    This override keeps true inspection/admin records in c_0 and sends likely
    inspection misfits back to UNKNOWN so they can be reconsidered by the
    hierarchical classification stages.

    Parameters:
    df (pd.DataFrame): Dataset containing text_norm and seed_cluster columns.

    Returns:
    pd.DataFrame: Dataset with corrected inspection-related seed labels.
    """
    df = df.copy()

    inspection_related = df["text_norm"].str.contains(
        (
            r"\binspection\b|\binsp\b|\bannual\b|"
            r"\bbiennial\b|\bnext due\b|\bad\b|\bservice bulletin\b"
        ),
        regex=True,
        na=False,
    )

    inspection_type = df.loc[inspection_related, "text_norm"].apply(classify_c0_type)

    true_idx = inspection_type[
        inspection_type.isin(["true_inspection", "inspection_admin"])
    ].index
    misfit_idx = inspection_type[
        inspection_type.isin(["inspection_panel_misfit", "misfit_recluster"])
    ].index

    df.loc[true_idx, "seed_cluster"] = "c_0"
    df.loc[
        (df.index.isin(misfit_idx)) & (df["seed_cluster"] == "c_0"),
        "seed_cluster",
    ] = "UNKNOWN"

    return df


def apply_seed_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deterministic overrides after seed-rule assignment.

    Purpose in pipeline:
    This stage enforces domain rules for patterns that should be handled
    directly rather than relying on ML.

    Parameters:
    df (pd.DataFrame): Dataset containing text_norm and seed_cluster columns.

    Returns:
    pd.DataFrame: Dataset with override-adjusted seed labels.
    """
    df = df.copy()

    flight_control_mask = df["text_norm"].str.contains(
        r"\b(?:aileron|elevator|rudder|flap|spoiler|trim)\b",
        regex=True,
        na=False,
    )
    df.loc[flight_control_mask, "seed_cluster"] = "c_42"

    oil_cooler_dirty = df["text_norm"].str.contains(
        r"\boil cooler\b", regex=True, na=False
    ) & df["text_norm"].str.contains(r"\bdirty\b", regex=True, na=False)
    df.loc[oil_cooler_dirty, "seed_cluster"] = "c_32"

    engine_dirty = df["text_norm"].str.contains(
        r"\bengine\b", regex=True, na=False
    ) & df["text_norm"].str.contains(r"\bdirty\b", regex=True, na=False)
    filter_dirty = df["text_norm"].str.contains(
        r"\bfilter\b", regex=True, na=False
    ) & df["text_norm"].str.contains(r"\bdirty\b", regex=True, na=False)
    dirty_at_end = df["text_norm"].str.contains(r"\bdirty\s*$", regex=True, na=False)

    # Appearance-related dirty labels should not override explicit engine/filter signals.
    appearance_dirty = dirty_at_end & ~oil_cooler_dirty & ~engine_dirty & ~filter_dirty
    df.loc[appearance_dirty, "seed_cluster"] = "c_43"

    return df


def analyze_seed_assignment(all_matches: dict[int, list[str]], df: pd.DataFrame) -> int:
    """
    Print diagnostics for the seed-labeling stage.

    Purpose in pipeline:
    This function summarizes rule coverage and ambiguity before ML is applied.

    Parameters:
    all_matches (dict[int, list[str]]): All matched seed labels per record.
    df (pd.DataFrame): Dataset containing seed_cluster assignments.

    Returns:
    int: Number of UNKNOWN records after the seed stage.
    """
    single_group = {k: v for k, v in all_matches.items() if len(v) == 1}
    multi_group = {k: v for k, v in all_matches.items() if len(v) > 1}
    no_group = {k: v for k, v in all_matches.items() if len(v) == 0}

    print("\nCLUSTER ASSIGNMENT ANALYSIS")
    print(f"Total records: {len(df)}")
    print(f"Exactly ONE group: {len(single_group)}")
    print(f"Multiple groups: {len(multi_group)}")
    print(f"No group: {len(no_group)}")

    before_unknown = int((df["seed_cluster"] == "UNKNOWN").sum())
    print(f"UNKNOWN after seed rules: {before_unknown}")
    return before_unknown


def build_features(corpus: pd.Series) -> tuple[TfidfVectorizer, TfidfVectorizer]:
    """
    Fit word-level and character-level TF-IDF vectorizers.

    Purpose in pipeline:
    TF-IDF converts normalized maintenance text into sparse numerical features
    suitable for Logistic Regression and semantic comparison.

    Key design decisions:
    - Word n-grams capture phrase-level meaning.
    - Character n-grams capture spelling variation, abbreviations, and
      fragmented maintenance shorthand.

    Parameters:
    corpus (pd.Series): Text corpus used to fit the vectorizers.

    Returns:
    tuple[TfidfVectorizer, TfidfVectorizer]: Fitted word and character vectorizers.
    """
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=40000)
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), max_features=20000
    )

    word_vec.fit(corpus)
    char_vec.fit(corpus)
    return word_vec, char_vec


def transform_text(
    text: pd.Series,
    word_vec: TfidfVectorizer,
    char_vec: TfidfVectorizer,
) -> spmatrix:
    """
    Transform text into the combined TF-IDF feature space.

    Parameters:
    text (pd.Series): Text to transform.
    word_vec (TfidfVectorizer): Fitted word-level TF-IDF vectorizer.
    char_vec (TfidfVectorizer): Fitted character-level TF-IDF vectorizer.

    Returns:
    spmatrix: Combined sparse feature matrix.
    """
    return hstack([word_vec.transform(text), char_vec.transform(text)])


def top2_margin(prob_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the gap between the top two predicted probabilities.

    Purpose in pipeline:
    The margin is used as an additional confidence signal in child-cluster
    classification. A larger gap indicates a more decisive prediction.

    Parameters:
    prob_matrix (np.ndarray): Predicted class probabilities.

    Returns:
    np.ndarray: Top-two probability margin for each row.
    """
    if prob_matrix.shape[1] < 2:
        return np.ones(prob_matrix.shape[0])

    partitioned = np.partition(prob_matrix, -2, axis=1)
    top1 = partitioned[:, -1]
    top2 = partitioned[:, -2]
    return top1 - top2


def semantic_predict(
    x_text: spmatrix,
    desc_dict: dict[str, str],
    word_vec: TfidfVectorizer,
    char_vec: TfidfVectorizer,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict labels by comparing record vectors to description vectors.

    Purpose in pipeline:
    This is the semantic fallback mechanism used when rule-based or ML
    confidence is weak. It assigns the label whose description is most
    similar to the record in TF-IDF space.

    Parameters:
    x_text (spmatrix): Feature matrix for input records.
    desc_dict (dict[str, str]): Mapping from label to text description.
    word_vec (TfidfVectorizer): Fitted word-level vectorizer.
    char_vec (TfidfVectorizer): Fitted character-level vectorizer.

    Returns:
    tuple[np.ndarray, np.ndarray]: Predicted labels and similarity scores.
    """
    desc_series = pd.Series(desc_dict)
    x_desc = transform_text(desc_series, word_vec, char_vec)

    similarity = cosine_similarity(x_text, x_desc)
    best_idx = similarity.argmax(axis=1)
    best_labels = desc_series.index.to_numpy()[best_idx]
    best_scores = similarity.max(axis=1)
    return best_labels, best_scores


def get_cluster_name(cluster_label: str) -> str:
    """
    Convert a cluster label into a readable cluster name.

    Parameters:
    cluster_label (str): Internal cluster label.

    Returns:
    str: Human-readable cluster name.
    """
    if cluster_label in cluster_name_map:
        return cluster_name_map[cluster_label]

    if cluster_label.endswith("_unspecified"):
        parent = cluster_label.replace("_unspecified", "")
        return parent + " unspecified"

    return "unknown"


def prepare_known_unknown(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into seed-labeled and unlabeled subsets.

    Purpose in pipeline:
    Known rows supervise the ML models. Unknown rows are the recovery target
    for the hierarchical classifier.

    Parameters:
    df (pd.DataFrame): Dataset containing seed_cluster assignments.

    Returns:
    tuple[pd.DataFrame, pd.DataFrame]: Known and unknown subsets.
    """
    known_df = df[df["seed_cluster"] != "UNKNOWN"].copy()
    unknown_df = df[df["seed_cluster"] == "UNKNOWN"].copy()

    print(f"Labeled records: {len(known_df)}")
    print(f"Unlabeled records: {len(unknown_df)}")
    return known_df, unknown_df


def fit_global_vectorizers(df: pd.DataFrame) -> tuple[TfidfVectorizer, TfidfVectorizer]:
    """
    Fit global TF-IDF vectorizers using records and cluster descriptions.

    Purpose in pipeline:
    Including cluster descriptions in the fit corpus ensures that both record
    text and semantic reference text share the same feature space.

    Parameters:
    df (pd.DataFrame): Full dataset containing text_norm.

    Returns:
    tuple[TfidfVectorizer, TfidfVectorizer]: Fitted global TF-IDF vectorizers.
    """
    fit_corpus = pd.concat(
        [
            df["text_norm"],
            pd.Series(list(cluster_desc.values())),
            pd.Series(list(parent_desc.values())),
        ],
        ignore_index=True,
    )
    return build_features(fit_corpus)


def run_parent_clustering(
    known_df: pd.DataFrame,
    unknown_df: pd.DataFrame,
    word_vec: TfidfVectorizer,
    char_vec: TfidfVectorizer,
) -> np.ndarray:
    """
    Predict parent/system labels for UNKNOWN records.

    Purpose in pipeline:
    This is the first hierarchical ML stage. It predicts broad system-level
    categories before child-cluster classification.

    Key design decisions:
    - Logistic Regression is used because it performs well on sparse TF-IDF data.
    - Semantic similarity acts as a fallback when classifier confidence is weak.
    - Conservative thresholds reduce noisy parent assignments.

    Parameters:
    known_df (pd.DataFrame): Seed-labeled records.
    unknown_df (pd.DataFrame): UNKNOWN records targeted for recovery.
    word_vec (TfidfVectorizer): Fitted word-level vectorizer.
    char_vec (TfidfVectorizer): Fitted character-level vectorizer.

    Returns:
    np.ndarray: Parent predictions for UNKNOWN records.
    """
    if len(unknown_df) == 0:
        return np.array([], dtype=object)

    known_df = known_df.copy()
    known_df["parent_seed"] = known_df["seed_cluster"].map(cluster_to_parent)

    x_parent = transform_text(known_df["text_norm"], word_vec, char_vec)
    y_parent = known_df["parent_seed"]

    parent_clf: Optional[LogisticRegression] = None
    if len(known_df) > 0 and y_parent.nunique() >= 2:
        min_parent_count = int(y_parent.value_counts().min())

        if min_parent_count >= 2:
            x_train_p, x_val_p, y_train_p, y_val_p = train_test_split(
                x_parent,
                y_parent,
                test_size=0.2,
                random_state=RANDOM_STATE,
                stratify=y_parent,
            )

            parent_clf = LogisticRegression(
                max_iter=300,
                class_weight="balanced",
                C=0.7,
            )
            parent_clf.fit(x_train_p, y_train_p)

            print("Phase 1 parent model training complete")
            y_val_parent_pred = parent_clf.predict(x_val_p)
            print(
                "\nParent Validation Accuracy:",
                accuracy_score(y_val_p, y_val_parent_pred),
            )
            print("\nParent Classification Report:\n")
            print(classification_report(y_val_p, y_val_parent_pred, zero_division=0))
        else:
            # If a stratified validation split is not reliable, fit on all known data.
            parent_clf = LogisticRegression(
                max_iter=300,
                class_weight="balanced",
                C=0.7,
            )
            parent_clf.fit(x_parent, y_parent)
            print(
                "Phase 1 parent model trained without validation due to small class counts."
            )

    x_unknown = transform_text(unknown_df["text_norm"], word_vec, char_vec)
    final_parent = np.array(["UNKNOWN"] * len(unknown_df), dtype=object)

    if parent_clf is not None:
        parent_proba = parent_clf.predict_proba(x_unknown)
        parent_pred_labels = parent_clf.classes_[parent_proba.argmax(axis=1)]
        parent_pred_conf = parent_proba.max(axis=1)

        parent_sim_labels, parent_sim_conf = semantic_predict(
            x_unknown,
            parent_desc,
            word_vec,
            char_vec,
        )

        # Confidence-based fallback mechanism:
        # 1. accept strong ML prediction
        # 2. accept weaker ML prediction if semantic prediction agrees
        # 3. otherwise accept strong semantic match
        for i in range(len(unknown_df)):
            if parent_pred_conf[i] >= 0.5:
                final_parent[i] = parent_pred_labels[i]
            elif (
                parent_pred_labels[i] == parent_sim_labels[i]
                and parent_pred_conf[i] >= 0.2
            ):
                final_parent[i] = parent_pred_labels[i]
            elif parent_sim_conf[i] >= 0.2:
                final_parent[i] = parent_sim_labels[i]
    else:
        parent_sim_labels, parent_sim_conf = semantic_predict(
            x_unknown,
            parent_desc,
            word_vec,
            char_vec,
        )
        for i in range(len(unknown_df)):
            if parent_sim_conf[i] >= 0.12:
                final_parent[i] = parent_sim_labels[i]

    # Domain constraint: parent "baffle" should only be allowed when the
    # record explicitly mentions "baffle".
    unknown_has_baffle = (
        unknown_df["text_norm"]
        .str.contains(r"\bbaffle\b", regex=True, na=False)
        .to_numpy()
    )
    for i in range(len(final_parent)):
        if final_parent[i] == "baffle" and not unknown_has_baffle[i]:
            final_parent[i] = "UNKNOWN"

    return final_parent


def run_child_clustering(
    df: pd.DataFrame,
    known_df: pd.DataFrame,
    unknown_df: pd.DataFrame,
    final_parent: np.ndarray,
    word_vec: TfidfVectorizer,
    char_vec: TfidfVectorizer,
) -> np.ndarray:
    """
    Predict fine-grained child clusters within each parent system.

    Purpose in pipeline:
    This is the second hierarchical ML stage. It predicts child clusters only
    within the parent group predicted in the previous stage.

    Key design decisions:
    - A separate classifier is trained per parent cluster.
    - If too little child training data exists, the method falls back to
      `parent_name + "_unspecified"` instead of forcing a noisy child label.
    - Logistic Regression is used for sparse TF-IDF features.
    - Semantic similarity and confidence thresholds support fallback behavior.

    Parameters:
    df (pd.DataFrame): Full dataset.
    known_df (pd.DataFrame): Seed-labeled records.
    unknown_df (pd.DataFrame): UNKNOWN records targeted for recovery.
    final_parent (np.ndarray): Parent predictions for UNKNOWN records.
    word_vec (TfidfVectorizer): Fitted word-level vectorizer.
    char_vec (TfidfVectorizer): Fitted character-level vectorizer.

    Returns:
    np.ndarray: Child-cluster predictions for UNKNOWN records.
    """
    final_pred = np.array(["UNKNOWN"] * len(unknown_df), dtype=object)

    for parent_name, child_clusters in PARENT_TO_CHILDREN.items():
        if len(unknown_df) == 0:
            break

        target_pos = np.where(final_parent == parent_name)[0]
        if len(target_pos) == 0:
            continue

        parent_unknown_idx = unknown_df.index[target_pos]
        unknown_subset = df.loc[parent_unknown_idx].copy()
        train_subset = known_df[known_df["seed_cluster"].isin(child_clusters)].copy()

        if len(child_clusters) == 1:
            final_pred[target_pos] = child_clusters[0]
            continue

        if len(train_subset) < 12 or train_subset["seed_cluster"].nunique() < 2:
            final_pred[target_pos] = parent_name + "_unspecified"
            continue

        x_train_child = transform_text(train_subset["text_norm"], word_vec, char_vec)
        y_train_child = train_subset["seed_cluster"]

        min_child_count = int(y_train_child.value_counts().min())
        if min_child_count >= 2:
            x_tr, x_va, y_tr, y_va = train_test_split(
                x_train_child,
                y_train_child,
                test_size=0.2,
                random_state=RANDOM_STATE,
                stratify=y_train_child,
            )

            child_val_clf = LogisticRegression(
                max_iter=300,
                class_weight="balanced",
                C=0.5,
            )
            child_val_clf.fit(x_tr, y_tr)

            y_child_val_pred = child_val_clf.predict(x_va)
            print(f"\nChild Validation for {parent_name}:")
            print("Accuracy:", accuracy_score(y_va, y_child_val_pred))
            print(classification_report(y_va, y_child_val_pred, zero_division=0))

        child_clf = LogisticRegression(
            max_iter=300,
            class_weight="balanced",
            C=0.5,
        )
        child_clf.fit(x_train_child, y_train_child)

        x_unknown_child = transform_text(
            unknown_subset["text_norm"],
            word_vec,
            char_vec,
        )
        child_proba = child_clf.predict_proba(x_unknown_child)
        child_pred_labels = child_clf.classes_[child_proba.argmax(axis=1)]
        child_pred_conf = child_proba.max(axis=1)
        child_margin = top2_margin(child_proba)

        local_desc = {cluster: cluster_desc[cluster] for cluster in child_clusters}
        child_sim_labels, child_sim_conf = semantic_predict(
            x_unknown_child,
            local_desc,
            word_vec,
            char_vec,
        )

        local_result = np.array(["UNKNOWN"] * len(unknown_subset), dtype=object)

        # Child-level confidence policy:
        # 1. accept strong ML prediction
        # 2. accept ML when semantic prediction agrees
        # 3. accept strong semantic match
        # 4. otherwise fall back to parent_unspecified
        for j in range(len(unknown_subset)):
            if child_pred_conf[j] >= 0.78 and child_margin[j] >= 0.22:
                local_result[j] = child_pred_labels[j]
            elif (
                child_pred_labels[j] == child_sim_labels[j]
                and child_pred_conf[j] >= 0.5
                and child_sim_conf[j] >= 0.32
            ):
                local_result[j] = child_pred_labels[j]
            elif child_sim_conf[j] >= 0.42:
                local_result[j] = child_sim_labels[j]
            else:
                local_result[j] = parent_name + "_unspecified"

        if len(local_result) != len(target_pos):
            raise ValueError(
                f"Length mismatch for parent {parent_name}: "
                f"{len(local_result)} results for {len(target_pos)} target positions"
            )

        final_pred[target_pos] = local_result

    return final_pred


def combine_final_clusters(
    df: pd.DataFrame,
    unknown_df: pd.DataFrame,
    final_parent: np.ndarray,
    final_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Combine seed labels, parent predictions, and child predictions.

    Purpose in pipeline:
    This function produces the final cluster column by preserving known
    seed labels and filling UNKNOWN rows with the hierarchical outputs.

    Parameters:
    df (pd.DataFrame): Full dataset.
    unknown_df (pd.DataFrame): Rows that were UNKNOWN after the seed stage.
    final_parent (np.ndarray): Parent predictions for UNKNOWN rows.
    final_pred (np.ndarray): Child predictions for UNKNOWN rows.

    Returns:
    pd.DataFrame: Dataset with the final cluster column populated.
    """
    df = df.copy()
    df["cluster"] = df["seed_cluster"]

    if len(unknown_df) > 0:
        df.loc[unknown_df.index, "cluster"] = final_pred

    for i, idx in enumerate(unknown_df.index):
        if df.loc[idx, "cluster"] == "UNKNOWN" and final_parent[i] != "UNKNOWN":
            df.loc[idx, "cluster"] = final_parent[i] + "_unspecified"

    return df


def apply_final_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply final domain-specific overrides after hierarchical classification.

    Purpose in pipeline:
    Some cases are better handled with explicit business rules than by ML alone,
    especially dirty/oil/filter/tire-related records.

    Key design decisions:
    - Specific filter cases are handled before generic filter fallbacks.
    - Tire logic only affects records that remain UNKNOWN.
    - If text_norm is absent, it is reconstructed from PROBLEM and ACTION.

    Parameters:
    df (pd.DataFrame): Dataset containing cluster information and maintenance text.

    Returns:
    pd.DataFrame: Dataset with final override corrections applied.
    """
    df = df.copy()

    if "PROBLEM" not in df.columns:
        df["PROBLEM"] = ""

    if "ACTION" not in df.columns:
        df["ACTION"] = ""

    if "text_norm" not in df.columns:
        problem_norm = normalize_series(df["PROBLEM"])
        action_norm = normalize_series(df["ACTION"])
        df["text_norm"] = (problem_norm + " " + action_norm).str.strip()

    text = df["text_norm"]

    engine_dirty = text.str.contains(
        r"\bengine\b",
        regex=True,
        na=False,
    ) & text.str.contains(r"\bdirty\b", regex=True, na=False)
    oil_cooler_dirty = text.str.contains(
        r"\boil cooler\b",
        regex=True,
        na=False,
    ) & text.str.contains(r"\bdirty\b", regex=True, na=False)

    oil_filter_dirty = text.str.contains(
        r"\boil filter\b",
        regex=True,
        na=False,
    ) & text.str.contains(r"\bdirty\b", regex=True, na=False)

    fuel_filter_dirty = text.str.contains(
        r"\bfuel filter\b",
        regex=True,
        na=False,
    ) & text.str.contains(r"\bdirty\b", regex=True, na=False)

    air_filter_dirty = text.str.contains(
        r"\bair filter\b",
        regex=True,
        na=False,
    ) & text.str.contains(r"\bdirty\b", regex=True, na=False)

    generic_filter_dirty = text.str.contains(
        r"\bfilter\b",
        regex=True,
        na=False,
    ) & text.str.contains(r"\bdirty\b", regex=True, na=False)

    # Specific subsystem cases are assigned before generic fallback logic.
    df.loc[oil_cooler_dirty, "cluster"] = "c_32"
    df.loc[oil_filter_dirty, "cluster"] = "oil_system_unspecified"
    df.loc[fuel_filter_dirty, "cluster"] = "fuel_control_unspecified"
    df.loc[air_filter_dirty, "cluster"] = "induction_intake_unspecified"

    df.loc[
        generic_filter_dirty
        & ~oil_filter_dirty
        & ~fuel_filter_dirty
        & ~air_filter_dirty
        & (df["cluster"] == "UNKNOWN"),
        "cluster",
    ] = "engine_general_unspecified"

    df.loc[
        engine_dirty & ~oil_cooler_dirty & (df["cluster"] == "UNKNOWN"),
        "cluster",
    ] = "engine_general_unspecified"

    tire_problem_unknown = df["PROBLEM"].fillna("").astype(
        str
    ).str.lower().str.contains(r"\btire\b", regex=True, na=False) & (
        df["cluster"] == "UNKNOWN"
    )
    df.loc[tire_problem_unknown, "cluster"] = "c_44"

    return df


def save_outputs(df: pd.DataFrame, before_unknown: int) -> None:
    """
    Save the final clustered dataset and print recovery diagnostics.

    Purpose in pipeline:
    This is the terminal output stage. It writes the final labeled dataset
    and reports how many UNKNOWN records were recovered.

    Parameters:
    df (pd.DataFrame): Final dataset containing cluster assignments.
    before_unknown (int): Number of UNKNOWN records after seed labeling.

    Returns:
    None: The function writes the final dataset to disk.
    """
    after_unknown = int((df["cluster"] == "UNKNOWN").sum())
    recovered = before_unknown - after_unknown

    print(f"UNKNOWN before semantic/hierarchical: {before_unknown}")
    print(f"UNKNOWN after semantic/hierarchical: {after_unknown}")
    print(f"Recovered records: {recovered}")
    if before_unknown > 0:
        print(f"Reduction: {(recovered / before_unknown) * 100:.2f}%")
    else:
        print("Reduction: 0.00%")

    print("\nTop clusters:")
    print(df["cluster"].value_counts().head(60))

    df["cluster_name"] = df["cluster"].apply(get_cluster_name)

    df[
        ["WKO#", "ATA_Code", "PROBLEM", "DATE", "ACTION", "cluster", "cluster_name"]
    ].to_csv(
        OUTPUT_DATASET,
        index=False,
    )

    print("Files saved successfully")


def apply_flight_control_override(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply flight-control-related seed overrides.

    Parameters:
    df (pd.DataFrame): Dataset used for override testing.

    Returns:
    pd.DataFrame: Dataset after seed overrides are applied.
    """
    return apply_seed_overrides(df)


def apply_dirty_seed_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply dirty-related seed overrides.

    Parameters:
    df (pd.DataFrame): Dataset used for override testing.

    Returns:
    pd.DataFrame: Dataset after seed overrides are applied.
    """
    return apply_seed_overrides(df)


def apply_final_dirty_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply final dirty/filter overrides.

    Parameters:
    df (pd.DataFrame): Dataset used for override testing.

    Returns:
    pd.DataFrame: Dataset after final overrides are applied.
    """
    return apply_final_overrides(df)


def apply_final_tire_override(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply final tire-related overrides.

    Parameters:
    df (pd.DataFrame): Dataset used for override testing.

    Returns:
    pd.DataFrame: Dataset after final overrides are applied.
    """
    return apply_final_overrides(df)


def main() -> None:
    """
    Execute the full maintenance clustering pipeline.

    Purpose in pipeline:
    This function orchestrates the complete workflow:
    1. load and normalize data
    2. assign seed labels using rules
    3. apply temperature split, prop split, inspection, and deterministic overrides
    4. fit TF-IDF features
    5. run parent-level ML classification
    6. run child-level ML classification
    7. apply final override logic
    8. save the final dataset

    Parameters:
    None: This function does not accept external parameters.

    Returns:
    None: The function writes pipeline outputs to disk.
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    df = load_input_data(INPUT_FILE)

    print("INITIAL SEED CLUSTERING")

    all_matches = get_all_matches(df["text_norm"], seed_rules, baffle_and_rules)
    df["seed_cluster"] = apply_seed_rules(
        df["text_norm"],
        seed_rules,
        baffle_and_rules,
    )

    # Temperature split is applied before other overrides so that
    # pure EGT-only and CHT-only records are separated early.
    df = apply_temperature_split_override(df)

    # Propeller split is applied before other overrides so that
    # pure overspeed-only and damage-only records are separated early.
    df = apply_prop_split_override(df)

    # Inspection cleanup is applied before general seed overrides so that
    # c_0 remains more precise.
    df = apply_inspection_override(df)
    df = apply_seed_overrides(df)

    before_unknown = analyze_seed_assignment(all_matches, df)
    known_df, unknown_df = prepare_known_unknown(df)

    print("FEATURE ENGINEERING")

    word_vec, char_vec = fit_global_vectorizers(df)
    print("Features completed")

    print("PHASE 1: PARENT CLUSTERING")
    final_parent = run_parent_clustering(known_df, unknown_df, word_vec, char_vec)

    print("PHASE 2: CHILD CLUSTERING")
    final_pred = run_child_clustering(
        df,
        known_df,
        unknown_df,
        final_parent,
        word_vec,
        char_vec,
    )

    print("FINAL CLUSTER ASSIGNMENT")

    df = combine_final_clusters(df, unknown_df, final_parent, final_pred)
    df = apply_final_overrides(df)

    save_outputs(df, before_unknown)


if __name__ == "__main__":
    main()
