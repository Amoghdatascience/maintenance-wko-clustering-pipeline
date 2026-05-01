"""
Aviation Maintenance Workorder Clustering Pipeline

This script converts raw aviation maintenance workorders into structured
cluster labels using a hybrid pipeline that combines:

1. Rule-based classification (seed rules + strict token rules)
2. Machine learning (Logistic Regression with TF-IDF features)
3. Semantic similarity (cosine similarity against cluster descriptions)
4. Hierarchical prediction (Parent -> Child clusters)

Outputs
-------
- datasets/dataset.csv: final clustered workorder records
- datasets/summary.csv: UNKNOWN before/after/recovered metrics

Author: Amogh Naik
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Configuration

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"

INPUT_FILE = DATASET_DIR / "Valid_Problems_Workorder.xlsx"
OUTPUT_DATASET = DATASET_DIR / "dataset.csv"
#OUTPUT_SUMMARY = DATASET_DIR / "summary.csv"
RANDOM_STATE = 42

abbr_map = {
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
    r"\bgpu\b": "ground power unit",
}

cluster_desc = {
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
    "c_16": "cylinder head temperature exhaust gas temperature",
    "c_17": "pushrod tube",
    "c_18": "drain line tube",
    "c_19": "carburetor carb",
    "c_20": "crankcase crankshaft firewall",
    "c_21": "engine failure fire power loss engine quit",
    "c_22": "engine idle rpm issue",
    "c_23": "engine repair reinstall clean remove replace engine",
    "c_24": "engine run rough rough running misfire",
    "c_25": "engine seal tube bolt loose",
    "c_26": "propeller overspeed prop damage",
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
    "c_43": "appearance cleaning paint wash dirty exterior clean fuselage surface finish cosmetic exterior surface",
    "c_44": "landing gear tire",
}

cluster_name_map = cluster_desc.copy()

parent_desc = {
    "inspection": "inspection routine scheduled inspection check",
    "start_system": "aircraft start starter hard start no start no crank external start ground power",
    "baffle": "baffle bolt bracket crack mount plug rivet screw seal spring tie rod",
    "cowling": "cowling cowl damage loose missing",
    "cylinder_Exhaust": "cylinder compression crack exhaust valve temperature push rod pushrod tube",
    "engine_general": "engine rough idle repair crankcase carburetor engine failure power loss",
    "propeller": "propeller overspeed propeller damage",
    "induction_intake": "induction intake gasket boot seal leak",
    "ignition": "magneto ignition spark plug mag drop fouled plug",
    "fuel_control": "mixture mixture control fuel adjust",
    "oil_system": "oil cooler dipstick filler tube oil leak oil pressure oil temperature return line",
    "pilot_reported": "pilot reported in flight pilot noticed",
    "valve_cover": "rocker cover valve cover sniffler valve",
    "electrical": "battery voltage charging alternator electrical",
    "appearance_cleaning": "paint wash dirty exterior clean fuselage surface finish cosmetic exterior appearance cleaning",
    "landing_gear_tire": "landing gear tire",
}

cluster_to_parent = {
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
}


def build_parent_to_children_map(
    cluster_parent_map: dict[str, str],
) -> dict[str, list[str]]:
    parent_to_children: dict[str, list[str]] = {}
    for child, parent in cluster_parent_map.items():
        parent_to_children.setdefault(parent, []).append(child)
    return parent_to_children


PARENT_TO_CHILDREN = build_parent_to_children_map(cluster_to_parent)

seed_rules = {
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
    "c_16": ["cylinder head temperature", "exhaust gas temperature", "egt", "cht"],
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
    "c_26": ["propeller overspeed", "overspeed", "prop damage"],
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
}

baffle_and_rules = {
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
    normalized = series.fillna("").astype(str).str.lower()
    normalized = normalized.str.replace(r"[/\-]", " ", regex=True)

    for pattern, replacement in abbr_map.items():
        normalized = normalized.str.replace(pattern, replacement, regex=True)

    normalized = normalized.str.replace(r"[^a-z0-9\s]", " ", regex=True)
    normalized = normalized.str.replace(r"\s+", " ", regex=True).str.strip()
    return normalized


def load_input_data(path: str | Path) -> pd.DataFrame:
    print("Loading dataset")
    df = pd.read_excel(path)
    print(f"Total records: {len(df)}")

    df["problem_norm"] = normalize_series(df["PROBLEM"])
    df["action_norm"] = normalize_series(df["ACTION"])
    df["text_norm"] = (df["problem_norm"] + " " + df["action_norm"]).str.strip()

    print("Text normalization complete")
    return df


def contains_all_tokens(text_series: pd.Series, tokens: list[str]) -> pd.Series:
    mask = pd.Series(True, index=text_series.index)
    for token in tokens:
        mask &= text_series.str.contains(
            rf"\b{re.escape(token)}\b", regex=True, na=False
        )
    return mask


def apply_seed_rules(
    text_series: pd.Series,
    rule_dict: dict[str, list[str]],
    and_rule_dict: dict[str, list[list[str]]] | None = None,
) -> pd.Series:
    """
    Assign high-confidence initial labels using domain-specific keyword rules.

    First matching rule wins.
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
    and_rule_dict: dict[str, list[list[str]]] | None = None,
) -> dict[int, list[str]]:
    matches = {idx: [] for idx in text_series.index}

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


def classify_c0_type(text: str) -> str:
    """
    Grounded inspection logic based on real c_0 trends.

    Returns one of:
    - true_inspection
    - inspection_admin
    - inspection_panel_misfit
    - misfit_recluster
    - review_needed
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
    Keep only grounded true inspection/admin records in c_0.
    Release panel/repair/defect inspection mentions back to UNKNOWN.
    """
    df = df.copy()

    inspection_related = df["text_norm"].str.contains(
        r"\binspection\b|\binsp\b|\bannual\b|\bbiennial\b|\bnext due\b|\bad\b|\bservice bulletin\b",
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

    appearance_dirty = dirty_at_end & ~oil_cooler_dirty & ~engine_dirty & ~filter_dirty
    df.loc[appearance_dirty, "seed_cluster"] = "c_43"

    return df


def analyze_seed_assignment(all_matches: dict[int, list[str]], df: pd.DataFrame) -> int:
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
):
    return hstack([word_vec.transform(text), char_vec.transform(text)])


def top2_margin(prob_matrix: np.ndarray) -> np.ndarray:
    if prob_matrix.shape[1] < 2:
        return np.ones(prob_matrix.shape[0])

    partitioned = np.partition(prob_matrix, -2, axis=1)
    top1 = partitioned[:, -1]
    top2 = partitioned[:, -2]
    return top1 - top2


def semantic_predict(
    x_text,
    desc_dict: dict[str, str],
    word_vec: TfidfVectorizer,
    char_vec: TfidfVectorizer,
) -> tuple[np.ndarray, np.ndarray]:
    desc_series = pd.Series(desc_dict)
    x_desc = transform_text(desc_series, word_vec, char_vec)

    similarity = cosine_similarity(x_text, x_desc)
    best_idx = similarity.argmax(axis=1)
    best_labels = desc_series.index.to_numpy()[best_idx]
    best_scores = similarity.max(axis=1)
    return best_labels, best_scores


def get_cluster_name(cluster_label: str) -> str:
    if cluster_label in cluster_name_map:
        return cluster_name_map[cluster_label]

    if cluster_label.endswith("_unspecified"):
        parent = cluster_label.replace("_unspecified", "")
        return parent + " unspecified"

    return "unknown"


def prepare_known_unknown(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    known_df = df[df["seed_cluster"] != "UNKNOWN"].copy()
    unknown_df = df[df["seed_cluster"] == "UNKNOWN"].copy()

    print(f"Labeled records: {len(known_df)}")
    print(f"Unlabeled records: {len(unknown_df)}")
    return known_df, unknown_df


def fit_global_vectorizers(df: pd.DataFrame) -> tuple[TfidfVectorizer, TfidfVectorizer]:
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
    if len(unknown_df) == 0:
        return np.array([], dtype=object)

    known_df = known_df.copy()
    known_df["parent_seed"] = known_df["seed_cluster"].map(cluster_to_parent)

    x_parent = transform_text(known_df["text_norm"], word_vec, char_vec)
    y_parent = known_df["parent_seed"]

    parent_clf = None
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
                max_iter=300, class_weight="balanced", C=0.7
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
            parent_clf = LogisticRegression(
                max_iter=300, class_weight="balanced", C=0.7
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
            x_unknown, parent_desc, word_vec, char_vec
        )

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
            x_unknown, parent_desc, word_vec, char_vec
        )
        for i in range(len(unknown_df)):
            if parent_sim_conf[i] >= 0.12:
                final_parent[i] = parent_sim_labels[i]

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
                max_iter=300, class_weight="balanced", C=0.5
            )
            child_val_clf.fit(x_tr, y_tr)

            y_child_val_pred = child_val_clf.predict(x_va)
            print(f"\nChild Validation for {parent_name}:")
            print("Accuracy:", accuracy_score(y_va, y_child_val_pred))
            print(classification_report(y_va, y_child_val_pred, zero_division=0))

        child_clf = LogisticRegression(max_iter=300, class_weight="balanced", C=0.5)
        child_clf.fit(x_train_child, y_train_child)

        x_unknown_child = transform_text(
            unknown_subset["text_norm"], word_vec, char_vec
        )
        child_proba = child_clf.predict_proba(x_unknown_child)
        child_pred_labels = child_clf.classes_[child_proba.argmax(axis=1)]
        child_pred_conf = child_proba.max(axis=1)
        child_margin = top2_margin(child_proba)

        local_desc = {cluster: cluster_desc[cluster] for cluster in child_clusters}
        child_sim_labels, child_sim_conf = semantic_predict(
            x_unknown_child, local_desc, word_vec, char_vec
        )

        local_result = np.array(["UNKNOWN"] * len(unknown_subset), dtype=object)

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
    Apply final domain-specific override rules after ML-based clustering.
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

    engine_dirty = text.str.contains(r"\bengine\b", regex=True, na=False) & text.str.contains(
        r"\bdirty\b", regex=True, na=False
    )
    oil_cooler_dirty = text.str.contains(
        r"\boil cooler\b", regex=True, na=False
    ) & text.str.contains(r"\bdirty\b", regex=True, na=False)

    oil_filter_dirty = text.str.contains(
        r"\boil filter\b", regex=True, na=False
    ) & text.str.contains(r"\bdirty\b", regex=True, na=False)

    fuel_filter_dirty = text.str.contains(
        r"\bfuel filter\b", regex=True, na=False
    ) & text.str.contains(r"\bdirty\b", regex=True, na=False)

    air_filter_dirty = text.str.contains(
        r"\bair filter\b", regex=True, na=False
    ) & text.str.contains(r"\bdirty\b", regex=True, na=False)

    generic_filter_dirty = text.str.contains(
        r"\bfilter\b", regex=True, na=False
    ) & text.str.contains(r"\bdirty\b", regex=True, na=False)

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

    tire_problem_unknown = (
        df["PROBLEM"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains(r"\btire\b", regex=True, na=False)
        & (df["cluster"] == "UNKNOWN")
    )
    df.loc[tire_problem_unknown, "cluster"] = "c_44"

    return df


def save_outputs(df: pd.DataFrame, before_unknown: int) -> None:
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
    return apply_seed_overrides(df)


def apply_dirty_seed_overrides(df: pd.DataFrame) -> pd.DataFrame:
    return apply_seed_overrides(df)


def apply_final_dirty_overrides(df: pd.DataFrame) -> pd.DataFrame:
    return apply_final_overrides(df)


def apply_final_tire_override(df: pd.DataFrame) -> pd.DataFrame:
    return apply_final_overrides(df)


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    df = load_input_data(INPUT_FILE)

    print("INITIAL SEED CLUSTERING")

    all_matches = get_all_matches(df["text_norm"], seed_rules, baffle_and_rules)
    df["seed_cluster"] = apply_seed_rules(df["text_norm"], seed_rules, baffle_and_rules)

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
        df, known_df, unknown_df, final_parent, word_vec, char_vec
    )

    print("FINAL CLUSTER ASSIGNMENT")

    df = combine_final_clusters(df, unknown_df, final_parent, final_pred)
    df = apply_final_overrides(df)

    save_outputs(df, before_unknown)


if __name__ == "__main__":
    main()