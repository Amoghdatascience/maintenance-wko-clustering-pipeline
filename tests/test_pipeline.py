import pandas as pd

from semantic_clustering_pipe import (
    normalize_series,
    contains_all_tokens,
    apply_seed_rules,
    seed_rules,
    baffle_and_rules,
    apply_flight_control_override,
    apply_dirty_seed_overrides,
    apply_final_dirty_overrides,
    apply_final_tire_override,
)

from merge import build_joined_dataset


# TEXT NORMALIZATION TESTS

def test_normalization_expands_abbreviations():
    s = pd.Series(["ENG RGH MAG DROP CHT HIGH"])
    result = normalize_series(s)

    assert "engine" in result.iloc[0]
    assert "rough" in result.iloc[0]
    assert "magneto" in result.iloc[0]
    assert "cylinder head temperature" in result.iloc[0]


def test_normalization_handles_nulls():
    s = pd.Series([None])
    result = normalize_series(s)

    assert result.iloc[0] == ""


# BAFFLE AND-TOKEN RULE TESTS

def test_contains_all_tokens_true():
    s = pd.Series(["baffle loose replaced screw"])
    result = contains_all_tokens(s, ["baffle", "screw"])

    assert result.iloc[0] == True


def test_contains_all_tokens_false():
    s = pd.Series(["screw replaced only"])
    result = contains_all_tokens(s, ["baffle", "screw"])

    assert result.iloc[0] == False


def test_baffle_child_requires_both_tokens():
    s = pd.Series([
        "baffle damaged replaced bolt",
        "bolt replaced only",
        "baffle inspected only",
    ])

    labels = apply_seed_rules(s, seed_rules, baffle_and_rules)

    assert labels.iloc[0] == "c_2"
    assert labels.iloc[1] == "UNKNOWN"
    assert labels.iloc[2] == "UNKNOWN"


def test_baffle_rivet_rule():
    s = pd.Series([
        "baffle loose installed rivet",
        "rivet installed",
    ])

    labels = apply_seed_rules(s, seed_rules, baffle_and_rules)

    assert labels.iloc[0] == "c_7"
    assert labels.iloc[1] == "UNKNOWN"


# FLIGHT CONTROL OVERRIDE TESTS

def test_flight_control_override_sets_c42():
    df = pd.DataFrame({
        "text_norm": ["aileron cable fairlead due", "engine rough"],
        "seed_cluster": ["UNKNOWN", "UNKNOWN"],
    })

    result = apply_flight_control_override(df)

    assert result.loc[0, "seed_cluster"] == "c_42"
    assert result.loc[1, "seed_cluster"] == "UNKNOWN"


def test_flight_control_override_for_rudder():
    df = pd.DataFrame({
        "text_norm": ["rudder trim issue"],
        "seed_cluster": ["UNKNOWN"],
    })

    result = apply_flight_control_override(df)

    assert result.loc[0, "seed_cluster"] == "c_42"


# DIRTY / APPEARANCE / FILTER OVERRIDE TESTS

def test_oil_cooler_dirty_seed_goes_to_c32():
    df = pd.DataFrame({
        "text_norm": ["oil cooler dirty"],
        "seed_cluster": ["UNKNOWN"],
    })

    result = apply_dirty_seed_overrides(df)

    assert result.loc[0, "seed_cluster"] == "c_32"


def test_dirty_at_end_goes_to_appearance_cluster():
    df = pd.DataFrame({
        "text_norm": ["aircraft exterior dirty"],
        "seed_cluster": ["UNKNOWN"],
    })

    result = apply_dirty_seed_overrides(df)

    assert result.loc[0, "seed_cluster"] == "c_43"


def test_engine_dirty_final_goes_to_engine_unspecified():
    df = pd.DataFrame({
        "text_norm": ["engine dirty"],
        "cluster": ["UNKNOWN"],
    })

    result = apply_final_dirty_overrides(df)

    assert result.loc[0, "cluster"] == "engine_general_unspecified"


def test_air_filter_dirty_goes_to_induction_unspecified():
    df = pd.DataFrame({
        "text_norm": ["air filter dirty"],
        "cluster": ["UNKNOWN"],
    })

    result = apply_final_dirty_overrides(df)

    assert result.loc[0, "cluster"] == "induction_intake_unspecified"


def test_oil_filter_dirty_goes_to_oil_unspecified():
    df = pd.DataFrame({
        "text_norm": ["oil filter dirty"],
        "cluster": ["UNKNOWN"],
    })

    result = apply_final_dirty_overrides(df)

    assert result.loc[0, "cluster"] == "oil_system_unspecified"


def test_fuel_filter_dirty_goes_to_fuel_unspecified():
    df = pd.DataFrame({
        "text_norm": ["fuel filter dirty"],
        "cluster": ["UNKNOWN"],
    })

    result = apply_final_dirty_overrides(df)

    assert result.loc[0, "cluster"] == "fuel_control_unspecified"


def test_generic_filter_dirty_goes_to_engine_unspecified():
    df = pd.DataFrame({
        "text_norm": ["filter dirty"],
        "cluster": ["UNKNOWN"],
    })

    result = apply_final_dirty_overrides(df)

    assert result.loc[0, "cluster"] == "engine_general_unspecified"


def test_oil_cooler_dirty_has_priority_over_engine_dirty():
    df = pd.DataFrame({
        "text_norm": ["engine oil cooler dirty"],
        "cluster": ["UNKNOWN"],
    })

    result = apply_final_dirty_overrides(df)

    assert result.loc[0, "cluster"] == "c_32"


# TIRE FINAL OVERRIDE TESTS

def test_tire_problem_unknown_goes_to_c44():
    df = pd.DataFrame({
        "PROBLEM": ["NOSE TIRE FLAT"],
        "ACTION": ["replaced tire"],
        "cluster": ["UNKNOWN"],
    })

    result = apply_final_tire_override(df)

    assert result.loc[0, "cluster"] == "c_44"


def test_tire_action_only_does_not_change_cluster():
    df = pd.DataFrame({
        "PROBLEM": ["ENGINE ISSUE"],
        "ACTION": ["replaced tire"],
        "cluster": ["UNKNOWN"],
    })

    result = apply_final_tire_override(df)

    assert result.loc[0, "cluster"] == "UNKNOWN"


def test_tire_does_not_override_existing_cluster():
    df = pd.DataFrame({
        "PROBLEM": ["NOSE TIRE FLAT"],
        "ACTION": ["replaced tire"],
        "cluster": ["c_0"],
    })

    result = apply_final_tire_override(df)

    assert result.loc[0, "cluster"] == "c_0"


# JOIN DATASET TEST

def test_build_joined_dataset_aggregates_and_joins():
    df = pd.DataFrame({
        "WKO#": ["1001.0", "1001.0", "1002.0"],
        "Date_Opened": ["2024-01-01", "2024-01-02", "2024-02-01"],
        "Date_Closed": ["2024-01-03", "2024-01-04", "2024-02-02"],
        "Registration#": ["N123", "N123", "N456"],
        "Part#": ["P1", "P2", "P3"],
        "Total_Time": [2.0, 3.0, 5.0],
        "TSO": [10, 20, 30],
    })

    dataset = pd.DataFrame({
        "WKO#": ["1001", "1003"],
        "ATA_Code": [560, 320],
        "PROBLEM": ["TEST PROBLEM", "OTHER"],
        "DATE": ["2024-01-02", "2024-03-01"],
        "ACTION": ["TEST ACTION", "OTHER ACTION"],
        "cluster": ["c_0", "UNKNOWN"],
        "cluster_name": ["inspection", "unknown"],
    })

    result = build_joined_dataset(df, dataset)

    assert len(result) == 1
    assert result.loc[0, "workorder"] == "1001"
    assert result.loc[0, "total_time"] == 5.0
    assert result.loc[0, "registration"] == "N123"
    assert result.loc[0, "ata_code"] == 560
    assert result.loc[0, "cluster"] == "c_0"
    assert result.loc[0, "cluster_name"] == "inspection"