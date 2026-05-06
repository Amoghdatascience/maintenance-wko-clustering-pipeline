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
    apply_temperature_split_override,
)
from merge import build_joined_dataset


def test_normalization_expands_abbreviations() -> None:
    """
    Verify that text normalization expands common aviation abbreviations.

    Purpose in pipeline:
    This test validates the preprocessing stage, which standardizes shorthand
    maintenance language before rule-based and ML classification.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts that key abbreviations are expanded correctly.
    """
    s = pd.Series(["ENG RGH MAG DROP CHT HIGH"])
    result = normalize_series(s)

    assert "engine" in result.iloc[0]
    assert "rough" in result.iloc[0]
    assert "magneto" in result.iloc[0]
    assert "cylinder head temperature" in result.iloc[0]


def test_normalization_handles_nulls() -> None:
    """
    Verify that text normalization handles null values safely.

    Purpose in pipeline:
    Maintenance datasets can contain missing text fields. This test ensures
    that preprocessing converts null values into empty strings rather than
    failing or propagating invalid values.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts that null input becomes an empty string.
    """
    s = pd.Series([None])
    result = normalize_series(s)

    assert result.iloc[0] == ""


def test_cht_only_goes_to_c16() -> None:
    """
    Verify that pure cylinder head temperature text maps to c_16.

    Purpose in pipeline:
    The old combined temperature cluster was split into separate CHT and EGT
    clusters. This test ensures that CHT-only records are assigned correctly.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts correct CHT override behavior.
    """
    df = pd.DataFrame(
        {
            "text_norm": ["cylinder head temperature high"],
            "seed_cluster": ["UNKNOWN"],
        }
    )

    result = apply_temperature_split_override(df)

    assert result.loc[0, "seed_cluster"] == "c_16"


def test_egt_only_goes_to_c45() -> None:
    """
    Verify that pure exhaust gas temperature text maps to c_45.

    Purpose in pipeline:
    This test ensures that EGT-only records are separated from CHT and assigned
    to the new c_45 cluster.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts correct EGT override behavior.
    """
    df = pd.DataFrame(
        {
            "text_norm": ["exhaust gas temperature high"],
            "seed_cluster": ["UNKNOWN"],
        }
    )

    result = apply_temperature_split_override(df)

    assert result.loc[0, "seed_cluster"] == "c_45"


def test_mixed_cht_egt_is_left_unchanged() -> None:
    """
    Verify that records mentioning both CHT and EGT are not force-split here.

    Purpose in pipeline:
    The temperature override only separates pure CHT-only and EGT-only cases.
    Mixed records are intentionally left unchanged so the rest of the pipeline
    can handle them.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts that mixed records remain unchanged.
    """
    df = pd.DataFrame(
        {
            "text_norm": ["cylinder head temperature and exhaust gas temperature high"],
            "seed_cluster": ["UNKNOWN"],
        }
    )

    result = apply_temperature_split_override(df)

    assert result.loc[0, "seed_cluster"] == "UNKNOWN"


def test_contains_all_tokens_true() -> None:
    """
    Verify that the strict token matcher returns True when all tokens exist.

    Purpose in pipeline:
    The baffle child rules depend on exact co-occurrence of multiple tokens.
    This test validates the helper used for strict AND-token matching.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts that all required tokens are detected.
    """
    s = pd.Series(["baffle loose replaced screw"])
    result = contains_all_tokens(s, ["baffle", "screw"])

    assert result.iloc[0] == True


def test_contains_all_tokens_false() -> None:
    """
    Verify that the strict token matcher returns False when a token is missing.

    Purpose in pipeline:
    This test ensures that partial matches do not trigger strict baffle rules.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts that missing tokens prevent a match.
    """
    s = pd.Series(["screw replaced only"])
    result = contains_all_tokens(s, ["baffle", "screw"])

    assert result.iloc[0] == False


def test_baffle_child_requires_both_tokens() -> None:
    """
    Verify that baffle child labeling requires full token co-occurrence.

    Purpose in pipeline:
    Baffle-related records are highly overlapping, so child rules must remain
    strict to avoid assigning labels from generic component mentions.

    Key design decisions:
    - The first record contains both required tokens and should match.
    - The remaining records are partial matches and should remain UNKNOWN.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts correct strict-rule behavior.
    """
    s = pd.Series(
        [
            "baffle damaged replaced bolt",
            "bolt replaced only",
            "baffle inspected only",
        ]
    )

    labels = apply_seed_rules(s, seed_rules, baffle_and_rules)

    assert labels.iloc[0] == "c_2"
    assert labels.iloc[1] == "UNKNOWN"
    assert labels.iloc[2] == "UNKNOWN"


def test_baffle_rivet_rule() -> None:
    """
    Verify that the baffle rivet rule is applied correctly.

    Purpose in pipeline:
    This test checks that specific baffle child rules remain stable even when
    multiple possible baffle-related terms occur in the same record.

    Key design decisions:
    - The first row should match the explicit baffle+rivet rule.
    - The second row lacks the required baffle term and should remain UNKNOWN.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts correct baffle child assignment.
    """
    s = pd.Series(
        [
            "baffle loose installed rivet",
            "rivet installed",
        ]
    )

    labels = apply_seed_rules(s, seed_rules, baffle_and_rules)

    assert labels.iloc[0] == "c_7"
    assert labels.iloc[1] == "UNKNOWN"


def test_flight_control_override_sets_c42() -> None:
    """
    Verify that flight-control terms force assignment to cluster c_42.

    Purpose in pipeline:
    Flight-control records are important domain-specific cases handled with
    deterministic overrides to improve precision.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts that only flight-control text triggers the override.
    """
    df = pd.DataFrame(
        {
            "text_norm": ["aileron cable fairlead due", "engine rough"],
            "seed_cluster": ["UNKNOWN", "UNKNOWN"],
        }
    )

    result = apply_flight_control_override(df)

    assert result.loc[0, "seed_cluster"] == "c_42"
    assert result.loc[1, "seed_cluster"] == "UNKNOWN"


def test_flight_control_override_for_rudder() -> None:
    """
    Verify that rudder-related text also triggers the flight-control override.

    Purpose in pipeline:
    This test ensures that the override covers multiple flight-control terms,
    not only one specific component.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts that rudder text maps to c_42.
    """
    df = pd.DataFrame(
        {
            "text_norm": ["rudder trim issue"],
            "seed_cluster": ["UNKNOWN"],
        }
    )

    result = apply_flight_control_override(df)

    assert result.loc[0, "seed_cluster"] == "c_42"


def test_oil_cooler_dirty_seed_goes_to_c32() -> None:
    """
    Verify that oil cooler dirty is assigned to c_32 during seed overrides.

    Purpose in pipeline:
    Selected dirty-related records are handled with deterministic overrides
    because they are easier to classify with explicit domain rules.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts correct oil-cooler override behavior.
    """
    df = pd.DataFrame(
        {
            "text_norm": ["oil cooler dirty"],
            "seed_cluster": ["UNKNOWN"],
        }
    )

    result = apply_dirty_seed_overrides(df)

    assert result.loc[0, "seed_cluster"] == "c_32"


def test_dirty_at_end_goes_to_appearance_cluster() -> None:
    """
    Verify that generic appearance-related dirty text maps to c_43.

    Purpose in pipeline:
    Cosmetic or exterior cleaning records should be separated from engine-
    related dirty cases.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts appearance-cluster assignment.
    """
    df = pd.DataFrame(
        {
            "text_norm": ["aircraft exterior dirty"],
            "seed_cluster": ["UNKNOWN"],
        }
    )

    result = apply_dirty_seed_overrides(df)

    assert result.loc[0, "seed_cluster"] == "c_43"


def test_engine_dirty_final_goes_to_engine_unspecified() -> None:
    """
    Verify that generic engine dirty text falls back to engine_general_unspecified.

    Purpose in pipeline:
    Final override logic handles difficult domain-specific cases after the
    hierarchical ML stage.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts final engine dirty fallback behavior.
    """
    df = pd.DataFrame(
        {
            "text_norm": ["engine dirty"],
            "cluster": ["UNKNOWN"],
        }
    )

    result = apply_final_dirty_overrides(df)

    assert result.loc[0, "cluster"] == "engine_general_unspecified"


def test_air_filter_dirty_goes_to_induction_unspecified() -> None:
    """
    Verify that air filter dirty maps to induction_intake_unspecified.

    Purpose in pipeline:
    Filter-related final overrides use subsystem-specific logic before applying
    generic engine fallbacks.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts correct air-filter override behavior.
    """
    df = pd.DataFrame(
        {
            "text_norm": ["air filter dirty"],
            "cluster": ["UNKNOWN"],
        }
    )

    result = apply_final_dirty_overrides(df)

    assert result.loc[0, "cluster"] == "induction_intake_unspecified"


def test_oil_filter_dirty_goes_to_oil_unspecified() -> None:
    """
    Verify that oil filter dirty maps to oil_system_unspecified.

    Purpose in pipeline:
    This test ensures that oil-system filter cases are handled before broader
    generic dirty fallback rules.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts correct oil-filter override behavior.
    """
    df = pd.DataFrame(
        {
            "text_norm": ["oil filter dirty"],
            "cluster": ["UNKNOWN"],
        }
    )

    result = apply_final_dirty_overrides(df)

    assert result.loc[0, "cluster"] == "oil_system_unspecified"


def test_fuel_filter_dirty_goes_to_fuel_unspecified() -> None:
    """
    Verify that fuel filter dirty maps to fuel_control_unspecified.

    Purpose in pipeline:
    Fuel filter records are a subsystem-specific case that should not be
    absorbed into generic engine or oil categories.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts correct fuel-filter override behavior.
    """
    df = pd.DataFrame(
        {
            "text_norm": ["fuel filter dirty"],
            "cluster": ["UNKNOWN"],
        }
    )

    result = apply_final_dirty_overrides(df)

    assert result.loc[0, "cluster"] == "fuel_control_unspecified"


def test_generic_filter_dirty_goes_to_engine_unspecified() -> None:
    """
    Verify that generic filter dirty falls back to engine_general_unspecified.

    Purpose in pipeline:
    When no subsystem-specific filter context is present, the final override
    should apply the generic engine fallback.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts generic filter fallback behavior.
    """
    df = pd.DataFrame(
        {
            "text_norm": ["filter dirty"],
            "cluster": ["UNKNOWN"],
        }
    )

    result = apply_final_dirty_overrides(df)

    assert result.loc[0, "cluster"] == "engine_general_unspecified"


def test_oil_cooler_dirty_has_priority_over_engine_dirty() -> None:
    """
    Verify that oil cooler dirty takes priority over the broader engine dirty rule.

    Purpose in pipeline:
    Override priority is important because the same record can satisfy both a
    general and a specific condition.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts that the specific oil-cooler rule wins.
    """
    df = pd.DataFrame(
        {
            "text_norm": ["engine oil cooler dirty"],
            "cluster": ["UNKNOWN"],
        }
    )

    result = apply_final_dirty_overrides(df)

    assert result.loc[0, "cluster"] == "c_32"


def test_tire_problem_unknown_goes_to_c44() -> None:
    """
    Verify that UNKNOWN records with tire in PROBLEM are reassigned to c_44.

    Purpose in pipeline:
    Tire-related records are handled with a final override because they can
    remain unresolved after earlier stages.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts tire override behavior for UNKNOWN rows.
    """
    df = pd.DataFrame(
        {
            "PROBLEM": ["NOSE TIRE FLAT"],
            "ACTION": ["replaced tire"],
            "cluster": ["UNKNOWN"],
        }
    )

    result = apply_final_tire_override(df)

    assert result.loc[0, "cluster"] == "c_44"


def test_tire_action_only_does_not_change_cluster() -> None:
    """
    Verify that tire mentions only in ACTION do not trigger the tire override.

    Purpose in pipeline:
    The final tire override is intentionally constrained to PROBLEM text so
    that procedural action statements do not create false positives.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts that cluster remains unchanged.
    """
    df = pd.DataFrame(
        {
            "PROBLEM": ["ENGINE ISSUE"],
            "ACTION": ["replaced tire"],
            "cluster": ["UNKNOWN"],
        }
    )

    result = apply_final_tire_override(df)

    assert result.loc[0, "cluster"] == "UNKNOWN"


def test_tire_does_not_override_existing_cluster() -> None:
    """
    Verify that the tire override does not overwrite an existing non-UNKNOWN label.

    Purpose in pipeline:
    Final override rules should be conservative and should not overwrite
    confident existing assignments.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts that the existing label is preserved.
    """
    df = pd.DataFrame(
        {
            "PROBLEM": ["NOSE TIRE FLAT"],
            "ACTION": ["replaced tire"],
            "cluster": ["c_0"],
        }
    )

    result = apply_final_tire_override(df)

    assert result.loc[0, "cluster"] == "c_0"


def test_build_joined_dataset_aggregates_and_joins() -> None:
    """
    Verify that the join pipeline correctly aggregates workorders and joins cluster output.

    Purpose in pipeline:
    This test validates the integration step that combines clustered records
    with aggregated workorder metadata.

    Key design decisions:
    - Workorder IDs are cleaned before joining.
    - Total time is aggregated per workorder.
    - Only overlapping workorders should appear in the final joined output.

    Parameters:
    None: This test does not accept external parameters.

    Returns:
    None: This test asserts correct aggregation and join behavior.
    """
    df = pd.DataFrame(
        {
            "WKO#": ["1001.0", "1001.0", "1002.0"],
            "Date_Opened": ["2024-01-01", "2024-01-02", "2024-02-01"],
            "Date_Closed": ["2024-01-03", "2024-01-04", "2024-02-02"],
            "Registration#": ["N123", "N123", "N456"],
            "Part#": ["P1", "P2", "P3"],
            "Total_Time": [2.0, 3.0, 5.0],
            "TSO": [10, 20, 30],
        }
    )

    dataset = pd.DataFrame(
        {
            "WKO#": ["1001", "1003"],
            "ATA_Code": [560, 320],
            "PROBLEM": ["TEST PROBLEM", "OTHER"],
            "DATE": ["2024-01-02", "2024-03-01"],
            "ACTION": ["TEST ACTION", "OTHER ACTION"],
            "cluster": ["c_0", "UNKNOWN"],
            "cluster_name": ["inspection", "unknown"],
        }
    )

    result = build_joined_dataset(df, dataset)

    assert len(result) == 1
    assert result.loc[0, "workorder"] == "1001"
    assert result.loc[0, "total_time"] == 5.0
    assert result.loc[0, "registration"] == "N123"
    assert result.loc[0, "ata_code"] == 560
    assert result.loc[0, "cluster"] == "c_0"
    assert result.loc[0, "cluster_name"] == "inspection"
