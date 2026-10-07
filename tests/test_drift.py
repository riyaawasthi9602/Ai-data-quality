import pandas as pd

from src.drift import (
    calculate_psi,
    detect_drift
)


def test_psi_no_drift():

    reference = pd.Series([
        10, 20, 30, 40, 50
    ])

    current = pd.Series([
        10, 20, 30, 40, 50
    ])

    psi = calculate_psi(
        reference,
        current
    )

    assert psi == 0.0


def test_psi_detects_shift():

    reference = pd.Series([
        10, 20, 30, 40, 50
    ])

    current = pd.Series([
        100, 110, 120, 130, 140
    ])

    psi = calculate_psi(
        reference,
        current
    )

    assert psi > 0.25


def test_missingness_drift(tmp_path):

    reference = pd.DataFrame({
        "Age": [20, 30, 40, 50, 60]
    })

    current = pd.DataFrame({
        "Age": [
            20,
            None,
            None,
            None,
            60
        ]
    })

    reference_path = (
        tmp_path / "reference.csv"
    )

    current_path = (
        tmp_path / "current.csv"
    )

    reference.to_csv(
        reference_path,
        index=False
    )

    current.to_csv(
        current_path,
        index=False
    )

    result = detect_drift(
        reference_path,
        current_path
    )

    missingness = result[
        "missingness"
    ][0]

    assert (
        missingness[
            "drift"
        ]
        is True
    )


def test_drift_detects_numerical_shift(
    tmp_path
):

    reference = pd.DataFrame({
        "Age": [
            20,
            21,
            22,
            23,
            24,
            25
        ]
    })

    current = pd.DataFrame({
        "Age": [
            80,
            81,
            82,
            83,
            84,
            85
        ]
    })

    reference_path = (
        tmp_path / "reference.csv"
    )

    current_path = (
        tmp_path / "current.csv"
    )

    reference.to_csv(
        reference_path,
        index=False
    )

    current.to_csv(
        current_path,
        index=False
    )

    result = detect_drift(
        reference_path,
        current_path
    )

    assert (
        result["drift_percentage"]
        == 100.0
    )

    assert (
        result["affected_features"]
        == 1
    )

    assert (
        result["overall_severity"]
        == "high"
    )


def test_categorical_no_drift(
    tmp_path
):

    reference = pd.DataFrame({
        "City": [
            "Delhi",
            "Delhi",
            "Mumbai",
            "Mumbai"
        ]
    })

    current = pd.DataFrame({
        "City": [
            "Delhi",
            "Delhi",
            "Mumbai",
            "Mumbai"
        ]
    })

    reference_path = (
        tmp_path / "reference.csv"
    )

    current_path = (
        tmp_path / "current.csv"
    )

    reference.to_csv(
        reference_path,
        index=False
    )

    current.to_csv(
        current_path,
        index=False
    )

    result = detect_drift(
        reference_path,
        current_path
    )

    categorical_result = result[
        "results"
    ][0]

    assert (
        categorical_result["drift"]
        is False
    )

    assert (
        categorical_result[
            "feature_severity"
        ]
        == "low"
    )