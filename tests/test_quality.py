import numpy as np
import pandas as pd
import pytest

from src.quality import DataQualityEngine
from src.quality_rules import QualityConfig


def test_quality_detects_missing_invalid_and_outliers():

    df = pd.DataFrame({
        "Age": [
            25,
            30,
            -5,
            200,
            np.nan,
            30
        ],

        "Email": [
            "a@example.com",
            "bad",
            "b@example.com",
            "c@example.com",
            None,
            "d@example.com"
        ],

        "ID": [
            1,
            2,
            2,
            4,
            5,
            6
        ]
    })

    config = QualityConfig(

        numeric_ranges={
            "Age": {
                "min": 0,
                "max": 120
            }
        },

        email_columns=[
            "Email"
        ],

        id_columns=[
            "ID"
        ]
    )

    result = DataQualityEngine(
        config
    ).analyze(df)

    assert (
        result["dataset"]["missing_cells"]
        == 2
    )

    assert (
        result["column_scores"]["Age"]
        ["invalid_count"]
        >= 2
    )

    assert (
        result["column_scores"]["Email"]
        ["invalid_count"]
        >= 1
    )

    assert (
        result["column_scores"]["ID"]
        ["duplicate_id_count"]
        == 1
    )

    assert (
        result["column_scores"]["Age"]
        ["outlier_count"]
        >= 1
    )

    assert (
        0
        <=
        result["overall_quality_score"]
        <=
        100
    )


def test_consistency_detection():

    df = pd.DataFrame({
        "City": [
            "India",
            "india ",
            "INDIA",
            "Delhi"
        ]
    })

    result = DataQualityEngine().analyze(
        df
    )

    issues = (
        result["column_scores"]
        ["City"]
        ["consistency_issues"]
    )

    assert "whitespace" in issues

    assert "capitalization" in issues


def test_timeliness():

    df = pd.DataFrame({
        "created_at": [
            "2026-10-06",
            "2026-10-07"
        ]
    })

    config = QualityConfig(
        date_columns=[
            "created_at"
        ],
        max_age_days=7
    )

    result = DataQualityEngine(
        config
    ).analyze(
        df,
        reference_time=pd.Timestamp(
            "2026-10-07"
        )
    )

    assert (
        result["timeliness"]
        ["available"]
        is True
    )

    assert (
        result["timeliness"]
        ["data_age_days"]
        == 0
    )

    assert (
        result["dimensions"]
        ["timeliness"]
        == 100
    )


def test_timeliness_not_available():

    df = pd.DataFrame({
        "name": [
            "Riya",
            "Aman"
        ]
    })

    result = DataQualityEngine().analyze(
        df
    )

    assert (
        result["timeliness"]
        ["available"]
        is False
    )

    assert (
        result["dimensions"]
        ["timeliness"]
        is None
    )


def test_empty_dataset():

    with pytest.raises(
        ValueError,
        match="Dataset is empty"
    ):

        DataQualityEngine().analyze(
            pd.DataFrame()
        )