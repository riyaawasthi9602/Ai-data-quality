import re
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .quality_rules import QualityConfig


EMAIL_PATTERN = re.compile(
    r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
)

PHONE_PATTERN = re.compile(
    r"^\+?[0-9][0-9\s().-]{6,}$"
)


def _percentage(count: int, total: int) -> float:
    if total == 0:
        return 0.0

    return round(
        (count / total) * 100,
        2
    )


def _quality_level(score: float) -> str:

    if score >= 90:
        return "EXCELLENT"

    if score >= 75:
        return "GOOD"

    if score >= 60:
        return "FAIR"

    return "POOR"


def _blank_mask(series: pd.Series) -> pd.Series:
    """
    Detect null and blank/whitespace-only values.
    """

    mask = series.isna()

    if (
        pd.api.types.is_object_dtype(series)
        or
        pd.api.types.is_string_dtype(series)
    ):
        mask = (
            mask
            |
            series.astype("string")
            .str.strip()
            .eq("")
        )

    return mask


def _is_id_like(
    column: str,
    config: QualityConfig
) -> bool:

    if column in config.id_columns:
        return True

    pattern = (
        r"(^|[_\s-])"
        r"(id|identifier|key)"
        r"([_\s-]|$)"
    )

    return bool(
        re.search(
            pattern,
            column.lower()
        )
    )


def _detect_date_columns(
    df: pd.DataFrame,
    config: QualityConfig
) -> list[str]:

    date_columns = []

    for column in df.columns:

        if column in config.date_columns:
            date_columns.append(column)
            continue

        if pd.api.types.is_datetime64_any_dtype(
            df[column]
        ):
            date_columns.append(column)
            continue

        if (
            pd.api.types.is_object_dtype(
                df[column]
            )
            or
            pd.api.types.is_string_dtype(
                df[column]
            )
        ):

            non_null = df[column].dropna()

            if len(non_null) < 2:
                continue

            parsed = pd.to_datetime(
                non_null,
                errors="coerce",
                format="mixed"
            )

            if parsed.notna().mean() >= 0.80:
                date_columns.append(column)

    return date_columns


def _consistency_issues(
    series: pd.Series
) -> list[str]:

    values = series.dropna()

    if values.empty:
        return []

    issues = []

    if (
        pd.api.types.is_object_dtype(series)
        or
        pd.api.types.is_string_dtype(series)
    ):

        text = values.astype(str)

        stripped = text.str.strip()

        # Whitespace inconsistency
        if (text != stripped).any():
            issues.append("whitespace")

        # India / india / INDIA
        if (
            stripped.nunique()
            >
            stripped.str.lower().nunique()
        ):
            issues.append("capitalization")

        # Mixed numeric/string representation
        numeric = pd.to_numeric(
            stripped,
            errors="coerce"
        )

        if (
            0
            <
            numeric.notna().mean()
            <
            1
        ):
            issues.append("mixed_types")

        # Date-format inconsistency
        parsed_dates = pd.to_datetime(
            stripped,
            errors="coerce",
            format="mixed"
        )

        if parsed_dates.notna().mean() >= 0.80:

            formats = set()

            for value in stripped[
                parsed_dates.notna()
            ].head(1000):

                if re.search(
                    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
                    value
                ):

                    formats.add(
                        "slash"
                        if "/"
                        in value
                        else
                        "dash"
                    )

            if len(formats) > 1:
                issues.append(
                    "date_format"
                )

    return issues


def _validate_column(
    column: str,
    series: pd.Series,
    config: QualityConfig
):
    """
    Validate one column using configurable rules.
    """

    invalid = pd.Series(
        False,
        index=series.index
    )

    reasons = []

    # --------------------------------------------------
    # NUMERICAL VALIDATION
    # --------------------------------------------------

    if pd.api.types.is_numeric_dtype(series):

        numeric = pd.to_numeric(
            series,
            errors="coerce"
        )

        # Infinite values
        infinite_mask = pd.Series(
            np.isinf(
                numeric.to_numpy()
            ),
            index=series.index
        )

        if infinite_mask.any():

            invalid |= infinite_mask

            reasons.append(
                "infinite_value"
            )

        # Configurable minimum/maximum
        if column in config.numeric_ranges:

            rules = config.numeric_ranges[column]

            if "min" in rules:

                min_mask = (
                    series.notna()
                    &
                    numeric.notna()
                    &
                    (
                        numeric < rules["min"]
                    )
                )

                if min_mask.any():

                    invalid |= min_mask

                    reasons.append(
                        "below_minimum"
                    )

            if "max" in rules:

                max_mask = (
                    series.notna()
                    &
                    numeric.notna()
                    &
                    (
                        numeric > rules["max"]
                    )
                )

                if max_mask.any():

                    invalid |= max_mask

                    reasons.append(
                        "above_maximum"
                    )

    # --------------------------------------------------
    # CATEGORICAL VALIDATION
    # --------------------------------------------------

    if column in config.allowed_categories:

        allowed = {
            str(value)
            for value in config.allowed_categories[column]
        }

        category_mask = (
            series.notna()
            &
            ~series.astype(str).isin(
                allowed
            )
        )

        if category_mask.any():

            invalid |= category_mask

            reasons.append(
                "unexpected_category"
            )

    # --------------------------------------------------
    # EMAIL VALIDATION
    # --------------------------------------------------

    if column in config.email_columns:

        email_mask = (
            series.notna()
            &
            ~series.astype(str)
            .str.strip()
            .map(
                lambda value:
                bool(
                    EMAIL_PATTERN.fullmatch(
                        str(value)
                    )
                )
            )
        )

        if email_mask.any():

            invalid |= email_mask

            reasons.append(
                "malformed_email"
            )

    # --------------------------------------------------
    # PHONE VALIDATION
    # --------------------------------------------------

    if column in config.phone_columns:

        phone_mask = (
            series.notna()
            &
            ~series.astype(str)
            .str.strip()
            .map(
                lambda value:
                bool(
                    PHONE_PATTERN.fullmatch(
                        str(value)
                    )
                )
            )
        )

        if phone_mask.any():

            invalid |= phone_mask

            reasons.append(
                "malformed_phone"
            )

    # --------------------------------------------------
    # DATE VALIDATION
    # --------------------------------------------------

    if column in config.date_columns:

        parsed = pd.to_datetime(
            series,
            errors="coerce",
            format="mixed"
        )

        date_mask = (
            series.notna()
            &
            parsed.isna()
        )

        if date_mask.any():

            invalid |= date_mask

            reasons.append(
                "invalid_date"
            )

    # --------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------

    return (
        int(invalid.sum()),
        sorted(set(reasons))
    )
    


def _outlier_count(
    series: pd.Series,
    config: QualityConfig
) -> int:

    if not pd.api.types.is_numeric_dtype(
        series
    ):
        return 0

    values = (
        series
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
        .astype(float)
    )

    if (
        len(values) < 4
        or
        values.nunique() <= 1
    ):
        return 0

    # IQR
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    iqr_mask = pd.Series(
        False,
        index=values.index
    )

    if iqr > 0:

        lower = (
            q1
            -
            config.iqr_multiplier * iqr
        )

        upper = (
            q3
            +
            config.iqr_multiplier * iqr
        )

        iqr_mask = (
            (values < lower)
            |
            (values > upper)
        )

    # Z-score
    std = values.std(ddof=0)

    z_mask = pd.Series(
        False,
        index=values.index
    )

    if std > 0:

        z_mask = (
            (
                (
                    values
                    -
                    values.mean()
                ).abs()
                /
                std
            )
            >
            config.z_score_threshold
        )

    # Union so same outlier isn't counted twice
    return int(
        (
            iqr_mask
            |
            z_mask
        ).sum()
    )


def _timeliness(
    df: pd.DataFrame,
    date_columns: list[str],
    config: QualityConfig,
    reference_time=None
):

    if not date_columns:

        return {
            "available": False,
            "score": None,
            "status": "Not Available",
            "timestamp_column": None,
            "latest_timestamp": None,
            "data_age_days": None,
        }

    candidates = []

    for column in date_columns:

        parsed = pd.to_datetime(
            df[column],
            errors="coerce",
            format="mixed"
        )

        if parsed.notna().any():

            candidates.append(
                (
                    column,
                    parsed.max()
                )
            )

    if not candidates:

        return {
            "available": False,
            "score": None,
            "status": "Not Available",
            "timestamp_column": None,
            "latest_timestamp": None,
            "data_age_days": None,
        }

    column, latest = max(
        candidates,
        key=lambda item: item[1]
    )

    if reference_time is not None:

        now = pd.Timestamp(
            reference_time
        )

    else:

        now = pd.Timestamp.now(
            tz="UTC"
        )

    if (
        latest.tzinfo is None
        and
        now.tzinfo is not None
    ):
        latest = latest.tz_localize(
            "UTC"
        )

    age_days = max(
        0.0,
        (
            now - latest
        ).total_seconds()
        /
        86400
    )

    score = max(
        0.0,
        min(
            100.0,
            100.0
            *
            (
                1
                -
                age_days
                /
                config.max_age_days
            )
        )
    )

    return {
        "available": True,
        "score": round(
            score,
            2
        ),
        "status": (
            "Fresh"
            if age_days
            <= config.max_age_days
            else
            "Stale"
        ),
        "timestamp_column": column,
        "latest_timestamp":
            latest.isoformat(),
        "data_age_days":
            round(age_days, 2),
    }


class DataQualityEngine:
    """
    Enterprise-style rule-driven
    Data Quality Engine.
    """

    def __init__(
        self,
        config: Optional[
            QualityConfig
        ] = None
    ):

        self.config = (
            config
            or
            QualityConfig()
        )

        self.config.validate()

    def analyze(
        self,
        df: pd.DataFrame,
        reference_time=None
    ) -> Dict[str, Any]:

        if not isinstance(
            df,
            pd.DataFrame
        ):
            raise TypeError(
                "Expected a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "Dataset is empty."
            )

        if df.shape[1] == 0:
            raise ValueError(
                "Dataset has no columns."
            )

        rows = len(df)
        columns = len(df.columns)

        total_cells = (
            rows * columns
        )

        missing_cells = int(
            df.isna()
            .sum()
            .sum()
        )

        empty_string_cells = 0

        for column in df.columns:

            empty_string_cells += int(
                (
                    _blank_mask(
                        df[column]
                    )
                    &
                    df[column].notna()
                ).sum()
            )

        duplicate_rows = int(
            df.duplicated().sum()
        )

        duplicate_percentage = (
            _percentage(
                duplicate_rows,
                rows
            )
        )

        dataset_uniqueness_score = max(
            0.0,
            100.0
            -
            duplicate_percentage
        )

        date_columns = (
            _detect_date_columns(
                df,
                self.config
            )
        )

        column_scores = {}

        validity_issue_cells = 0
        consistency_issue_columns = 0
        outlier_cells = 0
        id_duplicate_cells = 0

        for column in df.columns:

            series = df[column]

            total_values = len(
                series
            )

            missing = int(
                series.isna().sum()
            )

            empty_strings = int(
                (
                    _blank_mask(series)
                    &
                    series.notna()
                ).sum()
            )

            completeness_score = max(
                0.0,
                100.0
                -
                _percentage(
                    missing,
                    total_values
                )
            )

            invalid_count, validity_reasons = (
                _validate_column(
                    column,
                    series,
                    self.config
                )
            )

            validity_issue_cells += (
                invalid_count
            )

            validity_score = max(
                0.0,
                100.0
                -
                _percentage(
                    invalid_count,
                    total_values
                )
            )

            consistency_reasons = (
                _consistency_issues(
                    series
                )
            )

            if consistency_reasons:
                consistency_issue_columns += 1

            consistency_score = max(
                0.0,
                100.0
                -
                (
                    self.config
                    .consistency_penalty_per_issue
                    *
                    len(
                        consistency_reasons
                    )
                )
            )

            outliers = _outlier_count(
                series,
                self.config
            )

            outlier_cells += outliers

            valid_numeric_values = (
                series
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .notna()
                .sum()
            )

            outlier_score = max(
                0.0,
                100.0
                -
                _percentage(
                    outliers,
                    valid_numeric_values
                    or total_values
                )
            )

            duplicate_id_count = 0

            uniqueness_applicable = (
                _is_id_like(
                    column,
                    self.config
                )
            )

            uniqueness_score = 100.0

            if uniqueness_applicable:

                duplicate_id_count = int(
                    series
                    .duplicated(
                        keep="first"
                    )
                    .sum()
                )

                id_duplicate_cells += (
                    duplicate_id_count
                )

                uniqueness_score = max(
                    0.0,
                    100.0
                    -
                    _percentage(
                        duplicate_id_count,
                        total_values
                    )
                )

            dimensions = {
                "completeness":
                    completeness_score,

                "validity":
                    validity_score,

                "consistency":
                    consistency_score,

                "uniqueness":
                    uniqueness_score,

                "outliers":
                    outlier_score,
            }

            applicable_weights = {
                name:
                    self.config.weights[name]
                for name in (
                    "completeness",
                    "validity",
                    "consistency",
                    "outliers",
                )
                if name
                in self.config.weights
            }

            if (
                uniqueness_applicable
                and
                "uniqueness"
                in self.config.weights
            ):

                applicable_weights[
                    "uniqueness"
                ] = self.config.weights[
                    "uniqueness"
                ]

            weight_sum = (
                sum(
                    applicable_weights.values()
                )
                or
                1.0
            )

            column_quality_score = (
                sum(
                    dimensions[name]
                    *
                    weight
                    for name, weight
                    in applicable_weights.items()
                )
                /
                weight_sum
            )

            column_scores[column] = {

                "total_values":
                    total_values,

                "missing_values":
                    missing,

                "missing_percentage":
                    _percentage(
                        missing,
                        total_values
                    ),

                "empty_string_count":
                    empty_strings,

                "empty_string_percentage":
                    _percentage(
                        empty_strings,
                        total_values
                    ),

                "completeness_score":
                    round(
                        completeness_score,
                        2
                    ),

                "invalid_count":
                    invalid_count,

                "invalid_percentage":
                    _percentage(
                        invalid_count,
                        total_values
                    ),

                "validity_score":
                    round(
                        validity_score,
                        2
                    ),

                "validity_issues":
                    validity_reasons,

                "consistency_issues":
                    consistency_reasons,

                "consistency_score":
                    round(
                        consistency_score,
                        2
                    ),

                "duplicate_id_count":
                    duplicate_id_count,

                "uniqueness_applicable":
                    uniqueness_applicable,

                "uniqueness_score":
                    round(
                        uniqueness_score,
                        2
                    ),

                "outlier_count":
                    outliers,

                "outlier_percentage":
                    _percentage(
                        outliers,
                        valid_numeric_values
                        or
                        total_values
                    ),

                "outlier_score":
                    round(
                        outlier_score,
                        2
                    ),

                "quality_score":
                    round(
                        column_quality_score,
                        2
                    ),
            }

        completeness_score = max(
            0.0,
            100.0
            -
            _percentage(
                missing_cells,
                total_cells
            )
        )

        validity_score = max(
            0.0,
            100.0
            -
            _percentage(
                validity_issue_cells,
                total_cells
            )
        )

        consistency_score = max(
            0.0,
            100.0
            -
            _percentage(
                consistency_issue_columns,
                columns
            )
        )

        outlier_score = max(
            0.0,
            100.0
            -
            _percentage(
                outlier_cells,
                total_cells
            )
        )

        uniqueness_score = (
            dataset_uniqueness_score
        )

        if id_duplicate_cells:

            id_columns = [
                column
                for column in df.columns
                if _is_id_like(
                    column,
                    self.config
                )
            ]

            denominator = sum(
                len(df[column])
                for column
                in id_columns
            )

            uniqueness_score = min(
                uniqueness_score,
                max(
                    0.0,
                    100.0
                    -
                    _percentage(
                        id_duplicate_cells,
                        denominator
                        or 1
                    )
                )
            )

        timeliness = _timeliness(
            df,
            date_columns,
            self.config,
            reference_time
        )

        dimensions = {

            "completeness":
                round(
                    completeness_score,
                    2
                ),

            "validity":
                round(
                    validity_score,
                    2
                ),

            "consistency":
                round(
                    consistency_score,
                    2
                ),

            "uniqueness":
                round(
                    uniqueness_score,
                    2
                ),

            "outliers":
                round(
                    outlier_score,
                    2
                ),

            "timeliness":
                timeliness["score"],
        }

        available_weights = {
            name: weight
            for name, weight
            in self.config.weights.items()
            if dimensions.get(name)
            is not None
        }

        weight_sum = (
            sum(
                available_weights.values()
            )
            or
            1.0
        )

        overall_score = (
            sum(
                float(
                    dimensions[name]
                )
                *
                weight
                for name, weight
                in available_weights.items()
            )
            /
            weight_sum
        )

        return {

            "overall_quality_score":
                round(
                    overall_score,
                    2
                ),

            "quality_level":
                _quality_level(
                    overall_score
                ),

            "dataset": {

                "rows":
                    rows,

                "columns":
                    columns,

                "total_cells":
                    total_cells,

                "missing_cells":
                    missing_cells,

                "missing_percentage":
                    _percentage(
                        missing_cells,
                        total_cells
                    ),

                "empty_string_cells":
                    empty_string_cells,

                "empty_string_percentage":
                    _percentage(
                        empty_string_cells,
                        total_cells
                    ),

                "duplicate_rows":
                    duplicate_rows,

                "duplicate_row_percentage":
                    duplicate_percentage,
            },

            "dimensions":
                dimensions,

            "timeliness":
                timeliness,

            "column_scores":
                column_scores,

            "config": {

                "weights":
                    self.config.weights,

                "z_score_threshold":
                    self.config
                    .z_score_threshold,

                "iqr_multiplier":
                    self.config
                    .iqr_multiplier,

                "max_age_days":
                    self.config
                    .max_age_days,
            },
        }


def analyze_quality(
    file_path: str,
    config: Optional[
        QualityConfig
    ] = None
):

    df = pd.read_csv(
        file_path
    )

    engine = DataQualityEngine(
        config
    )

    return engine.analyze(
        df
    )


# --------------------------------------------------
# BACKWARD COMPATIBILITY
# --------------------------------------------------

def calculate_quality(
    reference_path: str,
    current_path: Optional[str] = None
) -> float:

    """
    Backward-compatible wrapper.

    Existing code can continue calling:

        calculate_quality(
            reference_path,
            current_path
        )

    The new engine scores the dataset being
    analyzed rather than using a reference dataset.
    """

    target_path = (
        current_path
        or
        reference_path
    )

    result = analyze_quality(
        target_path
    )

    return result[
        "overall_quality_score"
    ]