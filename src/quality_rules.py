from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class QualityConfig:
    """
    Configurable rules and scoring weights
    for the Data Quality Engine.
    """

    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "completeness": 0.25,
            "validity": 0.20,
            "consistency": 0.20,
            "uniqueness": 0.15,
            "outliers": 0.10,
            "timeliness": 0.10,
        }
    )

    # Consistency scoring
    consistency_penalty_per_issue: float = 10.0

    # Numerical validation rules
    numeric_ranges: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )

    # Allowed categorical values
    allowed_categories: Dict[str, List[Any]] = field(
        default_factory=dict
    )

    # Columns that should behave like IDs
    id_columns: List[str] = field(
        default_factory=list
    )

    # Special validation columns
    email_columns: List[str] = field(
        default_factory=list
    )

    phone_columns: List[str] = field(
        default_factory=list
    )

    date_columns: List[str] = field(
        default_factory=list
    )

    # Timeliness
    max_age_days: float = 7.0

    # Outlier configuration
    z_score_threshold: float = 3.0
    iqr_multiplier: float = 1.5

    def validate(self):
        if not self.weights:
            raise ValueError(
                "Quality weights cannot be empty."
            )

        if any(
            value < 0
            for value in self.weights.values()
        ):
            raise ValueError(
                "Quality weights cannot be negative."
            )

        if sum(self.weights.values()) <= 0:
            raise ValueError(
                "Quality weights must have a positive total."
            )

        if self.max_age_days <= 0:
            raise ValueError(
                "max_age_days must be greater than zero."
            )

        if self.z_score_threshold <= 0:
            raise ValueError(
                "z_score_threshold must be greater than zero."
            )

        if self.iqr_multiplier <= 0:
            raise ValueError(
                "iqr_multiplier must be greater than zero."
            )