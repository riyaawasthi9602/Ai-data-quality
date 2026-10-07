from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ExplainabilityConfig:
    """
    Configuration for anomaly explainability.
    """

    z_score_threshold: float = 3.0
    top_features: int = 3

    def validate(self):
        """
        Validate configuration values.
        """

        if self.z_score_threshold <= 0:
            raise ValueError(
                "z_score_threshold must be greater than 0."
            )

        if self.top_features <= 0:
            raise ValueError(
                "top_features must be greater than 0."
            )

        return True

    def __post_init__(self):
        self.validate()


class AnomalyExplainer:
    """
    Explain anomalies using feature-level deviation
    from reference dataset statistics.

    Z-score:

        z = |current_value - reference_mean| / reference_std

    Higher z-score means the current observation is
    farther from the reference distribution.
    """

    def __init__(
        self,
        config: Optional[ExplainabilityConfig] = None,
        exclude_columns: Optional[list] = None,
    ):
        self.config = config or ExplainabilityConfig()

        # Identifier columns should not be used for
        # meaningful anomaly explanations.
        self.exclude_columns = (
            exclude_columns
            if exclude_columns is not None
            else ["Patient Number"]
        )

    # =========================================================
    # PREPARE CURRENT ROW
    # =========================================================

    def _prepare_current_row(self, current_row):
        """
        Convert a Series or one-row DataFrame into a Series.
        """

        if isinstance(current_row, pd.DataFrame):

            if len(current_row) != 1:
                raise ValueError(
                    "current_row DataFrame must contain exactly one row."
                )

            return current_row.iloc[0]

        if isinstance(current_row, pd.Series):
            return current_row

        raise TypeError(
            "current_row must be a pandas Series or one-row DataFrame."
        )

    # =========================================================
    # FEATURE DEVIATION
    # =========================================================

    def calculate_feature_deviation(
        self,
        reference_df: pd.DataFrame,
        current_row,
    ):
        """
        Calculate feature-level z-score deviations.

        Returns:
            list of dictionaries containing:

            - feature
            - value
            - reference_mean
            - reference_std
            - z_score
            - is_significant
        """

        if not isinstance(reference_df, pd.DataFrame):
            raise TypeError(
                "reference_df must be a pandas DataFrame."
            )

        current_series = self._prepare_current_row(
            current_row
        )

        # Select numeric reference columns only.
        numeric_reference = reference_df.select_dtypes(
            include=[np.number]
        ).copy()

        # Remove excluded identifier columns.
        columns_to_use = [
            column
            for column in numeric_reference.columns
            if column not in self.exclude_columns
        ]

        numeric_reference = numeric_reference[
            columns_to_use
        ]

        explanations = []

        for feature in numeric_reference.columns:

            # Feature must exist in current row.
            if feature not in current_series.index:
                continue

            # Convert reference values safely.
            reference_values = pd.to_numeric(
                numeric_reference[feature],
                errors="coerce",
            )

            # Remove infinity values.
            reference_values = reference_values.replace(
                [np.inf, -np.inf],
                np.nan,
            )

            reference_values = reference_values.dropna()

            if reference_values.empty:
                continue

            # Convert current value safely.
            current_value = pd.to_numeric(
                pd.Series(
                    [current_series[feature]]
                ),
                errors="coerce",
            ).iloc[0]

            if pd.isna(current_value):
                continue

            if np.isinf(current_value):
                continue

            # Reference statistics.
            reference_mean = float(
                reference_values.mean()
            )

            reference_std = float(
                reference_values.std(ddof=0)
            )

            # Avoid division by zero.
            if (
                reference_std == 0
                or np.isnan(reference_std)
            ):
                z_score = 0.0

            else:
                z_score = abs(
                    float(current_value)
                    - reference_mean
                ) / reference_std

            is_significant = bool(
                z_score
                >= self.config.z_score_threshold
            )

            explanations.append(
                {
                    "feature": feature,
                    "value": float(current_value),
                    "reference_mean": reference_mean,
                    "reference_std": reference_std,
                    "z_score": float(z_score),
                    "is_significant": is_significant,
                }
            )

        # Highest deviation first.
        explanations.sort(
            key=lambda item: item["z_score"],
            reverse=True,
        )

        return explanations

    # =========================================================
    # EXPLAIN ONE ROW
    # =========================================================

    def explain_row(
        self,
        reference_df: pd.DataFrame,
        current_row,
        anomaly_category: str = "normal",
        autoencoder_score: Optional[float] = None,
        isolation_forest_score: Optional[float] = None,
        ensemble_score: Optional[float] = None,
    ):
        """
        Generate an explanation for one observation.
        """

        feature_deviations = (
            self.calculate_feature_deviation(
                reference_df=reference_df,
                current_row=current_row,
            )
        )

        # Get top N features.
        top_features = feature_deviations[
            : self.config.top_features
        ]

        # Get statistically significant features.
        significant_features = [
            feature
            for feature in feature_deviations
            if feature["is_significant"]
        ]

        significant_feature_names = [
            feature["feature"]
            for feature in significant_features
        ]

        # =====================================================
        # SUMMARY
        # =====================================================

        if anomaly_category == "strong_anomaly":

            if significant_feature_names:

                summary = (
                    "Strong anomaly detected. "
                    "The observation shows significant "
                    "deviation in: "
                    + ", ".join(
                        significant_feature_names
                    )
                    + "."
                )

            else:

                summary = (
                    "Strong anomaly detected, but no "
                    "individual feature exceeded the "
                    "configured z-score threshold."
                )

        elif anomaly_category == "possible_anomaly":

            if significant_feature_names:

                summary = (
                    "Possible anomaly detected. "
                    "The most significant deviations "
                    "are in: "
                    + ", ".join(
                        significant_feature_names
                    )
                    + "."
                )

            else:

                summary = (
                    "Possible anomaly detected, but no "
                    "individual feature exceeded the "
                    "configured z-score threshold."
                )

        else:

            summary = (
                "No strong anomaly: anomaly detection "
                "models did not identify this observation "
                "as a strong anomaly."
            )

        return {
            "summary": summary,
            "anomaly_category": anomaly_category,

            "top_features": top_features,

            "significant_features": (
                significant_features
            ),

            "significant_feature_count": len(
                significant_features
            ),

            "autoencoder_score": (
                float(autoencoder_score)
                if autoencoder_score is not None
                else None
            ),

            "isolation_forest_score": (
                float(isolation_forest_score)
                if isolation_forest_score is not None
                else None
            ),

            "ensemble_score": (
                float(ensemble_score)
                if ensemble_score is not None
                else None
            ),
        }

    # =========================================================
    # EXPLAIN DATASET
    # =========================================================

    def explain_dataset(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        ensemble_results: Optional[pd.DataFrame] = None,
        ensemble_result=None,
        only_anomalies: bool = True,
    ):
        """
        Generate explanations for observations in a dataset.

        Supports both:

            ensemble_results=dataframe

        and:

            ensemble_result={"results": dataframe}

        This preserves compatibility with the existing
        project tests and API.
        """

        # -----------------------------------------------------
        # Validate reference dataset
        # -----------------------------------------------------

        if not isinstance(
            reference_df,
            pd.DataFrame,
        ):
            raise TypeError(
                "reference_df must be a pandas DataFrame."
            )

        # -----------------------------------------------------
        # Validate current dataset
        # -----------------------------------------------------

        if not isinstance(
            current_df,
            pd.DataFrame,
        ):
            raise TypeError(
                "current_df must be a pandas DataFrame."
            )

        # -----------------------------------------------------
        # Backward compatibility
        # -----------------------------------------------------

        if (
            ensemble_results is None
            and ensemble_result is not None
        ):

            # Existing tests may pass:
            #
            # ensemble_result={
            #     "results": DataFrame(...)
            # }

            if isinstance(
                ensemble_result,
                dict,
            ):

                ensemble_results = (
                    ensemble_result.get(
                        "results"
                    )
                )

            elif isinstance(
                ensemble_result,
                pd.DataFrame,
            ):

                ensemble_results = (
                    ensemble_result
                )

        # -----------------------------------------------------
        # Ensure ensemble result exists
        # -----------------------------------------------------

        if ensemble_results is None:
            raise ValueError(
                "ensemble_results or ensemble_result "
                "must be provided."
            )

        # -----------------------------------------------------
        # Validate ensemble DataFrame
        # -----------------------------------------------------

        if not isinstance(
            ensemble_results,
            pd.DataFrame,
        ):
            raise TypeError(
                "ensemble_results must be a pandas DataFrame."
            )

        # -----------------------------------------------------
        # Validate row count
        # -----------------------------------------------------

        if len(current_df) != len(
            ensemble_results
        ):
            raise ValueError(
                "current_df and ensemble_results "
                "must contain the same number of rows."
            )

        explanations = []

        # =====================================================
        # PROCESS EACH ROW
        # =====================================================

        for position in range(
            len(current_df)
        ):

            current_row = (
                current_df.iloc[position]
            )

            ensemble_row = (
                ensemble_results.iloc[position]
            )

            # -------------------------------------------------
            # Anomaly category
            # -------------------------------------------------

            anomaly_category = (
                ensemble_row.get(
                    "anomaly_category",
                    "normal",
                )
            )

            # -------------------------------------------------
            # Autoencoder score
            # -------------------------------------------------

            autoencoder_score = (
                ensemble_row.get(
                    "autoencoder_score",
                    ensemble_row.get(
                        "reconstruction_error",
                        None,
                    ),
                )
            )

            # -------------------------------------------------
            # Isolation Forest score
            # -------------------------------------------------

            isolation_forest_score = (
                ensemble_row.get(
                    "isolation_forest_score",
                    ensemble_row.get(
                        "isolation_score",
                        None,
                    ),
                )
            )

            # -------------------------------------------------
            # Ensemble score
            # -------------------------------------------------

            ensemble_score = (
                ensemble_row.get(
                    "ensemble_score",
                    None,
                )
            )

            # -------------------------------------------------
            # Ensemble anomaly flag
            # -------------------------------------------------

            ensemble_anomaly = (
                ensemble_row.get(
                    "ensemble_anomaly",
                    False,
                )
            )

            ensemble_anomaly = bool(
                ensemble_anomaly
            )

            # -------------------------------------------------
            # Skip normal observations
            # -------------------------------------------------

            if (
                only_anomalies
                and not ensemble_anomaly
            ):
                continue

            # -------------------------------------------------
            # Generate row explanation
            # -------------------------------------------------

            explanation = self.explain_row(
                reference_df=reference_df,
                current_row=current_row,
                anomaly_category=(
                    anomaly_category
                ),
                autoencoder_score=(
                    autoencoder_score
                ),
                isolation_forest_score=(
                    isolation_forest_score
                ),
                ensemble_score=(
                    ensemble_score
                ),
            )

            # -------------------------------------------------
            # Add original row index
            # -------------------------------------------------

            explanation["row_index"] = int(
                current_df.index[position]
            )

            explanations.append(
                explanation
            )

        return explanations