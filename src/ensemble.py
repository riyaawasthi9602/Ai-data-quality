import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class EnsembleConfig:
    autoencoder_weight: float = 0.5
    isolation_forest_weight: float = 0.5
    threshold: float = 0.5

    def validate(self):
        if self.autoencoder_weight < 0:
            raise ValueError(
                "autoencoder_weight must be non-negative."
            )

        if self.isolation_forest_weight < 0:
            raise ValueError(
                "isolation_forest_weight must be non-negative."
            )

        total_weight = (
            self.autoencoder_weight
            + self.isolation_forest_weight
        )

        if total_weight <= 0:
            raise ValueError(
                "At least one ensemble weight must be greater than 0."
            )

        if not 0 <= self.threshold <= 1:
            raise ValueError(
                "threshold must be between 0 and 1."
            )


def normalize_scores(scores):
    """
    Normalize anomaly scores to [0, 1].
    Higher value means more anomalous.
    """

    values = np.asarray(scores, dtype=float)

    if len(values) == 0:
        return np.array([], dtype=float)

    min_value = np.min(values)
    max_value = np.max(values)

    if np.isclose(max_value, min_value):
        return np.zeros(len(values), dtype=float)

    normalized = (
        (values - min_value)
        / (max_value - min_value)
    )

    return np.clip(normalized, 0.0, 1.0)


class EnsembleAnomalyDetector:

    def __init__(self, config=None):
        self.config = (
            config
            if config is not None
            else EnsembleConfig()
        )

        self.config.validate()

    def combine_scores(
        self,
        autoencoder_scores,
        isolation_forest_scores
    ):
        autoencoder_scores = np.asarray(
            autoencoder_scores,
            dtype=float
        )

        isolation_forest_scores = np.asarray(
            isolation_forest_scores,
            dtype=float
        )

        if len(autoencoder_scores) != len(
            isolation_forest_scores
        ):
            raise ValueError(
                "Autoencoder and Isolation Forest "
                "score arrays must have the same length."
            )

        normalized_autoencoder = normalize_scores(
            autoencoder_scores
        )

        normalized_isolation_forest = normalize_scores(
            isolation_forest_scores
        )

        total_weight = (
            self.config.autoencoder_weight
            + self.config.isolation_forest_weight
        )

        ae_weight = (
            self.config.autoencoder_weight
            / total_weight
        )

        if_weight = (
            self.config.isolation_forest_weight
            / total_weight
        )

        ensemble_scores = (
            ae_weight * normalized_autoencoder
            + if_weight * normalized_isolation_forest
        )

        return {
            "autoencoder_score": normalized_autoencoder,
            "isolation_forest_score": normalized_isolation_forest,
            "ensemble_score": ensemble_scores
        }

    def predict(
        self,
        autoencoder_scores,
        isolation_forest_scores,
        autoencoder_flags=None,
        isolation_forest_flags=None
    ):

        score_results = self.combine_scores(
            autoencoder_scores,
            isolation_forest_scores
        )

        ensemble_scores = score_results["ensemble_score"]

        # Default score-based ensemble decision
        score_flags = (
            ensemble_scores >= self.config.threshold
        )

        result = pd.DataFrame({
            "autoencoder_score":
                score_results["autoencoder_score"],

            "isolation_forest_score":
                score_results["isolation_forest_score"],

            "ensemble_score":
                ensemble_scores,

            "ensemble_anomaly":
                score_flags
        })

        # Add individual model flags when available
        if autoencoder_flags is not None:
            ae_flags = np.asarray(
                autoencoder_flags,
                dtype=bool
            )

            if len(ae_flags) != len(result):
                raise ValueError(
                    "Autoencoder flags must have the same "
                    "length as the score arrays."
                )

            result["autoencoder_anomaly"] = ae_flags

        else:
            ae_flags = None

        if isolation_forest_flags is not None:
            if_flags = np.asarray(
                isolation_forest_flags,
                dtype=bool
            )

            if len(if_flags) != len(result):
                raise ValueError(
                    "Isolation Forest flags must have the "
                    "same length as the score arrays."
                )

            result["isolation_forest_anomaly"] = if_flags

        else:
            if_flags = None

        # Strong / possible / normal classification
        if ae_flags is not None and if_flags is not None:

            result["both_models_anomaly"] = (
                ae_flags & if_flags
            )

            result["model_agreement"] = (
                ae_flags == if_flags
            )

            result["anomaly_category"] = np.select(
                [
                    ae_flags & if_flags,
                    ae_flags | if_flags
                ],
                [
                    "strong_anomaly",
                    "possible_anomaly"
                ],
                default="normal"
            )

            # Final ensemble decision:
            # both models agree OR score is very high
            result["ensemble_anomaly"] = (
                (ae_flags & if_flags)
                | (
                    ensemble_scores
                    >= self.config.threshold
                )
            )

        anomaly_count = int(
            result["ensemble_anomaly"].sum()
        )

        total_rows = len(result)

        anomaly_percentage = (
            anomaly_count / total_rows * 100
            if total_rows > 0
            else 0.0
        )

        agreement_count = None
        agreement_percentage = None

        if "model_agreement" in result.columns:

            agreement_count = int(
                result["model_agreement"].sum()
            )

            agreement_percentage = (
                agreement_count / total_rows * 100
                if total_rows > 0
                else 0.0
            )

        strong_anomaly_count = None
        possible_anomaly_count = None

        if "anomaly_category" in result.columns:

            strong_anomaly_count = int(
                (
                    result["anomaly_category"]
                    == "strong_anomaly"
                ).sum()
            )

            possible_anomaly_count = int(
                (
                    result["anomaly_category"]
                    == "possible_anomaly"
                ).sum()
            )

        return {
            "results": result,

            "anomaly_count":
                anomaly_count,

            "anomaly_percentage":
                float(anomaly_percentage),

            "agreement_count":
                agreement_count,

            "agreement_percentage":
                (
                    float(agreement_percentage)
                    if agreement_percentage is not None
                    else None
                ),

            "strong_anomaly_count":
                strong_anomaly_count,

            "possible_anomaly_count":
                possible_anomaly_count
        }