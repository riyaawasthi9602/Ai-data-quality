from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class HealthScoreConfig:
    """
    Configuration for the unified data health score.

    The final score combines:
    - Data quality
    - Data drift
    - Anomaly detection
    """

    quality_weight: float = 0.40
    drift_weight: float = 0.30
    anomaly_weight: float = 0.30

    def validate(self):
        weights = [
            self.quality_weight,
            self.drift_weight,
            self.anomaly_weight,
        ]

        if any(weight < 0 for weight in weights):
            raise ValueError(
                "Health score weights must be non-negative."
            )

        total = sum(weights)

        if total <= 0:
            raise ValueError(
                "Health score weights must have a positive total."
            )


class DataHealthScore:
    """
    Calculates a unified 0-100 data health score.
    """

    def __init__(self, config=None):
        self.config = (
            config
            if config is not None
            else HealthScoreConfig()
        )

        self.config.validate()

    # ==================================================
    # QUALITY SCORE
    # ==================================================

    def calculate_quality_score(
        self,
        quality_result: Dict[str, Any],
    ) -> float:

        score = quality_result.get(
            "overall_quality_score"
        )

        if score is None:
            raise ValueError(
                "overall_quality_score is missing "
                "from quality result."
            )

        return max(
            0.0,
            min(100.0, float(score)),
        )

    # ==================================================
    # DRIFT SCORE
    # ==================================================

    def calculate_drift_score(
        self,
        drift_result: Dict[str, Any],
    ) -> float:

        drift_percentage = float(
            drift_result.get(
                "drift_percentage",
                0.0,
            )
        )

        # 0% drift = 100 score
        # 100% drift = 0 score

        score = 100.0 - drift_percentage

        return max(
            0.0,
            min(100.0, score),
        )

    # ==================================================
    # ANOMALY SCORE
    # ==================================================

    def calculate_anomaly_score(
        self,
        anomaly_result: Dict[str, Any],
    ) -> float:

        anomaly_percentage = float(
            anomaly_result.get(
                "anomaly_percentage",
                0.0,
            )
        )

        # 0% anomalies = 100 score
        # 100% anomalies = 0 score

        score = 100.0 - anomaly_percentage

        return max(
            0.0,
            min(100.0, score),
        )

    # ==================================================
    # HEALTH LEVEL
    # ==================================================

    def get_health_level(
        self,
        score: float,
    ) -> str:

        if score >= 90:
            return "EXCELLENT"

        if score >= 75:
            return "GOOD"

        if score >= 60:
            return "NEEDS ATTENTION"

        if score >= 40:
            return "POOR"

        return "CRITICAL"

    # ==================================================
    # COMPLETE HEALTH SCORE
    # ==================================================

    def calculate_health_score(
        self,
        quality_result: Dict[str, Any],
        drift_result: Dict[str, Any],
        anomaly_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        quality_score = (
            self.calculate_quality_score(
                quality_result
            )
        )

        drift_score = (
            self.calculate_drift_score(
                drift_result
            )
        )

        anomaly_score = (
            self.calculate_anomaly_score(
                anomaly_result
            )
        )

        total_weight = (
            self.config.quality_weight
            + self.config.drift_weight
            + self.config.anomaly_weight
        )

        health_score = (
            (
                quality_score
                * self.config.quality_weight
            )
            + (
                drift_score
                * self.config.drift_weight
            )
            + (
                anomaly_score
                * self.config.anomaly_weight
            )
        ) / total_weight

        health_score = round(
            max(
                0.0,
                min(
                    100.0,
                    health_score,
                ),
            ),
            2,
        )

        return {
            "health_score": health_score,
            "health_level": self.get_health_level(
                health_score
            ),
            "components": {
                "quality_score": round(
                    quality_score,
                    2,
                ),
                "drift_score": round(
                    drift_score,
                    2,
                ),
                "anomaly_score": round(
                    anomaly_score,
                    2,
                ),
            },
            "weights": {
                "quality": self.config.quality_weight,
                "drift": self.config.drift_weight,
                "anomaly": self.config.anomaly_weight,
            },
        }