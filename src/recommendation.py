from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class RecommendationConfig:
    """
    Configuration for rule-based data quality recommendations.
    """

    high_drift_threshold: float = 0.25
    moderate_drift_threshold: float = 0.05

    high_anomaly_percentage: float = 10.0
    moderate_anomaly_percentage: float = 5.0

    low_quality_score: float = 70.0
    moderate_quality_score: float = 85.0

    def validate(self):

        if self.high_drift_threshold < 0:
            raise ValueError(
                "high_drift_threshold must be non-negative."
            )

        if self.moderate_drift_threshold < 0:
            raise ValueError(
                "moderate_drift_threshold must be non-negative."
            )

        if (
            self.high_drift_threshold
            < self.moderate_drift_threshold
        ):
            raise ValueError(
                "high_drift_threshold must be greater than "
                "or equal to moderate_drift_threshold."
            )

        if not 0 <= self.high_anomaly_percentage <= 100:
            raise ValueError(
                "high_anomaly_percentage must be between 0 and 100."
            )

        if not 0 <= self.moderate_anomaly_percentage <= 100:
            raise ValueError(
                "moderate_anomaly_percentage must be between 0 and 100."
            )

        if not 0 <= self.low_quality_score <= 100:
            raise ValueError(
                "low_quality_score must be between 0 and 100."
            )

        if not 0 <= self.moderate_quality_score <= 100:
            raise ValueError(
                "moderate_quality_score must be between 0 and 100."
            )

        if (
            self.low_quality_score
            > self.moderate_quality_score
        ):
            raise ValueError(
                "low_quality_score must be less than or equal "
                "to moderate_quality_score."
            )


class RecommendationEngine:
    """
    Rule-based recommendation engine.

    Converts quality, drift, anomaly and explainability
    results into actionable recommendations.
    """

    def __init__(self, config=None):

        self.config = (
            config
            if config is not None
            else RecommendationConfig()
        )

        self.config.validate()

    def _add_recommendation(
        self,
        recommendations: List[Dict[str, Any]],
        category: str,
        severity: str,
        title: str,
        message: str,
        action: str,
    ):

        recommendations.append(
            {
                "category": category,
                "severity": severity,
                "title": title,
                "message": message,
                "recommended_action": action,
            }
        )

    # ==================================================
    # QUALITY RECOMMENDATIONS
    # ==================================================

    def generate_quality_recommendations(
        self,
        quality_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        recommendations = []

        overall_score = quality_result.get(
            "overall_quality_score"
        )

        if overall_score is not None:

            if (
                overall_score
                < self.config.low_quality_score
            ):

                self._add_recommendation(
                    recommendations,
                    category="quality",
                    severity="high",
                    title="Low overall data quality",
                    message=(
                        f"Overall data quality score is "
                        f"{overall_score:.2f}/100."
                    ),
                    action=(
                        "Perform data cleaning and investigate "
                        "the quality dimensions with the lowest scores."
                    ),
                )

            elif (
                overall_score
                < self.config.moderate_quality_score
            ):

                self._add_recommendation(
                    recommendations,
                    category="quality",
                    severity="moderate",
                    title="Data quality needs attention",
                    message=(
                        f"Overall data quality score is "
                        f"{overall_score:.2f}/100."
                    ),
                    action=(
                        "Review completeness, validity, consistency, "
                        "uniqueness and outlier results."
                    ),
                )

        dimensions = quality_result.get(
            "dimensions",
            {},
        )

        for dimension, score in dimensions.items():

            if score is None:
                continue

            if (
                score
                < self.config.low_quality_score
            ):

                self._add_recommendation(
                    recommendations,
                    category="quality",
                    severity="high",
                    title=f"Poor {dimension}",
                    message=(
                        f"The {dimension} quality score is "
                        f"{score:.2f}/100."
                    ),
                    action=(
                        f"Investigate and improve the "
                        f"{dimension} issues in the dataset."
                    ),
                )

        return recommendations

    # ==================================================
    # DRIFT RECOMMENDATIONS
    # ==================================================

    def generate_drift_recommendations(
        self,
        drift_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        recommendations = []

        overall_severity = drift_result.get(
            "overall_severity",
            "low",
        )

        drift_percentage = drift_result.get(
            "drift_percentage",
            0.0,
        )

        affected_features = drift_result.get(
            "affected_features",
            0,
        )

        if overall_severity == "high":

            self._add_recommendation(
                recommendations,
                category="drift",
                severity="high",
                title="High data drift detected",
                message=(
                    f"{affected_features} feature(s) are affected "
                    f"by drift and {drift_percentage:.2f}% of "
                    "features are flagged."
                ),
                action=(
                    "Investigate changes in data distribution, "
                    "upstream data sources, sensors, collection "
                    "processes and patient population."
                ),
            )

        elif overall_severity == "moderate":

            self._add_recommendation(
                recommendations,
                category="drift",
                severity="moderate",
                title="Moderate data drift detected",
                message=(
                    f"{affected_features} feature(s) show "
                    "moderate distribution changes."
                ),
                action=(
                    "Monitor the affected features and investigate "
                    "whether the distribution change is expected."
                ),
            )

        missingness_percentage = drift_result.get(
            "missingness_drift_percentage",
            0.0,
        )

        if missingness_percentage > 0:

            self._add_recommendation(
                recommendations,
                category="missingness_drift",
                severity="moderate",
                title="Missingness drift detected",
                message=(
                    f"{missingness_percentage:.2f}% of features "
                    "show a change in missing-value rate."
                ),
                action=(
                    "Check the upstream ingestion pipeline, "
                    "data collection process and source-system "
                    "availability."
                ),
            )

        return recommendations

    # ==================================================
    # ANOMALY RECOMMENDATIONS
    # ==================================================

    def generate_anomaly_recommendations(
        self,
        anomaly_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        recommendations = []

        anomaly_percentage = anomaly_result.get(
            "anomaly_percentage",
            0.0,
        )

        strong_anomaly_count = anomaly_result.get(
            "strong_anomaly_count",
            0,
        )

        if (
            anomaly_percentage
            >= self.config.high_anomaly_percentage
        ):

            self._add_recommendation(
                recommendations,
                category="anomaly",
                severity="high",
                title="High anomaly rate",
                message=(
                    f"{anomaly_percentage:.2f}% of records "
                    "are flagged as anomalies."
                ),
                action=(
                    "Investigate anomalous records and check "
                    "for data collection, sensor, transformation "
                    "or pipeline problems."
                ),
            )

        elif (
            anomaly_percentage
            >= self.config.moderate_anomaly_percentage
        ):

            self._add_recommendation(
                recommendations,
                category="anomaly",
                severity="moderate",
                title="Moderate anomaly rate",
                message=(
                    f"{anomaly_percentage:.2f}% of records "
                    "are flagged as anomalies."
                ),
                action=(
                    "Review anomalous records and determine "
                    "whether they represent real events or "
                    "data-quality problems."
                ),
            )

        if strong_anomaly_count > 0:

            self._add_recommendation(
                recommendations,
                category="ensemble",
                severity="high",
                title="Strong anomalies detected",
                message=(
                    f"{strong_anomaly_count} records were flagged "
                    "by both anomaly detection models."
                ),
                action=(
                    "Prioritize these records for investigation "
                    "because multiple models agree that they "
                    "are unusual."
                ),
            )

        return recommendations

    # ==================================================
    # EXPLAINABILITY RECOMMENDATIONS
    # ==================================================

    def generate_explainability_recommendations(
        self,
        explanations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        recommendations = []

        if not explanations:
            return recommendations

        feature_counts = {}

        for explanation in explanations:

            significant_features = explanation.get(
                "significant_features",
                [],
            )

            for feature_info in significant_features:

                feature = feature_info.get(
                    "feature"
                )

                if feature:

                    feature_counts[feature] = (
                        feature_counts.get(
                            feature,
                            0,
                        )
                        + 1
                    )

        if feature_counts:

            sorted_features = sorted(
                feature_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            top_feature, count = sorted_features[0]

            self._add_recommendation(
                recommendations,
                category="explainability",
                severity="moderate",
                title="Frequently anomalous feature",
                message=(
                    f"'{top_feature}' is a significant contributing "
                    f"feature in {count} explained anomaly record(s)."
                ),
                action=(
                    f"Investigate the '{top_feature}' feature for "
                    "sensor errors, unusual values, collection "
                    "issues or distribution changes."
                ),
            )

        return recommendations

    # ==================================================
    # COMPLETE RECOMMENDATION ENGINE
    # ==================================================

    def generate_recommendations(
        self,
        quality_result=None,
        drift_result=None,
        anomaly_result=None,
        explanations=None,
    ) -> Dict[str, Any]:

        recommendations = []

        if quality_result is not None:

            recommendations.extend(
                self.generate_quality_recommendations(
                    quality_result
                )
            )

        if drift_result is not None:

            recommendations.extend(
                self.generate_drift_recommendations(
                    drift_result
                )
            )

        if anomaly_result is not None:

            recommendations.extend(
                self.generate_anomaly_recommendations(
                    anomaly_result
                )
            )

        if explanations is not None:

            recommendations.extend(
                self.generate_explainability_recommendations(
                    explanations
                )
            )

        severity_order = {
            "high": 0,
            "moderate": 1,
            "low": 2,
        }

        recommendations.sort(
            key=lambda item: severity_order.get(
                item["severity"],
                3,
            )
        )

        high_count = sum(
            1
            for item in recommendations
            if item["severity"] == "high"
        )

        moderate_count = sum(
            1
            for item in recommendations
            if item["severity"] == "moderate"
        )

        return {
            "recommendations": recommendations,
            "total_recommendations": len(
                recommendations
            ),
            "high_priority": high_count,
            "moderate_priority": moderate_count,
        }