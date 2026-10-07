import pandas as pd

from src.quality import analyze_quality
from src.drift import detect_drift

from src.anomaly import AutoencoderAnomalyDetector
from src.isolation_forest import (
    IsolationForestAnomalyDetector,
    IsolationForestConfig,
)

from src.ensemble import (
    EnsembleAnomalyDetector,
    EnsembleConfig,
)

from src.explainability import AnomalyExplainer
from src.recommendation import RecommendationEngine
from src.health_score import DataHealthScore

# Historical Monitoring
from src.monitoring import (
    HistoricalMonitor,
    MonitoringConfig,
)


def run_complete_analysis(
    reference_path,
    current_path,
):
    """
    Run the complete AI-based data quality,
    drift, anomaly, explainability,
    recommendation, health-score and
    historical-monitoring pipeline.
    """

    # ==================================================
    # 1. LOAD DATASETS
    # ==================================================

    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    if reference_df.empty:
        raise ValueError("Reference dataset is empty.")

    if current_df.empty:
        raise ValueError("Current dataset is empty.")

    # ==================================================
    # 2. DATA QUALITY ANALYSIS
    # ==================================================

    quality_result = analyze_quality(
        reference_path
    )

    # ==================================================
    # 3. DRIFT DETECTION
    # ==================================================

    drift_result = detect_drift(
        reference_path,
        current_path,
    )

    # ==================================================
    # 4. AUTOENCODER
    # ==================================================

    autoencoder = AutoencoderAnomalyDetector()

    autoencoder.load()

    autoencoder_result = autoencoder.predict(
        current_df
    )

    # ==================================================
    # 5. ISOLATION FOREST
    # ==================================================

    isolation_forest = IsolationForestAnomalyDetector(
        config=IsolationForestConfig(
            n_estimators=200,
            contamination=0.05,
        ),
        exclude_columns=["Patient Number"],
    )

    isolation_forest.load()

    isolation_result = isolation_forest.predict(
        current_df
    )

    # ==================================================
    # 6. ENSEMBLE ANOMALY DETECTION
    # ==================================================

    ensemble = EnsembleAnomalyDetector(
        EnsembleConfig(
            autoencoder_weight=0.5,
            isolation_forest_weight=0.5,
            threshold=0.5,
        )
    )

    ensemble_result = ensemble.predict(
        autoencoder_result["results"][
            "reconstruction_error"
        ].values,

        isolation_result["results"][
            "isolation_score"
        ].values,

        autoencoder_result["results"][
            "is_anomaly"
        ].values,

        isolation_result["results"][
            "is_anomaly"
        ].values,
    )

    # ==================================================
    # 7. EXPLAINABILITY
    # ==================================================

    explainer = AnomalyExplainer()

    explanations = explainer.explain_dataset(
        reference_df=reference_df,
        current_df=current_df,
        ensemble_result=ensemble_result,
        only_anomalies=True,
    )

    # ==================================================
    # 8. RECOMMENDATION ENGINE
    # ==================================================

    recommendation_engine = RecommendationEngine()

    recommendation_result = (
        recommendation_engine.generate_recommendations(
            quality_result=quality_result,
            drift_result=drift_result,
            anomaly_result=ensemble_result,
            explanations=explanations,
        )
    )

    # ==================================================
    # 9. UNIFIED DATA HEALTH SCORE
    # ==================================================

    health_calculator = DataHealthScore()

    health_result = (
        health_calculator.calculate_health_score(
            quality_result=quality_result,
            drift_result=drift_result,
            anomaly_result=ensemble_result,
        )
    )

    # ==================================================
    # 10. HISTORICAL MONITORING
    # ==================================================

    monitor = HistoricalMonitor(
        MonitoringConfig(
            history_path="data/analysis_history.json"
        )
    )

    monitoring_result = monitor.record_run(
        result={
            "quality": quality_result,
            "drift": drift_result,
            "ensemble": ensemble_result,
            "health_score": health_result,
            "recommendations": recommendation_result,
        },
        reference_path=reference_path,
        current_path=current_path,
    )

    # ==================================================
    # 11. FINAL RESULT
    # ==================================================

    return {
        "quality": quality_result,

        "drift": drift_result,

        "autoencoder": {
            "anomaly_count": autoencoder_result[
                "anomaly_count"
            ],
            "anomaly_percentage": autoencoder_result[
                "anomaly_percentage"
            ],
            "threshold": autoencoder_result[
                "threshold"
            ],
        },

        "isolation_forest": {
            "anomaly_count": isolation_result[
                "anomaly_count"
            ],
            "anomaly_percentage": isolation_result[
                "anomaly_percentage"
            ],
        },

        "ensemble": {
            "anomaly_count": ensemble_result[
                "anomaly_count"
            ],
            "anomaly_percentage": ensemble_result[
                "anomaly_percentage"
            ],
            "strong_anomaly_count": ensemble_result[
                "strong_anomaly_count"
            ],
            "possible_anomaly_count": ensemble_result[
                "possible_anomaly_count"
            ],
            "agreement_percentage": ensemble_result[
                "agreement_percentage"
            ],
        },

        "explainability": {
            "explained_anomalies": len(
                explanations
            ),
            "explanations": explanations,
        },

        "recommendations": recommendation_result,

        "health_score": health_result,

        "monitoring": monitoring_result,
    }


# ======================================================
# MAIN PROGRAM
# ======================================================

if __name__ == "__main__":

    result = run_complete_analysis(
        "data/reference.csv",
        "data/current.csv",
    )

    print("\n" + "=" * 60)
    print("COMPLETE DATA ANALYSIS")
    print("=" * 60)

    # ==================================================
    # QUALITY
    # ==================================================

    print("\nQUALITY")

    print(
        "Overall Score:",
        result["quality"].get(
            "overall_quality_score"
        ),
    )

    print(
        "Quality Level:",
        result["quality"].get(
            "quality_level"
        ),
    )

    # ==================================================
    # DRIFT
    # ==================================================

    print("\nDRIFT")

    print(
        "Drift Percentage:",
        result["drift"].get(
            "drift_percentage"
        ),
    )

    print(
        "Affected Features:",
        result["drift"].get(
            "affected_features"
        ),
    )

    print(
        "Overall Severity:",
        result["drift"].get(
            "overall_severity"
        ),
    )

    print(
        "Missingness Drift:",
        result["drift"].get(
            "missingness_drift_percentage"
        ),
    )

    # ==================================================
    # ANOMALIES
    # ==================================================

    print("\nANOMALIES")

    print(
        "Autoencoder:",
        result["autoencoder"][
            "anomaly_count"
        ],
    )

    print(
        "Autoencoder Percentage:",
        result["autoencoder"][
            "anomaly_percentage"
        ],
    )

    print(
        "Isolation Forest:",
        result["isolation_forest"][
            "anomaly_count"
        ],
    )

    print(
        "Isolation Forest Percentage:",
        result["isolation_forest"][
            "anomaly_percentage"
        ],
    )

    print(
        "Ensemble:",
        result["ensemble"][
            "anomaly_count"
        ],
    )

    print(
        "Ensemble Percentage:",
        result["ensemble"][
            "anomaly_percentage"
        ],
    )

    print(
        "Strong Anomalies:",
        result["ensemble"][
            "strong_anomaly_count"
        ],
    )

    print(
        "Possible Anomalies:",
        result["ensemble"][
            "possible_anomaly_count"
        ],
    )

    print(
        "Model Agreement:",
        result["ensemble"][
            "agreement_percentage"
        ],
    )

    # ==================================================
    # EXPLAINABILITY
    # ==================================================

    print("\nEXPLAINABILITY")

    print(
        "Explained Anomalies:",
        result["explainability"][
            "explained_anomalies"
        ],
    )

    # ==================================================
    # DATA HEALTH SCORE
    # ==================================================

    print("\nDATA HEALTH SCORE")

    print(
        "Health Score:",
        result["health_score"][
            "health_score"
        ],
    )

    print(
        "Health Level:",
        result["health_score"][
            "health_level"
        ],
    )

    print(
        "Quality Component:",
        result["health_score"][
            "components"
        ]["quality_score"],
    )

    print(
        "Drift Component:",
        result["health_score"][
            "components"
        ]["drift_score"],
    )

    print(
        "Anomaly Component:",
        result["health_score"][
            "components"
        ]["anomaly_score"],
    )

    # ==================================================
    # HISTORICAL MONITORING
    # ==================================================

    print("\nHISTORICAL MONITORING")

    print(
        "Run ID:",
        result["monitoring"][
            "run_id"
        ],
    )

    print(
        "Timestamp:",
        result["monitoring"][
            "timestamp"
        ],
    )

    print(
        "History File:",
        "data/analysis_history.json",
    )

    # ==================================================
    # RECOMMENDATIONS
    # ==================================================

    print("\nRECOMMENDATIONS")

    recommendations = result[
        "recommendations"
    ]["recommendations"]

    print(
        "Total Recommendations:",
        len(recommendations),
    )

    for number, recommendation in enumerate(
        recommendations,
        start=1,
    ):

        print(
            f"\n{number}. "
            f"[{recommendation['severity'].upper()}] "
            f"{recommendation['title']}"
        )

        print(
            "   Message:",
            recommendation["message"],
        )

        print(
            "   Action:",
            recommendation[
                "recommended_action"
            ],
        )

    print("\n" + "=" * 60)