from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

import hashlib
import os
import shutil
import tempfile

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

from src.monitoring import (
    HistoricalMonitor,
    MonitoringConfig,
)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="AI Data Quality & Drift Detection Platform",
    description=(
        "AI-based enterprise data quality, drift detection, "
        "anomaly detection, explainability, recommendations "
        "and historical data health monitoring platform."
    ),
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# DIRECTORIES
# =========================================================

UPLOAD_DIR = "uploads"
BASELINE_DIR = "baselines"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(BASELINE_DIR, exist_ok=True)


# =========================================================
# HOME ENDPOINT
# =========================================================

@app.get("/")
def home():
    return {
        "message": (
            "AI Data Quality & Drift Detection API is running"
        ),
        "version": "1.0.0",
        "status": "healthy",
    }


# =========================================================
# HEALTH ENDPOINT
# =========================================================

@app.get("/health")
def api_health():
    return {
        "status": "healthy",
        "service": (
            "AI Data Quality & Drift Detection Platform"
        ),
    }


# =========================================================
# DATASET ID
# =========================================================

def create_dataset_id(
    df: pd.DataFrame,
) -> str:
    """
    Create a stable dataset ID from the dataset schema.
    """

    schema = "|".join(
        [
            f"{column}:{df[column].dtype}"
            for column in df.columns
        ]
    )

    return hashlib.md5(
        schema.encode("utf-8")
    ).hexdigest()[:12]


# =========================================================
# BASELINE PATH
# =========================================================

def get_baseline_path(
    dataset_id: str,
) -> str:

    return os.path.join(
        BASELINE_DIR,
        f"{dataset_id}.csv",
    )


# =========================================================
# SAVE UPLOADED FILE
# =========================================================

def save_uploaded_file(
    file: UploadFile,
) -> str:
    """
    Save the uploaded CSV safely inside uploads/.
    """

    safe_filename = os.path.basename(
        file.filename or "uploaded.csv"
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        safe_filename,
    )

    with open(
        file_path,
        "wb",
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer,
        )

    return file_path


# =========================================================
# DATASET INFORMATION
# =========================================================

def get_dataset_info(
    df: pd.DataFrame,
) -> dict:

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
    }


# =========================================================
# RUN COMPLETE AI ANALYSIS
# =========================================================

def run_ai_analysis(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> dict:
    """
    Run the complete analysis pipeline:

    Quality
        ↓
    Drift
        ↓
    Autoencoder
        ↓
    Isolation Forest
        ↓
    Ensemble
        ↓
    Explainability
        ↓
    Recommendations
        ↓
    Health Score
        ↓
    Historical Monitoring
    """

    # =====================================================
    # TEMPORARY FILES
    # =====================================================

    reference_temp = tempfile.NamedTemporaryFile(
        suffix=".csv",
        delete=False,
    )

    current_temp = tempfile.NamedTemporaryFile(
        suffix=".csv",
        delete=False,
    )

    reference_temp.close()
    current_temp.close()

    reference_path = reference_temp.name
    current_path = current_temp.name

    try:

        # -------------------------------------------------
        # Save DataFrames temporarily
        # -------------------------------------------------

        reference_df.to_csv(
            reference_path,
            index=False,
        )

        current_df.to_csv(
            current_path,
            index=False,
        )

        # =================================================
        # 1. ADVANCED DATA QUALITY
        # =================================================

        quality_result = analyze_quality(
            reference_path,
        )

        # =================================================
        # 2. ADVANCED DRIFT DETECTION
        # =================================================

        drift_result = detect_drift(
            reference_path,
            current_path,
        )

        # =================================================
        # 3. AUTOENCODER
        # =================================================

        autoencoder = (
            AutoencoderAnomalyDetector()
        )

        autoencoder.load()

        autoencoder_result = (
            autoencoder.predict(
                current_df,
            )
        )

        # =================================================
        # 4. ISOLATION FOREST
        # =================================================

        isolation_forest = (
            IsolationForestAnomalyDetector(
                config=IsolationForestConfig(
                    n_estimators=200,
                    contamination=0.05,
                ),
                exclude_columns=[
                    "Patient Number"
                ],
            )
        )

        isolation_forest.load()

        isolation_result = (
            isolation_forest.predict(
                current_df,
            )
        )

        # =================================================
        # 5. ENSEMBLE ANOMALY DETECTION
        # =================================================

        ensemble = (
            EnsembleAnomalyDetector(
                EnsembleConfig(
                    autoencoder_weight=0.5,
                    isolation_forest_weight=0.5,
                    threshold=0.5,
                )
            )
        )

        ensemble_result = (
            ensemble.predict(

                autoencoder_result[
                    "results"
                ][
                    "reconstruction_error"
                ].values,

                isolation_result[
                    "results"
                ][
                    "isolation_score"
                ].values,

                autoencoder_result[
                    "results"
                ][
                    "is_anomaly"
                ].values,

                isolation_result[
                    "results"
                ][
                    "is_anomaly"
                ].values,
            )
        )

        # =================================================
        # 6. EXPLAINABILITY
        # =================================================

        explainer = AnomalyExplainer()

        explanations = (
            explainer.explain_dataset(
                reference_df=reference_df,
                current_df=current_df,
                ensemble_result=ensemble_result,
                only_anomalies=True,
            )
        )

        # -------------------------------------------------
        # API RESPONSE LIMIT
        # -------------------------------------------------
        #
        # We still calculate ALL explanations internally.
        # Only a limited number are returned through the API
        # to prevent huge Swagger/browser responses.
        #

        MAX_EXPLANATIONS_IN_RESPONSE = 10

        top_explanations = explanations[
            :MAX_EXPLANATIONS_IN_RESPONSE
        ]

        # =================================================
        # 7. RECOMMENDATION ENGINE
        # =================================================

        recommendation_engine = (
            RecommendationEngine()
        )

        recommendation_result = (
            recommendation_engine
            .generate_recommendations(
                quality_result=quality_result,
                drift_result=drift_result,
                anomaly_result=ensemble_result,
                explanations=explanations,
            )
        )

        # =================================================
        # 8. DATA HEALTH SCORE
        # =================================================

        health_calculator = (
            DataHealthScore()
        )

        health_result = (
            health_calculator
            .calculate_health_score(
                quality_result=quality_result,
                drift_result=drift_result,
                anomaly_result=ensemble_result,
            )
        )

        # =================================================
        # 9. HISTORICAL MONITORING
        # =================================================

        monitor = HistoricalMonitor(
            MonitoringConfig(
                history_path=(
                    "data/analysis_history.json"
                )
            )
        )

        monitoring_result = (
            monitor.record_run(
                result={
                    "quality": quality_result,
                    "drift": drift_result,
                    "ensemble": ensemble_result,
                    "health_score": health_result,
                    "recommendations": (
                        recommendation_result
                    ),
                },
                reference_path=reference_path,
                current_path=current_path,
            )
        )

        # =================================================
        # 10. RETURN COMPLETE RESULT
        # =================================================

        return {

            # -------------------------------------------------
            # DATA QUALITY
            # -------------------------------------------------

            "data_quality": quality_result,

            # -------------------------------------------------
            # DRIFT
            # -------------------------------------------------

            "drift": drift_result,

            # -------------------------------------------------
            # AUTOENCODER
            # -------------------------------------------------

            "autoencoder": {
                "anomaly_count": (
                    autoencoder_result[
                        "anomaly_count"
                    ]
                ),

                "anomaly_percentage": (
                    autoencoder_result[
                        "anomaly_percentage"
                    ]
                ),

                "threshold": (
                    autoencoder_result[
                        "threshold"
                    ]
                ),
            },

            # -------------------------------------------------
            # ISOLATION FOREST
            # -------------------------------------------------

            "isolation_forest": {
                "anomaly_count": (
                    isolation_result[
                        "anomaly_count"
                    ]
                ),

                "anomaly_percentage": (
                    isolation_result[
                        "anomaly_percentage"
                    ]
                ),
            },

            # -------------------------------------------------
            # ENSEMBLE
            # -------------------------------------------------

            "ensemble": {

                "anomaly_count": (
                    ensemble_result[
                        "anomaly_count"
                    ]
                ),

                "anomaly_percentage": (
                    ensemble_result[
                        "anomaly_percentage"
                    ]
                ),

                "strong_anomaly_count": (
                    ensemble_result[
                        "strong_anomaly_count"
                    ]
                ),

                "possible_anomaly_count": (
                    ensemble_result[
                        "possible_anomaly_count"
                    ]
                ),

                "agreement_percentage": (
                    ensemble_result[
                        "agreement_percentage"
                    ]
                ),
            },

            # -------------------------------------------------
            # EXPLAINABILITY
            # -------------------------------------------------

            "explainability": {

                # Total number of anomalies explained
                "explained_anomalies": len(
                    explanations
                ),

                # Number actually returned to API client
                "returned_explanations": len(
                    top_explanations
                ),

                # Only first 10 are sent to browser
                "explanations": top_explanations,
            },

            # -------------------------------------------------
            # RECOMMENDATIONS
            # -------------------------------------------------

            "recommendations": (
                recommendation_result
            ),

            # -------------------------------------------------
            # HEALTH SCORE
            # -------------------------------------------------

            "health_score": health_result,

            # -------------------------------------------------
            # HISTORICAL MONITORING
            # -------------------------------------------------

            "monitoring": monitoring_result,
        }

    finally:

        # =================================================
        # CLEAN TEMPORARY FILES
        # =================================================

        if os.path.exists(
            reference_path
        ):

            os.remove(
                reference_path
            )

        if os.path.exists(
            current_path
        ):

            os.remove(
                current_path
            )


# =========================================================
# ANALYZE DATASET ENDPOINT
# =========================================================

@app.post("/analyze")
async def analyze_dataset(
    file: UploadFile = File(...),
    dataset_name: str = Form(
        "default_dataset"
    ),
):

    # =====================================================
    # 1. VALIDATE FILE
    # =====================================================

    if not file.filename:

        return {
            "status": "ERROR",
            "message": "No file was provided.",
        }

    if not file.filename.lower().endswith(
        ".csv"
    ):

        return {
            "status": "ERROR",
            "message": (
                "Only CSV files are supported."
            ),
        }

    # =====================================================
    # 2. SAVE UPLOADED FILE
    # =====================================================

    try:

        file_path = (
            save_uploaded_file(file)
        )

    except Exception as exc:

        return {
            "status": "ERROR",
            "message": (
                f"Unable to save uploaded file: "
                f"{str(exc)}"
            ),
        }

    # =====================================================
    # 3. READ CSV
    # =====================================================

    try:

        df = pd.read_csv(
            file_path
        )

    except Exception as exc:

        return {
            "status": "ERROR",
            "message": (
                f"Unable to read CSV: "
                f"{str(exc)}"
            ),
        }

    # =====================================================
    # 4. EMPTY DATASET CHECK
    # =====================================================

    if df.empty:

        return {
            "status": "ERROR",
            "message": (
                "Uploaded dataset is empty."
            ),
        }

    # =====================================================
    # 5. DATASET ID
    # =====================================================

    dataset_id = create_dataset_id(
        df
    )

    baseline_path = (
        get_baseline_path(
            dataset_id
        )
    )

    dataset_info = (
        get_dataset_info(df)
    )

    # =====================================================
    # 6. FIRST UPLOAD
    # =====================================================

    if not os.path.exists(
        baseline_path
    ):

        df.to_csv(
            baseline_path,
            index=False,
        )

        return {

            "filename": file.filename,

            "dataset_name": dataset_name,

            "dataset_id": dataset_id,

            "dataset": dataset_info,

            "baseline": {

                "created": True,

                "message": (
                    "Baseline created successfully. "
                    "Future uploads with the same schema "
                    "will be compared against this baseline."
                ),
            },

            "status": (
                "BASELINE_CREATED"
            ),
        }

    # =====================================================
    # 7. LOAD EXISTING BASELINE
    # =====================================================

    try:

        baseline_df = pd.read_csv(
            baseline_path
        )

    except Exception as exc:

        return {
            "status": "ERROR",
            "message": (
                f"Unable to read baseline: "
                f"{str(exc)}"
            ),
        }

    # =====================================================
    # 8. RUN COMPLETE AI ANALYSIS
    # =====================================================

    try:

        analysis_result = (
            run_ai_analysis(
                reference_df=baseline_df,
                current_df=df,
            )
        )

    except Exception as exc:

        return {
            "status": "ERROR",
            "message": (
                f"Analysis failed: "
                f"{str(exc)}"
            ),
        }

    # =====================================================
    # 9. DETERMINE OVERALL STATUS
    # =====================================================

    health_result = (
        analysis_result[
            "health_score"
        ]
    )

    health_level = (
        health_result.get(
            "health_level"
        )
    )

    if health_level == "EXCELLENT":

        status = "GOOD"

    elif health_level in (
        "GOOD",
        "NEEDS ATTENTION",
    ):

        status = "WARNING"

    else:

        status = "CRITICAL"

    # =====================================================
    # 10. FINAL RESPONSE
    # =====================================================

    return {

        "filename": file.filename,

        "dataset_name": dataset_name,

        "dataset_id": dataset_id,

        "dataset": dataset_info,

        "baseline": {

            "created": False,

            "message": (
                "Existing baseline used "
                "for analysis."
            ),
        },

        # -------------------------------------------------
        # DATA QUALITY
        # -------------------------------------------------

        "data_quality": (
            analysis_result[
                "data_quality"
            ]
        ),

        # -------------------------------------------------
        # DRIFT
        # -------------------------------------------------

        "drift": (
            analysis_result[
                "drift"
            ]
        ),

        # -------------------------------------------------
        # ANOMALY DETECTION
        # -------------------------------------------------

        "anomaly_detection": {

            "autoencoder": (
                analysis_result[
                    "autoencoder"
                ]
            ),

            "isolation_forest": (
                analysis_result[
                    "isolation_forest"
                ]
            ),

            "ensemble": (
                analysis_result[
                    "ensemble"
                ]
            ),
        },

        # -------------------------------------------------
        # EXPLAINABILITY
        # -------------------------------------------------

        "explainability": (
            analysis_result[
                "explainability"
            ]
        ),

        # -------------------------------------------------
        # RECOMMENDATIONS
        # -------------------------------------------------

        "recommendations": (
            analysis_result[
                "recommendations"
            ]
        ),

        # -------------------------------------------------
        # HEALTH SCORE
        # -------------------------------------------------

        "health_score": (
            analysis_result[
                "health_score"
            ]
        ),

        # -------------------------------------------------
        # MONITORING
        # -------------------------------------------------

        "monitoring": (
            analysis_result[
                "monitoring"
            ]
        ),

        # -------------------------------------------------
        # OVERALL STATUS
        # -------------------------------------------------

        "status": status,
    }