from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
import os
import shutil
import joblib
import json
import hashlib

from tensorflow.keras.models import load_model

from src.drift import detect_drift
from src.anomaly import prepare_data


app = FastAPI(
    title="AI Data Quality & Drift Detection"
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
# EXISTING AI MODEL
# =========================================================

MODEL_PATH = "models/autoencoder.keras"
SCALER_PATH = "models/scaler.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "AI Data Quality API is running"
    }


# =========================================================
# DATA QUALITY
# =========================================================

def calculate_quality(df):

    total_cells = df.shape[0] * df.shape[1]

    missing_cells = int(
        df.isnull().sum().sum()
    )

    missing_percentage = (
        missing_cells / total_cells
    ) * 100 if total_cells > 0 else 0

    duplicate_rows = int(
        df.duplicated().sum()
    )

    duplicate_percentage = (
        duplicate_rows / len(df)
    ) * 100 if len(df) > 0 else 0

    missing_score = max(
        0,
        100 - missing_percentage
    )

    duplicate_score = max(
        0,
        100 - duplicate_percentage
    )

    quality_score = (
        missing_score * 0.6 +
        duplicate_score * 0.4
    )

    return {
        "score": round(
            float(quality_score),
            2
        ),
        "missing_values": missing_cells,
        "missing_percentage": round(
            float(missing_percentage),
            2
        ),
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": round(
            float(duplicate_percentage),
            2
        )
    }


# =========================================================
# DATASET ID
# =========================================================

def create_dataset_id(df):

    schema = "|".join(
        [
            f"{column}:{df[column].dtype}"
            for column in df.columns
        ]
    )

    return hashlib.md5(
        schema.encode()
    ).hexdigest()[:12]


# =========================================================
# BASELINE PATH
# =========================================================

def get_baseline_path(dataset_id):

    return os.path.join(
        BASELINE_DIR,
        f"{dataset_id}.csv"
    )


# =========================================================
# AI ANOMALY DETECTION
# =========================================================

def detect_ai_anomalies(df, baseline_df):

    current_data = prepare_data(df)
    baseline_data = prepare_data(baseline_df)

    common_columns = [
        column
        for column in baseline_data.columns
        if column in current_data.columns
    ]

    if len(common_columns) == 0:

        return {
            "error": "No compatible numerical features found."
        }

    current_data = current_data[
        common_columns
    ]

    baseline_data = baseline_data[
        common_columns
    ]

    # Check model compatibility

    if (
        current_data.shape[1]
        != scaler.n_features_in_
    ):

        return {
            "error":
                "Dataset schema is incompatible with "
                "the currently trained AI model.",
            "expected_features":
                int(scaler.n_features_in_),
            "received_features":
                int(current_data.shape[1]),
            "features":
                list(current_data.columns)
        }

    # Scale data

    X_current = scaler.transform(
        current_data
    )

    X_baseline = scaler.transform(
        baseline_data
    )

    # Current reconstruction

    reconstructed_current = model.predict(
        X_current,
        verbose=0
    )

    current_error = np.mean(
        np.square(
            X_current -
            reconstructed_current
        ),
        axis=1
    )

    # Baseline reconstruction

    reconstructed_baseline = model.predict(
        X_baseline,
        verbose=0
    )

    baseline_error = np.mean(
        np.square(
            X_baseline -
            reconstructed_baseline
        ),
        axis=1
    )

    # Threshold

    threshold = np.percentile(
        baseline_error,
        95
    )

    anomalies = (
        current_error >
        threshold
    )

    anomaly_count = int(
        anomalies.sum()
    )

    anomaly_percentage = (
        anomaly_count /
        len(anomalies)
    ) * 100 if len(anomalies) > 0 else 0

    return {

        "anomalies":
            anomaly_count,

        "percentage":
            round(
                float(anomaly_percentage),
                2
            ),

        "threshold":
            float(threshold)
    }


# =========================================================
# ANALYZE DATASET
# =========================================================

@app.post("/analyze")
async def analyze_dataset(

    file: UploadFile = File(...),

    dataset_name: str = Form(
        "default_dataset"
    )
):

    # -----------------------------------------------------
    # Save uploaded file
    # -----------------------------------------------------

    safe_filename = os.path.basename(
        file.filename
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        safe_filename
    )

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    # -----------------------------------------------------
    # Read dataset
    # -----------------------------------------------------

    try:

        df = pd.read_csv(
            file_path
        )

    except Exception as e:

        return {
            "error":
                f"Unable to read CSV: {str(e)}"
        }

    # -----------------------------------------------------
    # Dataset ID
    # -----------------------------------------------------

    dataset_id = create_dataset_id(
        df
    )

    baseline_path = get_baseline_path(
        dataset_id
    )

    # -----------------------------------------------------
    # DATA QUALITY
    # -----------------------------------------------------

    quality = calculate_quality(
        df
    )

    # =====================================================
    # FIRST UPLOAD
    # =====================================================

    if not os.path.exists(
        baseline_path
    ):

        df.to_csv(
            baseline_path,
            index=False
        )

        return {

            "filename":
                file.filename,

            "dataset_name":
                dataset_name,

            "dataset": {

                "rows":
                    len(df),

                "columns":
                    len(df.columns),

                "column_names":
                    list(df.columns)
            },

            "data_quality":
                quality,

            "baseline": {

                "created":
                    True,

                "message":
                    "Baseline created successfully."
            },

            "drift": {

                "status":
                    "NOT_AVAILABLE",

                "percentage":
                    0,

                "columns":
                    []
            },

            "ai_anomaly": {

                "status":
                    "BASELINE_CREATED",

                "message":
                    "AI anomaly detection will be performed on subsequent uploads."
            },

            "status":
                "BASELINE_CREATED"
        }

    # =====================================================
    # FUTURE UPLOAD
    # =====================================================

    baseline_df = pd.read_csv(
        baseline_path
    )

    # -----------------------------------------------------
    # DRIFT
    # -----------------------------------------------------

    drift_results, drift_percentage = (
        detect_drift(
            baseline_path,
            file_path
        )
    )

    # -----------------------------------------------------
    # AI ANOMALY
    # -----------------------------------------------------

    ai_result = detect_ai_anomalies(
        df,
        baseline_df
    )

    # -----------------------------------------------------
    # STATUS
    # -----------------------------------------------------

    if "error" in ai_result:

        status = "WARNING"

    else:

        anomaly_percentage = (
            ai_result["percentage"]
        )

        if (
            anomaly_percentage < 5
            and drift_percentage < 10
        ):

            status = "GOOD"

        elif (
            anomaly_percentage < 15
            and drift_percentage < 30
        ):

            status = "WARNING"

        else:

            status = "CRITICAL"

    # =====================================================
    # RESPONSE
    # =====================================================

    return {

        "filename":
            file.filename,

        "dataset_name":
            dataset_name,

        "dataset": {

            "rows":
                len(df),

            "columns":
                len(df.columns),

            "column_names":
                list(df.columns)
        },

        "data_quality":
            quality,

        "baseline": {

            "created":
                False,

            "message":
                "Existing baseline used for analysis."
        },

        "drift": {

            "status":
                "ANALYZED",

            "percentage":
                round(
                    float(drift_percentage),
                    2
                ),

            "columns":
                drift_results
        },

        "ai_anomaly":
            ai_result,

        "status":
            status
    }