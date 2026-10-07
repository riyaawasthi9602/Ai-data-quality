import os
import joblib
import numpy as np
import pandas as pd

from dataclasses import dataclass, asdict

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class IsolationForestConfig:

    contamination: float = 0.05
    n_estimators: int = 200
    max_samples: str = "auto"
    random_state: int = 42

    model_path: str = "models/isolation_forest.pkl"

    scaler_path: str = (
        "models/isolation_forest_scaler.pkl"
    )

    metadata_path: str = (
        "models/isolation_forest_metadata.pkl"
    )

    def validate(self):

        if not 0 < self.contamination <= 0.5:
            raise ValueError(
                "contamination must be between 0 and 0.5."
            )

        if self.n_estimators <= 0:
            raise ValueError(
                "n_estimators must be greater than 0."
            )


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_isolation_data(
    df,
    exclude_columns=None
):
    """
    Prepare numerical data for Isolation Forest.
    """

    if exclude_columns is None:
        exclude_columns = []

    data = df.select_dtypes(
        include=[np.number]
    ).copy()

    columns_to_remove = [
        column
        for column in exclude_columns
        if column in data.columns
    ]

    if columns_to_remove:
        data = data.drop(
            columns=columns_to_remove
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    for column in data.columns:

        median_value = data[column].median()

        if pd.isna(median_value):
            median_value = 0.0

        data[column] = data[column].fillna(
            median_value
        )

    return data


# ============================================================
# MAIN DETECTOR
# ============================================================

class IsolationForestAnomalyDetector:

    def __init__(
        self,
        config=None,
        exclude_columns=None
    ):

        self.config = (
            config
            if config is not None
            else IsolationForestConfig()
        )

        self.config.validate()

        self.exclude_columns = (
            exclude_columns
            if exclude_columns is not None
            else []
        )

        self.model = None
        self.scaler = None
        self.feature_columns = []


    # ========================================================
    # TRAIN
    # ========================================================

    def train(
        self,
        reference_df
    ):
        """
        Train Isolation Forest on reference data.
        """

        reference_data = prepare_isolation_data(
            reference_df,
            exclude_columns=self.exclude_columns
        )

        if reference_data.empty:
            raise ValueError(
                "Reference dataset contains "
                "no usable numerical features."
            )

        if reference_data.shape[1] == 0:
            raise ValueError(
                "No numerical features available "
                "for Isolation Forest."
            )

        self.feature_columns = list(
            reference_data.columns
        )

        self.scaler = StandardScaler()

        X_reference = self.scaler.fit_transform(
            reference_data
        )

        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state,
            n_jobs=-1
        )

        self.model.fit(
            X_reference
        )

        return self


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        current_df
    ):
        """
        Detect anomalies in current data.
        """

        if self.model is None:
            raise RuntimeError(
                "Isolation Forest has not been trained."
            )

        if self.scaler is None:
            raise RuntimeError(
                "Scaler has not been initialized."
            )

        if not self.feature_columns:
            raise RuntimeError(
                "Feature metadata is missing."
            )

        current_data = prepare_isolation_data(
            current_df,
            exclude_columns=self.exclude_columns
        )

        missing_features = [
            column
            for column in self.feature_columns
            if column not in current_data.columns
        ]

        if missing_features:
            raise ValueError(
                "Current dataset is missing expected "
                f"features: {missing_features}"
            )

        current_data = current_data[
            self.feature_columns
        ]

        X_current = self.scaler.transform(
            current_data
        )

        predictions = self.model.predict(
            X_current
        )

        # Higher value = more anomalous
        anomaly_scores = (
            -self.model.decision_function(
                X_current
            )
        )

        anomaly_flags = (
            predictions == -1
        )

        anomaly_count = int(
            anomaly_flags.sum()
        )

        total_rows = len(
            anomaly_flags
        )

        if total_rows > 0:
            anomaly_percentage = (
                anomaly_count /
                total_rows
            ) * 100
        else:
            anomaly_percentage = 0.0

        results = current_df.copy()

        results["isolation_score"] = (
            anomaly_scores
        )

        results["is_anomaly"] = (
            anomaly_flags
        )

        return {
            "results": results,

            "anomaly_count":
                anomaly_count,

            "anomaly_percentage":
                float(anomaly_percentage),

            "feature_columns":
                self.feature_columns
        }


    # ========================================================
    # SAVE
    # ========================================================

    def save(self):

        os.makedirs(
            "models",
            exist_ok=True
        )

        if self.model is None:
            raise RuntimeError(
                "Cannot save an untrained "
                "Isolation Forest."
            )

        if self.scaler is None:
            raise RuntimeError(
                "Cannot save because scaler "
                "is not initialized."
            )

        joblib.dump(
            self.model,
            self.config.model_path
        )

        joblib.dump(
            self.scaler,
            self.config.scaler_path
        )

        metadata = {
            "feature_columns":
                self.feature_columns,

            "exclude_columns":
                self.exclude_columns,

            "config":
                asdict(self.config)
        }

        joblib.dump(
            metadata,
            self.config.metadata_path
        )

        return {
            "model_path":
                self.config.model_path,

            "scaler_path":
                self.config.scaler_path,

            "metadata_path":
                self.config.metadata_path
        }


    # ========================================================
    # LOAD
    # ========================================================

    def load(self):

        required_files = [
            self.config.model_path,
            self.config.scaler_path,
            self.config.metadata_path
        ]

        missing_files = [
            path
            for path in required_files
            if not os.path.exists(path)
        ]

        if missing_files:
            raise FileNotFoundError(
                "Required Isolation Forest files "
                f"are missing: {missing_files}"
            )

        self.model = joblib.load(
            self.config.model_path
        )

        self.scaler = joblib.load(
            self.config.scaler_path
        )

        metadata = joblib.load(
            self.config.metadata_path
        )

        self.feature_columns = metadata.get(
            "feature_columns",
            []
        )

        self.exclude_columns = metadata.get(
            "exclude_columns",
            []
        )

        return self