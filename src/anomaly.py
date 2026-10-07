import os
import joblib
import numpy as np
import pandas as pd

from dataclasses import dataclass, asdict

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class AutoencoderConfig:
    """
    Configuration for the Autoencoder anomaly detector.
    """

    validation_size: float = 0.20

    epochs: int = 50

    batch_size: int = 256

    learning_rate: float = 0.001

    threshold_percentile: float = 95.0

    random_state: int = 42

    patience: int = 5

    encoder_layers: tuple = (32, 16, 8)

    model_path: str = "models/autoencoder.keras"

    scaler_path: str = "models/scaler.pkl"

    metadata_path: str = "models/autoencoder_metadata.pkl"

    def validate(self):
        if not 0 < self.validation_size < 1:
            raise ValueError(
                "validation_size must be between 0 and 1."
            )

        if self.epochs <= 0:
            raise ValueError(
                "epochs must be greater than 0."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        if not 0 < self.threshold_percentile < 100:
            raise ValueError(
                "threshold_percentile must be between 0 and 100."
            )

        if self.patience < 0:
            raise ValueError(
                "patience cannot be negative."
            )

        if not self.encoder_layers:
            raise ValueError(
                "encoder_layers cannot be empty."
            )


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(
    df,
    exclude_columns=None
):
    """
    Prepare numerical data for Autoencoder training.

    Steps:
    1. Select numeric columns dynamically.
    2. Remove explicitly excluded columns.
    3. Replace infinite values with NaN.
    4. Fill missing values using column medians.
    """

    if exclude_columns is None:
        exclude_columns = []

    # --------------------------------------------------------
    # Select numeric columns dynamically
    # --------------------------------------------------------

    data = df.select_dtypes(
        include=[np.number]
    ).copy()

    # --------------------------------------------------------
    # Remove configured identifier columns
    # --------------------------------------------------------

    columns_to_remove = [
        column
        for column in exclude_columns
        if column in data.columns
    ]

    if columns_to_remove:
        data = data.drop(
            columns=columns_to_remove
        )

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Fill missing values
    # --------------------------------------------------------

    for column in data.columns:

        median_value = data[column].median()

        if pd.isna(median_value):
            median_value = 0.0

        data[column] = data[column].fillna(
            median_value
        )

    return data


# ============================================================
# AUTOENCODER ARCHITECTURE
# ============================================================

def build_autoencoder(
    input_dim,
    encoder_layers=(32, 16, 8)
):
    """
    Build a configurable Autoencoder.
    """

    if input_dim <= 0:
        raise ValueError(
            "input_dim must be greater than 0."
        )

    input_layer = Input(
        shape=(input_dim,)
    )

    encoded = input_layer

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    for units in encoder_layers:

        encoded = Dense(
            units,
            activation="relu"
        )(encoded)

    # --------------------------------------------------------
    # Decoder
    # --------------------------------------------------------

    decoded = encoded

    for units in reversed(
        encoder_layers[:-1]
    ):

        decoded = Dense(
            units,
            activation="relu"
        )(decoded)

    decoded = Dense(
        input_dim,
        activation="linear"
    )(decoded)

    model = Model(
        inputs=input_layer,
        outputs=decoded
    )

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


# ============================================================
# RECONSTRUCTION ERROR
# ============================================================

def calculate_reconstruction_error(
    model,
    data
):
    """
    Calculate reconstruction error for every row.
    """

    reconstructed = model.predict(
        data,
        verbose=0
    )

    error = np.mean(
        np.square(
            data - reconstructed
        ),
        axis=1
    )

    return error


# ============================================================
# THRESHOLD
# ============================================================

def calculate_anomaly_threshold(
    reconstruction_errors,
    percentile=95.0
):
    """
    Learn anomaly threshold from baseline
    reconstruction errors.
    """

    errors = np.asarray(
        reconstruction_errors,
        dtype=float
    )

    errors = errors[
        np.isfinite(errors)
    ]

    if len(errors) == 0:
        raise ValueError(
            "Cannot calculate anomaly threshold "
            "because reconstruction errors are empty."
        )

    return float(
        np.percentile(
            errors,
            percentile
        )
    )


# ============================================================
# MAIN ANALYZER
# ============================================================

class AutoencoderAnomalyDetector:

    def __init__(
        self,
        config=None,
        exclude_columns=None
    ):

        self.config = (
            config
            if config is not None
            else AutoencoderConfig()
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

        self.threshold = None

        self.reference_mean_error = None


    # ========================================================
    # TRAIN
    # ========================================================

    def train(
        self,
        reference_df
    ):
        """
        Train Autoencoder using reference/baseline data.
        """

        reference_data = prepare_data(
            reference_df,
            exclude_columns=self.exclude_columns
        )

        if reference_data.empty:
            raise ValueError(
                "Reference dataset contains no usable "
                "numerical features."
            )

        if reference_data.shape[1] == 0:
            raise ValueError(
                "No numerical features available "
                "for Autoencoder."
            )

        self.feature_columns = list(
            reference_data.columns
        )

        # ----------------------------------------------------
        # Scale reference data
        # ----------------------------------------------------

        self.scaler = StandardScaler()

        X_reference = self.scaler.fit_transform(
            reference_data
        )

        # ----------------------------------------------------
        # Train / validation split
        # ----------------------------------------------------

        X_train, X_validation = train_test_split(
            X_reference,
            test_size=self.config.validation_size,
            random_state=self.config.random_state
        )

        # ----------------------------------------------------
        # Build model
        # ----------------------------------------------------

        self.model = build_autoencoder(
            input_dim=X_train.shape[1],
            encoder_layers=self.config.encoder_layers
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.config.patience,
            restore_best_weights=True
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        history = self.model.fit(
            X_train,
            X_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=(
                X_validation,
                X_validation
            ),
            callbacks=[early_stopping],
            verbose=1
        )

        # ----------------------------------------------------
        # Calculate baseline reconstruction error
        # ----------------------------------------------------

        validation_error = (
            calculate_reconstruction_error(
                self.model,
                X_validation
            )
        )

        self.reference_mean_error = float(
            np.mean(validation_error)
        )

        # ----------------------------------------------------
        # Learn anomaly threshold
        # ----------------------------------------------------

        self.threshold = (
            calculate_anomaly_threshold(
                validation_error,
                percentile=self.config.threshold_percentile
            )
        )

        return history


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        current_df
    ):
        """
        Detect anomalies in a new dataset.
        """

        if self.model is None:
            raise RuntimeError(
                "Autoencoder has not been trained."
            )

        if self.scaler is None:
            raise RuntimeError(
                "Scaler has not been initialized."
            )

        if not self.feature_columns:
            raise RuntimeError(
                "Feature metadata is missing."
            )

        if self.threshold is None:
            raise RuntimeError(
                "Anomaly threshold has not been initialized."
            )

        current_data = prepare_data(
            current_df,
            exclude_columns=self.exclude_columns
        )

        # ----------------------------------------------------
        # Ensure expected features exist
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Use same feature order as training
        # ----------------------------------------------------

        current_data = current_data[
            self.feature_columns
        ]

        # ----------------------------------------------------
        # Scale using reference scaler
        # ----------------------------------------------------

        X_current = self.scaler.transform(
            current_data
        )

        # ----------------------------------------------------
        # Reconstruction error
        # ----------------------------------------------------

        current_error = (
            calculate_reconstruction_error(
                self.model,
                X_current
            )
        )

        # ----------------------------------------------------
        # Anomaly flags
        # ----------------------------------------------------

        anomaly_flags = (
            current_error > self.threshold
        )

        anomaly_count = int(
            anomaly_flags.sum()
        )

        total_rows = len(
            current_error
        )

        if total_rows > 0:

            anomaly_percentage = (
                anomaly_count /
                total_rows
            ) * 100

        else:

            anomaly_percentage = 0.0

        # ----------------------------------------------------
        # Result dataframe
        # ----------------------------------------------------

        results = current_df.copy()

        results["reconstruction_error"] = (
            current_error
        )

        results["is_anomaly"] = (
            anomaly_flags
        )

        # ----------------------------------------------------
        # Mean error
        # ----------------------------------------------------

        current_mean_error = float(
            np.mean(current_error)
        ) if total_rows > 0 else 0.0

        # ----------------------------------------------------
        # Error increase
        # ----------------------------------------------------

        if self.reference_mean_error is not None:

            error_increase = (
                (
                    current_mean_error -
                    self.reference_mean_error
                )
                /
                (
                    self.reference_mean_error +
                    1e-10
                )
            ) * 100

        else:

            error_increase = 0.0

        return {
            "results": results,

            "threshold": float(
                self.threshold
            ),

            "reference_mean_error": float(
                self.reference_mean_error
            ),

            "current_mean_error": (
                current_mean_error
            ),

            "anomaly_count": (
                anomaly_count
            ),

            "anomaly_percentage": (
                float(anomaly_percentage)
            ),

            "error_increase": (
                float(error_increase)
            ),

            "feature_columns": (
                self.feature_columns
            )
        }


    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self
    ):
        """
        Persist model, scaler and metadata.
        """

        os.makedirs(
            "models",
            exist_ok=True
        )

        if self.model is None:
            raise RuntimeError(
                "Cannot save an untrained Autoencoder."
            )

        if self.scaler is None:
            raise RuntimeError(
                "Cannot save because scaler is not initialized."
            )

        if self.threshold is None:
            raise RuntimeError(
                "Cannot save because anomaly threshold "
                "is not initialized."
            )

        self.model.save(
            self.config.model_path
        )

        joblib.dump(
            self.scaler,
            self.config.scaler_path
        )

        metadata = {
            "feature_columns":
                self.feature_columns,

            "threshold":
                self.threshold,

            "reference_mean_error":
                self.reference_mean_error,

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

    def load(
        self
    ):
        """
        Load a previously trained Autoencoder,
        scaler and metadata.
        """

        # ----------------------------------------------------
        # Check required files
        # ----------------------------------------------------

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
                "Required Autoencoder files are missing: "
                f"{missing_files}"
            )

        # ----------------------------------------------------
        # Load trained Keras model
        # ----------------------------------------------------

        from tensorflow.keras.models import load_model

        self.model = load_model(
            self.config.model_path
        )

        # ----------------------------------------------------
        # Load scaler
        # ----------------------------------------------------

        self.scaler = joblib.load(
            self.config.scaler_path
        )

        # ----------------------------------------------------
        # Load metadata
        # ----------------------------------------------------

        metadata = joblib.load(
            self.config.metadata_path
        )

        # ----------------------------------------------------
        # Restore feature information
        # ----------------------------------------------------

        self.feature_columns = metadata.get(
            "feature_columns",
            []
        )

        self.threshold = metadata.get(
            "threshold"
        )

        self.reference_mean_error = metadata.get(
            "reference_mean_error"
        )

        self.exclude_columns = metadata.get(
            "exclude_columns",
            []
        )

        return self


# ============================================================
# BACKWARD-COMPATIBLE FUNCTION
# ============================================================

def train_and_evaluate(
    reference_path,
    current_path
):
    """
    Backward-compatible wrapper.

    Existing code can continue calling:

        train_and_evaluate(
            reference_path,
            current_path
        )
    """

    reference_df = pd.read_csv(
        reference_path
    )

    current_df = pd.read_csv(
        current_path
    )

    print(
        "\nReference rows:",
        len(reference_df)
    )

    print(
        "Current rows:",
        len(current_df)
    )

    # --------------------------------------------------------
    # Default identifier exclusion
    # --------------------------------------------------------

    exclude_columns = []

    if "Patient Number" in reference_df.columns:
        exclude_columns.append(
            "Patient Number"
        )

    detector = AutoencoderAnomalyDetector(
        exclude_columns=exclude_columns
    )

    print(
        "\n========== TRAINING AUTOENCODER =========="
    )

    detector.train(
        reference_df
    )

    result = detector.predict(
        current_df
    )

    print(
        "\n========== AUTOENCODER RESULTS =========="
    )

    print(
        "Input features:",
        len(
            result["feature_columns"]
        )
    )

    print(
        "Reference mean reconstruction error:",
        result["reference_mean_error"]
    )

    print(
        "Current mean reconstruction error:",
        result["current_mean_error"]
    )

    print(
        "Reference threshold:",
        result["threshold"]
    )

    print(
        "Current anomalies:",
        result["anomaly_count"]
    )

    print(
        "Current anomaly percentage:",
        round(
            result["anomaly_percentage"],
            2
        ),
        "%"
    )

    print(
        "Reconstruction error change:",
        round(
            result["error_increase"],
            2
        ),
        "%"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    saved_paths = detector.save()

    print(
        "\n========== MODEL SAVED =========="
    )

    print(
        "Model:",
        saved_paths["model_path"]
    )

    print(
        "Scaler:",
        saved_paths["scaler_path"]
    )

    print(
        "Metadata:",
        saved_paths["metadata_path"]
    )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "model":
            detector.model,

        "scaler":
            detector.scaler,

        "threshold":
            result["threshold"],

        "reference_mean_error":
            result["reference_mean_error"],

        "current_mean_error":
            result["current_mean_error"],

        "anomaly_count":
            result["anomaly_count"],

        "anomaly_percentage":
            result["anomaly_percentage"],

        "error_increase":
            result["error_increase"],

        "results":
            result["results"],

        "feature_columns":
            result["feature_columns"]
    }