import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping


def prepare_data(df):
    """
    Select numerical features and remove identifier columns.
    """

    data = df.select_dtypes(
        include=["int64", "float64"]
    ).copy()

    # Patient Number is only an ID
    if "Patient Number" in data.columns:
        data = data.drop(
            columns=["Patient Number"]
        )

    # Fill missing numerical values
    data = data.fillna(
        data.median()
    )

    return data


def build_autoencoder(input_dim):

    # Input layer
    input_layer = Input(
        shape=(input_dim,)
    )

    # Encoder
    encoded = Dense(
        32,
        activation="relu"
    )(input_layer)

    encoded = Dense(
        16,
        activation="relu"
    )(encoded)

    encoded = Dense(
        8,
        activation="relu"
    )(encoded)

    # Decoder
    decoded = Dense(
        16,
        activation="relu"
    )(encoded)

    decoded = Dense(
        32,
        activation="relu"
    )(decoded)

    decoded = Dense(
        input_dim,
        activation="linear"
    )(decoded)

    # Autoencoder model
    model = Model(
        input_layer,
        decoded
    )

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


def train_and_evaluate(
    reference_path,
    current_path
):

    # =====================================
    # 1. LOAD DATA
    # =====================================

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

    # =====================================
    # 2. PREPARE DATA
    # =====================================

    reference_data = prepare_data(
        reference_df
    )

    current_data = prepare_data(
        current_df
    )

    # Use only common columns
    common_columns = [
        column
        for column in reference_data.columns
        if column in current_data.columns
    ]

    reference_data = reference_data[
        common_columns
    ]

    current_data = current_data[
        common_columns
    ]

    # =====================================
    # 3. SCALE DATA
    # =====================================

    scaler = StandardScaler()

    # IMPORTANT:
    # Fit scaler ONLY on reference data
    X_reference = scaler.fit_transform(
        reference_data
    )

    # Transform current data using
    # the same scaler
    X_current = scaler.transform(
        current_data
    )

    # =====================================
    # 4. SPLIT REFERENCE DATA
    # =====================================

    X_train, X_validation = train_test_split(
        X_reference,
        test_size=0.20,
        random_state=42
    )

    # =====================================
    # 5. BUILD AUTOENCODER
    # =====================================

    model = build_autoencoder(
        X_train.shape[1]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    print(
        "\n========== TRAINING AUTOENCODER =========="
    )

    # =====================================
    # 6. TRAIN MODEL
    # =====================================

    model.fit(
        X_train,
        X_train,
        epochs=50,
        batch_size=256,
        validation_data=(
            X_validation,
            X_validation
        ),
        callbacks=[early_stopping],
        verbose=1
    )

    # =====================================
    # 7. REFERENCE RECONSTRUCTION ERROR
    # =====================================

    reference_reconstructed = model.predict(
        X_validation,
        verbose=0
    )

    reference_error = np.mean(
        np.square(
            X_validation -
            reference_reconstructed
        ),
        axis=1
    )

    # Threshold learned from
    # reference/normal data
    threshold = np.percentile(
        reference_error,
        95
    )

    # =====================================
    # 8. CURRENT DATA RECONSTRUCTION
    # =====================================

    current_reconstructed = model.predict(
        X_current,
        verbose=0
    )

    current_error = np.mean(
        np.square(
            X_current -
            current_reconstructed
        ),
        axis=1
    )

    # =====================================
    # 9. DETECT CURRENT ANOMALIES
    # =====================================

    current_anomalies = (
        current_error > threshold
    )

    anomaly_count = int(
        current_anomalies.sum()
    )

    anomaly_percentage = (
        anomaly_count /
        len(current_error)
    ) * 100

    # =====================================
    # 10. CALCULATE ERROR STATISTICS
    # =====================================

    reference_mean_error = float(
        np.mean(reference_error)
    )

    current_mean_error = float(
        np.mean(current_error)
    )

    error_increase = (
        (
            current_mean_error -
            reference_mean_error
        )
        /
        (reference_mean_error + 1e-10)
    ) * 100

    # =====================================
    # 11. PRINT RESULTS
    # =====================================

    print(
        "\n========== AUTOENCODER RESULTS =========="
    )

    print(
        "Input features:",
        X_reference.shape[1]
    )

    print(
        "Reference mean reconstruction error:",
        reference_mean_error
    )

    print(
        "Current mean reconstruction error:",
        current_mean_error
    )

    print(
        "Reference threshold:",
        threshold
    )

    print(
        "Current anomalies:",
        anomaly_count
    )

    print(
        "Current anomaly percentage:",
        round(
            anomaly_percentage,
            2
        ),
        "%"
    )

    print(
        "Reconstruction error change:",
        round(
            error_increase,
            2
        ),
        "%"
    )

    # =====================================
    # 12. SAVE MODEL
    # =====================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    model.save(
        "models/autoencoder.keras"
    )

    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )

    print(
        "\n========== MODEL SAVED =========="
    )

    print(
        "Model: models/autoencoder.keras"
    )

    print(
        "Scaler: models/scaler.pkl"
    )

    # =====================================
    # 13. RETURN RESULTS
    # =====================================

    return {
        "model": model,
        "scaler": scaler,
        "threshold": threshold,
        "reference_mean_error":
            reference_mean_error,
        "current_mean_error":
            current_mean_error,
        "anomaly_count":
            anomaly_count,
        "anomaly_percentage":
            anomaly_percentage,
        "error_increase":
            error_increase
    }