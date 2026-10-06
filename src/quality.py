import pandas as pd


def calculate_quality(reference_path, current_path):

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    # -----------------------------
    # 1. Missing values
    # -----------------------------

    total_cells = current.shape[0] * current.shape[1]

    missing_cells = current.isnull().sum().sum()

    missing_percentage = (
        missing_cells / total_cells
    ) * 100

    # -----------------------------
    # 2. Duplicate rows
    # -----------------------------

    duplicate_rows = current.duplicated().sum()

    duplicate_percentage = (
        duplicate_rows / len(current)
    ) * 100

    # -----------------------------
    # 3. Schema check
    # -----------------------------

    reference_columns = set(reference.columns)
    current_columns = set(current.columns)

    missing_columns = (
        reference_columns - current_columns
    )

    extra_columns = (
        current_columns - reference_columns
    )

    if len(missing_columns) == 0:
        schema_score = 100
    else:
        schema_score = max(
            0,
            100 -
            (len(missing_columns) /
             len(reference_columns) * 100)
        )

    # -----------------------------
    # 4. Quality score
    # -----------------------------

    missing_score = max(
        0,
        100 - missing_percentage
    )

    duplicate_score = max(
        0,
        100 - duplicate_percentage
    )

    quality_score = (
        missing_score * 0.4 +
        duplicate_score * 0.3 +
        schema_score * 0.3
    )

    print("\n========== DATA QUALITY ==========")

    print(
        "Missing values:",
        missing_cells
    )

    print(
        "Missing percentage:",
        round(missing_percentage, 2),
        "%"
    )

    print(
        "Duplicate rows:",
        duplicate_rows
    )

    print(
        "Duplicate percentage:",
        round(duplicate_percentage, 2),
        "%"
    )

    print(
        "Missing columns:",
        list(missing_columns)
    )

    print(
        "Extra columns:",
        list(extra_columns)
    )

    print(
        "Data Quality Score:",
        round(quality_score, 2),
        "/ 100"
    )

    return quality_score