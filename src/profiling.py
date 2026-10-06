import pandas as pd


def profile_dataset(file_path):
    df = pd.read_csv(file_path)

    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "numerical_columns": [],
        "categorical_columns": [],
        "date_columns": []
    }

    for col in df.columns:

        # Numerical columns
        if pd.api.types.is_numeric_dtype(df[col]):
            profile["numerical_columns"].append(col)

        # Date columns
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            profile["date_columns"].append(col)

        # Categorical columns
        else:
            profile["categorical_columns"].append(col)

    return profile