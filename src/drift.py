import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency


def detect_drift(reference_path, current_path):

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    results = []

    common_columns = [
        col for col in reference.columns
        if col in current.columns
    ]

    for column in common_columns:

        # Numerical columns
        if (
            pd.api.types.is_numeric_dtype(reference[column])
            and
            pd.api.types.is_numeric_dtype(current[column])
        ):

            ref = reference[column].dropna()
            cur = current[column].dropna()

            if len(ref) > 0 and len(cur) > 0:

                statistic, p_value = ks_2samp(
                    ref,
                    cur
                )

                results.append({
                    "column": column,
                    "type": "numerical",
                    "score": float(statistic),
                    "p_value": float(p_value),
                    "drift": p_value < 0.05
                })

        # Categorical columns
        else:

            ref_counts = reference[column].value_counts()
            cur_counts = current[column].value_counts()

            categories = set(
                ref_counts.index
            ) | set(
                cur_counts.index
            )

            ref_values = [
                ref_counts.get(category, 0)
                for category in categories
            ]

            cur_values = [
                cur_counts.get(category, 0)
                for category in categories
            ]

            try:

                _, p_value, _, _ = chi2_contingency(
                    [ref_values, cur_values]
                )

                results.append({
                    "column": column,
                    "type": "categorical",
                    "score": None,
                    "p_value": float(p_value),
                    "drift": p_value < 0.05
                })

            except ValueError:
                pass

    # Overall drift percentage
    if results:

        drifted_columns = sum(
            result["drift"]
            for result in results
        )

        drift_percentage = (
            drifted_columns /
            len(results)
        ) * 100

    else:

        drift_percentage = 0

    return results, drift_percentage