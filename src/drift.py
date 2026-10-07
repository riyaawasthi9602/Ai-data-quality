import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency


# ==========================================================
# PSI CALCULATION
# ==========================================================

def calculate_psi(reference, current, bins=10):
    """
    Calculate Population Stability Index (PSI)
    between reference and current numerical data.
    """

    reference = (
        pd.to_numeric(reference, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    current = (
        pd.to_numeric(current, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(reference) < 2 or len(current) < 2:
        return None

    # ------------------------------------------------------
    # Create bins from reference distribution
    # ------------------------------------------------------

    quantiles = np.linspace(
        0,
        1,
        bins + 1
    )

    breakpoints = np.quantile(
        reference,
        quantiles
    )

    breakpoints = np.unique(
        breakpoints
    )

    # If reference has no variation
    if len(breakpoints) < 2:
        return 0.0

    # ------------------------------------------------------
    # Extend first and last bins
    #
    # This allows current values outside the reference
    # range to still be included in the calculation.
    # ------------------------------------------------------

    extended_breakpoints = breakpoints.copy()

    extended_breakpoints[0] = -np.inf
    extended_breakpoints[-1] = np.inf

    # ------------------------------------------------------
    # Create bins
    # ------------------------------------------------------

    reference_bins = pd.cut(
        reference,
        bins=extended_breakpoints,
        include_lowest=True,
        duplicates="drop"
    )

    current_bins = pd.cut(
        current,
        bins=extended_breakpoints,
        include_lowest=True,
        duplicates="drop"
    )

    # ------------------------------------------------------
    # Calculate distributions
    # ------------------------------------------------------

    reference_distribution = (
        reference_bins.value_counts(
            normalize=True,
            sort=False
        )
    )

    current_distribution = (
        current_bins.value_counts(
            normalize=True,
            sort=False
        )
    )

    # Make sure both distributions use
    # exactly the same bins
    current_distribution = (
        current_distribution
        .reindex(
            reference_distribution.index,
            fill_value=0.0
        )
    )

    # ------------------------------------------------------
    # Prevent zero probabilities
    # ------------------------------------------------------

    epsilon = 1e-6

    reference_distribution = (
        reference_distribution
        .clip(lower=epsilon)
    )

    current_distribution = (
        current_distribution
        .clip(lower=epsilon)
    )

    # ------------------------------------------------------
    # Calculate PSI
    # ------------------------------------------------------

    psi = (
        (
            current_distribution
            - reference_distribution
        )
        *
        np.log(
            current_distribution
            /
            reference_distribution
        )
    ).sum()

    return float(psi)


# ==========================================================
# PSI SEVERITY
# ==========================================================

def get_psi_severity(psi):

    if psi is None:
        return "not_available"

    if psi < 0.10:
        return "low"

    if psi < 0.25:
        return "moderate"

    return "high"


# ==========================================================
# CHI-SQUARE SEVERITY
# ==========================================================

def get_chi_square_severity(
    p_value,
    alpha=0.05
):

    if p_value >= alpha:
        return "low"

    if p_value >= alpha / 10:
        return "moderate"

    return "high"


# ==========================================================
# MISSINGNESS SEVERITY
# ==========================================================

def get_missingness_severity(
    difference,
    threshold=0.05
):

    absolute_difference = abs(
        difference
    )

    if absolute_difference < threshold:
        return "low"

    if absolute_difference < threshold * 2:
        return "moderate"

    return "high"


# ==========================================================
# FEATURE SEVERITY
# ==========================================================

def get_feature_severity(
    feature_result
):

    if feature_result["type"] == "numerical":

        return feature_result.get(
            "psi_severity",
            "low"
        )

    return feature_result.get(
        "severity",
        "low"
    )


# ==========================================================
# OVERALL SEVERITY
# ==========================================================

def calculate_overall_severity(
    results
):

    severities = [
        result.get(
            "feature_severity",
            "low"
        )
        for result in results
    ]

    if "high" in severities:
        return "high"

    if "moderate" in severities:
        return "moderate"

    return "low"


# ==========================================================
# MISSINGNESS DRIFT
# ==========================================================

def detect_missingness_drift(
    reference,
    current,
    threshold=0.05
):

    results = []

    common_columns = [
        col
        for col in reference.columns
        if col in current.columns
    ]

    for column in common_columns:

        reference_missing_count = int(
            reference[column].isna().sum()
        )

        current_missing_count = int(
            current[column].isna().sum()
        )

        reference_missing_rate = (
            reference_missing_count
            /
            len(reference)
            if len(reference) > 0
            else 0.0
        )

        current_missing_rate = (
            current_missing_count
            /
            len(current)
            if len(current) > 0
            else 0.0
        )

        difference = (
            current_missing_rate
            -
            reference_missing_rate
        )

        severity = get_missingness_severity(
            difference,
            threshold
        )

        drift = (
            abs(difference)
            >= threshold
        )

        results.append({
            "column":
                column,

            "reference_missing_count":
                reference_missing_count,

            "current_missing_count":
                current_missing_count,

            "reference_missing_rate":
                float(
                    reference_missing_rate
                ),

            "current_missing_rate":
                float(
                    current_missing_rate
                ),

            "missingness_difference":
                float(
                    difference
                ),

            "missingness_threshold":
                float(
                    threshold
                ),

            "severity":
                severity,

            "drift":
                bool(drift)
        })

    return results


# ==========================================================
# MAIN DRIFT DETECTION
# ==========================================================

def detect_drift(
    reference_path,
    current_path,
    alpha=0.05,
    psi_bins=10,
    psi_threshold=0.25,
    missingness_threshold=0.05
):

    reference = pd.read_csv(
        reference_path
    )

    current = pd.read_csv(
        current_path
    )

    results = []

    common_columns = [
        col
        for col in reference.columns
        if col in current.columns
    ]

    # ======================================================
    # ANALYZE COLUMNS
    # ======================================================

    for column in common_columns:

        # ==================================================
        # NUMERICAL COLUMNS
        # ==================================================

        if (
            pd.api.types.is_numeric_dtype(
                reference[column]
            )
            and
            pd.api.types.is_numeric_dtype(
                current[column]
            )
        ):

            ref = (
                pd.to_numeric(
                    reference[column],
                    errors="coerce"
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .dropna()
            )

            cur = (
                pd.to_numeric(
                    current[column],
                    errors="coerce"
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .dropna()
            )

            if (
                len(ref) >= 2
                and
                len(cur) >= 2
            ):

                # ------------------------------------------
                # KS TEST
                # ------------------------------------------

                statistic, p_value = ks_2samp(
                    ref,
                    cur
                )

                # ------------------------------------------
                # PSI
                # ------------------------------------------

                psi = calculate_psi(
                    ref,
                    cur,
                    bins=psi_bins
                )

                psi_severity = (
                    get_psi_severity(
                        psi
                    )
                )

                # ------------------------------------------
                # DRIFT DECISION
                # ------------------------------------------

                drift = (
                    p_value < alpha
                    or
                    (
                        psi is not None
                        and
                        psi >= psi_threshold
                    )
                )

                result = {

                    "column":
                        column,

                    "type":
                        "numerical",

                    "method":
                        "KS + PSI",

                    "ks_score":
                        float(
                            statistic
                        ),

                    "p_value":
                        float(
                            p_value
                        ),

                    "alpha":
                        float(
                            alpha
                        ),

                    "psi":
                        psi,

                    "psi_threshold":
                        float(
                            psi_threshold
                        ),

                    "psi_severity":
                        psi_severity,

                    "drift":
                        bool(
                            drift
                        ),

                    "reference_count":
                        int(
                            len(ref)
                        ),

                    "current_count":
                        int(
                            len(cur)
                        )
                }

                result[
                    "feature_severity"
                ] = get_feature_severity(
                    result
                )

                results.append(
                    result
                )

        # ==================================================
        # CATEGORICAL COLUMNS
        # ==================================================

        else:

            ref_series = (
                reference[column]
                .dropna()
                .astype(str)
                .str.strip()
            )

            cur_series = (
                current[column]
                .dropna()
                .astype(str)
                .str.strip()
            )

            ref_counts = (
                ref_series.value_counts()
            )

            cur_counts = (
                cur_series.value_counts()
            )

            categories = sorted(
                set(ref_counts.index)
                |
                set(cur_counts.index)
            )

            if len(categories) < 2:
                continue

            ref_values = [
                int(
                    ref_counts.get(
                        category,
                        0
                    )
                )
                for category in categories
            ]

            cur_values = [
                int(
                    cur_counts.get(
                        category,
                        0
                    )
                )
                for category in categories
            ]

            try:

                (
                    chi2_statistic,
                    p_value,
                    _,
                    _
                ) = chi2_contingency(
                    [
                        ref_values,
                        cur_values
                    ]
                )

                severity = (
                    get_chi_square_severity(
                        p_value,
                        alpha
                    )
                )

                reference_distribution = {
                    category:
                    round(
                        ref_counts.get(
                            category,
                            0
                        )
                        /
                        len(ref_series),
                        6
                    )
                    for category in categories
                }

                current_distribution = {
                    category:
                    round(
                        cur_counts.get(
                            category,
                            0
                        )
                        /
                        len(cur_series),
                        6
                    )
                    for category in categories
                }

                result = {

                    "column":
                        column,

                    "type":
                        "categorical",

                    "method":
                        "chi-square",

                    "chi2_statistic":
                        float(
                            chi2_statistic
                        ),

                    "p_value":
                        float(
                            p_value
                        ),

                    "alpha":
                        float(
                            alpha
                        ),

                    "drift":
                        bool(
                            p_value < alpha
                        ),

                    "severity":
                        severity,

                    "category_count":
                        int(
                            len(categories)
                        ),

                    "reference_distribution":
                        reference_distribution,

                    "current_distribution":
                        current_distribution,

                    "reference_count":
                        int(
                            len(ref_series)
                        ),

                    "current_count":
                        int(
                            len(cur_series)
                        )
                }

                result[
                    "feature_severity"
                ] = get_feature_severity(
                    result
                )

                results.append(
                    result
                )

            except (
                ValueError,
                ZeroDivisionError
            ):
                continue

    # ======================================================
    # MISSINGNESS DRIFT
    # ======================================================

    missingness_results = (
        detect_missingness_drift(
            reference,
            current,
            threshold=
            missingness_threshold
        )
    )

    # ======================================================
    # DISTRIBUTION DRIFT PERCENTAGE
    # ======================================================

    if results:

        drifted_columns = sum(
            result["drift"]
            for result in results
        )

        drift_percentage = (
            drifted_columns
            /
            len(results)
        ) * 100

    else:

        drifted_columns = 0

        drift_percentage = 0.0

    # ======================================================
    # OVERALL SEVERITY
    # ======================================================

    overall_severity = (
        calculate_overall_severity(
            results
        )
    )

    # ======================================================
    # MISSINGNESS DRIFT PERCENTAGE
    # ======================================================

    missingness_drifted = sum(
        item["drift"]
        for item in missingness_results
    )

    if missingness_results:

        missingness_drift_percentage = (
            missingness_drifted
            /
            len(missingness_results)
        ) * 100

    else:

        missingness_drift_percentage = 0.0

    # ======================================================
    # FINAL RESULT
    # ======================================================

    return {

        "results":
            results,

        "missingness":
            missingness_results,

        "drift_percentage":
            float(
                drift_percentage
            ),

        "missingness_drift_percentage":
            float(
                missingness_drift_percentage
            ),

        "overall_severity":
            overall_severity,

        "affected_features":
            int(
                drifted_columns
            )
    }