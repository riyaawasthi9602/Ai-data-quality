import pandas as pd

from src.quality import DataQualityEngine
from src.quality_rules import QualityConfig


df = pd.DataFrame({
    "Patient ID": [
        1,
        2,
        2,
        4,
        5
    ],

    "Age": [
        25,
        30,
        -10,
        200,
        None
    ],

    "Email": [
        "riya@gmail.com",
        "wrong-email",
        "aman@gmail.com",
        None,
        "test@gmail.com"
    ],

    "City": [
        "India",
        "india",
        "INDIA",
        "Delhi",
        " Delhi "
    ]
})


config = QualityConfig(

    numeric_ranges={
        "Age": {
            "min": 0,
            "max": 120
        }
    },

    email_columns=[
        "Email"
    ],

    id_columns=[
        "Patient ID"
    ]
)


engine = DataQualityEngine(config)

result = engine.analyze(df)


print("\n==============================")
print("BAD DATA QUALITY TEST")
print("==============================")

print(
    "Overall Score:",
    result["overall_quality_score"]
)

print(
    "Quality Level:",
    result["quality_level"]
)


print("\nDimensions:")

for name, score in result[
    "dimensions"
].items():

    print(
        f"{name}: {score}"
    )


print("\nColumn Scores:")

for column, data in result[
    "column_scores"
].items():

    print(
        f"\n{column}"
    )

    print(
        "Quality:",
        data["quality_score"]
    )

    print(
        "Missing:",
        data["missing_values"]
    )

    print(
        "Invalid:",
        data["invalid_count"]
    )

    print(
        "Consistency Issues:",
        data["consistency_issues"]
    )

    print(
        "Duplicate IDs:",
        data["duplicate_id_count"]
    )

    print(
        "Outliers:",
        data["outlier_count"]
    )