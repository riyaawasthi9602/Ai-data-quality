from src.quality import analyze_quality

result = analyze_quality(
    "data/Synthetic_patient-HealthCare-Monitoring_dataset.csv"
)

print("\n==============================")
print("DATA QUALITY RESULT")
print("==============================")

print(
    "Overall Score:",
    result["overall_quality_score"]
)

print(
    "Quality Level:",
    result["quality_level"]
)

print(
    "\nDimensions:"
)

for name, score in result["dimensions"].items():
    print(
        f"{name}: {score}"
    )

print(
    "\nColumn Scores:"
)

for column, data in result[
    "column_scores"
].items():

    print(
        column,
        "->",
        data["quality_score"]
    )