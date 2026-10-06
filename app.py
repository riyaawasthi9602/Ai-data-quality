from src.profiling import profile_dataset
from src.drift import detect_drift
from src.anomaly import train_and_evaluate
from src.quality import calculate_quality

REFERENCE_PATH = "data/reference.csv"
CURRENT_PATH = "data/current.csv"


# ==========================================
# 1. DATASET PROFILING
# ==========================================

print("\n========== DATA PROFILING ==========")

reference = profile_dataset(REFERENCE_PATH)
current = profile_dataset(CURRENT_PATH)

print("\nREFERENCE DATASET")

for key, value in reference.items():
    print(key, ":", value)

print("\nCURRENT DATASET")

for key, value in current.items():
    print(key, ":", value)


# ==========================================
# 2. DRIFT DETECTION
# ==========================================

print("\n========== DRIFT DETECTION ==========")

results, drift_percentage = detect_drift(
    REFERENCE_PATH,
    CURRENT_PATH
)

for result in results:

    print(
        f"{result['column']} | "
        f"{result['type']} | "
        f"p-value: {result['p_value']} | "
        f"Drift: {result['drift']}"
    )
    # Data quality
    quality_score = calculate_quality(
       REFERENCE_PATH,
       CURRENT_PATH
)

    

# ==========================================
# 3. DEEP LEARNING - AUTOENCODER
# ==========================================

print("\n========== DEEP LEARNING MODEL ==========")

results = train_and_evaluate(
    REFERENCE_PATH,
    CURRENT_PATH
)       
print(
    "\nOverall statistical drift:",
    round(drift_percentage, 2),
    "%"
)