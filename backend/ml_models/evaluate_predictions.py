import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# STEP 1. LOAD ONLY REQUIRED COLUMNS
# ------------------------------
pred_file = "predictions/predicted_results_3way_0829-AH.csv"
results_url = "https://www.football-data.co.uk/mmz4281/2526/Latest_Results.csv"

# Columns we want from the prediction file
pred_cols = [
    "Div", "Date", "Team", "Opponent",
    "Predicted Result", "Prob_HomeWin", "Prob_Draw", "Prob_AwayWin"
]

# Read prediction CSV with selected columns only
df_pred = pd.read_csv(pred_file, usecols=pred_cols)

# Read football-data results
df_res = pd.read_csv(results_url)

# ------------------------------
# STEP 2. PREPROCESS DATA
# ------------------------------
df_pred.columns = df_pred.columns.str.strip()
df_res.columns = df_res.columns.str.strip()

# Convert dates to datetime format
df_pred["Date"] = pd.to_datetime(df_pred["Date"], dayfirst=True, errors="coerce")
df_res["Date"] = pd.to_datetime(df_res["Date"], dayfirst=True, errors="coerce")

# Map Predicted Result (W/D/L) → FTR format (H/D/A)
result_map = {"W": "H", "D": "D", "L": "A"}
df_pred["Predicted_FTR"] = df_pred["Predicted Result"].map(result_map)

# ------------------------------
# STEP 3. MATCH PREDICTIONS WITH RESULTS
# ------------------------------
# Match when Team = Home, Opponent = Away
merged1 = pd.merge(
    df_pred,
    df_res,
    left_on=["Div", "Date", "Team", "Opponent"],
    right_on=["Div", "Date", "HomeTeam", "AwayTeam"],
    how="inner"
)

# Match when Team = Away, Opponent = Home
merged2 = pd.merge(
    df_pred,
    df_res,
    left_on=["Div", "Date", "Opponent", "Team"],
    right_on=["Div", "Date", "HomeTeam", "AwayTeam"],
    how="inner"
)

# Combine matches & drop duplicates
merged = pd.concat([merged1, merged2], ignore_index=True).drop_duplicates()

# ------------------------------
# STEP 4. CALCULATE PREDICTION ACCURACY
# ------------------------------
# Add Correct column
merged["Correct"] = (merged["Predicted_FTR"] == merged["FTR"]).astype(int)

# Overall accuracy
overall_accuracy = merged["Correct"].mean()

# Accuracy per outcome (H/D/A)
accuracy_per_outcome = merged.groupby("Predicted_FTR")["Correct"].mean()

# ------------------------------
# STEP 5. ADD PROBABILITY BUCKETS
# ------------------------------
def prob_bucket(prob):
    if prob >= 0.9:
        return "≥0.9"
    elif prob >= 0.8:
        return "≥0.8"
    elif prob >= 0.7:
        return "≥0.7"
    else:
        return "<0.7"

# Assign probability buckets based on predicted outcome
merged["Prob_Bucket"] = merged.apply(
    lambda row: prob_bucket(
        row["Prob_HomeWin"]
        if row["Predicted_FTR"] == "H"
        else row["Prob_Draw"]
        if row["Predicted_FTR"] == "D"
        else row["Prob_AwayWin"]
    ),
    axis=1
)

# ------------------------------
# STEP 6. ACCURACY BY CONFIDENCE BUCKET
# ------------------------------
bucket_accuracy = (
    merged.groupby(["Predicted_FTR", "Prob_Bucket"])["Correct"]
    .mean()
    .unstack(fill_value=0)
)

# ------------------------------
# STEP 7. DISPLAY RESULTS IN TERMINAL
# ------------------------------
print("\n=== OVERALL PREDICTION ACCURACY ===")
print(f"Accuracy: {overall_accuracy:.2%}")

print("\n=== ACCURACY PER PREDICTED OUTCOME (H/D/A) ===")
print(accuracy_per_outcome)

print("\n=== ACCURACY BY CONFIDENCE BUCKETS ===")
print(bucket_accuracy)

# ------------------------------
# STEP 8. VISUALIZATIONS
# ------------------------------
sns.set(style="whitegrid")

# Chart 1: Accuracy per outcome
plt.figure(figsize=(6, 4))
accuracy_per_outcome.plot(kind="bar", color=["#2ecc71", "#f39c12", "#e74c3c"])
plt.title("Accuracy per Predicted Outcome (H/D/A)")
plt.xlabel("Predicted Outcome")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("evaluate/accuracy_per_outcome.png")
plt.show()

# Chart 2: Accuracy by probability bucket
plt.figure(figsize=(8, 6))
bucket_accuracy.T.plot(kind="bar", figsize=(8, 6))
plt.title("Accuracy by Probability Bucket")
plt.xlabel("Probability Bucket")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("evaluate/accuracy_by_probability.png")
plt.show()

# ------------------------------
# STEP 9. SAVE FINAL RESULTS
# ------------------------------
output_cols = [
    "Div", "Date", "Team", "Opponent", "Predicted Result",
    "Prob_HomeWin", "Prob_Draw", "Prob_AwayWin",
    "Predicted_FTR", "FTR", "Correct", "Prob_Bucket"
]
merged[output_cols].to_csv("evaluate/prediction_evaluation_results.csv", index=False)

print("\n✅ Results saved to: prediction_evaluation_results.csv")
print("✅ Charts saved: accuracy_per_outcome.png & accuracy_by_probability.png")
