import pandas as pd
import numpy as np
import joblib
import requests
from io import StringIO
from xgboost import XGBClassifier
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT.parent

# === Load Artifacts ===
model = joblib.load(ROOT / "artifacts_goals" / "over_under_model.pkl")
le_team = joblib.load(ROOT / "artifacts_goals" / "team_encoder.pkl")

# === Config ===
#MATCHES_CSV_URL = "fixtures.csv"
MATCHES_CSV_URL = "https://www.football-data.co.uk/fixtures.csv"
RISK_SCORE_CSV_PATH = BACKEND / "test" / "updated_team_form_ref.csv"
OUTPUT_DIR = ROOT / "predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "pred_goals.csv"

# === Load New Matches CSV ===
try:
    new_matches = pd.read_csv(MATCHES_CSV_URL)
except Exception as e:
    print(f"⚠️ Error loading CSV: {e}")
    raise

# === Rename and Standardize Columns ===
df = new_matches.rename(columns={
    "HomeTeam": "Team",
    "AwayTeam": "Opponent",
    "B365H": "Home Odds",
    "B365D": "Draw Odds",
    "B365A": "Away Odds"
})

# === Bookmaker Margin + Fair Odds ===
def compute_margin_fair_odds(row):
    h, d, a = row["Home Odds"], row["Draw Odds"], row["Away Odds"]
    margin = (1/h + 1/d + 1/a) - 1
    fair_h = 1 / ((1/h) / (1 + margin))
    fair_d = 1 / ((1/d) / (1 + margin))
    fair_a = 1 / ((1/a) / (1 + margin))
    return pd.Series([margin, fair_h, fair_d, fair_a])

df[["Bookmaker Margin", "Fair Odds(H)", "Fair Odds(D)", "Fair Odds(A)"]] = df.apply(compute_margin_fair_odds, axis=1)

# === Load Risk Score Data ===
risk_df = pd.read_csv(RISK_SCORE_CSV_PATH)

def lookup_risk(team_name):
    row = risk_df[risk_df["Team"] == team_name]
    if row.empty:
        return pd.Series([np.nan, np.nan])
    return pd.Series([row["Risk Score"].values[0], row["Cumulative Score"].values[0]])

df[["Risk Score(H)", "Cumulative Score(H)"]] = df["Team"].apply(lookup_risk)
df[["Risk Score(A)", "Cumulative Score(A)"]] = df["Opponent"].apply(lookup_risk)

# === Remove rows with missing team encodings ===
valid_mask = df["Team"].isin(le_team.classes_) & df["Opponent"].isin(le_team.classes_)
df_valid = df.loc[valid_mask].copy()

# === Team Encoding ===
df_valid["Team_encoded"] = le_team.transform(df_valid["Team"])
df_valid["Opponent_encoded"] = le_team.transform(df_valid["Opponent"])

# === Features for Prediction ===
X = df_valid[[
    "Home Odds", "Draw Odds", "Away Odds",
    "Bookmaker Margin", "Fair Odds(H)", "Fair Odds(D)", "Fair Odds(A)",
    "Risk Score(H)", "Cumulative Score(H)", "Risk Score(A)", "Cumulative Score(A)",
    "Team_encoded", "Opponent_encoded", "AHh", "B365AHH", "B365AHA", "B365>2.5", "B365<2.5"
]]
X.columns = [col.replace('<', 'lt').replace('>', 'gt') for col in X.columns]

# === Predict ===
probs = model.predict_proba(X)
preds = model.predict(X)

# === Assign Predictions ===
df_valid.loc[:, "Predicted_Goals"] = ["Over 2.5" if pred == 1 else "Under 2.5" for pred in preds]
df_valid.loc[:, "Prob_Over"] = probs[:, 1]
df_valid.loc[:, "Prob_Under"] = probs[:, 0]

# === Output full predictions ===
output_cols = [
    "Div", "Date", "Team", "Opponent",
    "Predicted_Goals", "Prob_Over", "Prob_Under",
    "B365>2.5", "B365<2.5"  # Include original odds for comparison
]
df_valid[output_cols].to_csv(OUTPUT_PATH, index=False)
print(f"✅ Predictions saved to {OUTPUT_PATH}")

# === Output high-confidence predictions (prob >= 0.75) ===
high_conf_mask = (df_valid["Prob_Over"] >= 0.75) | (df_valid["Prob_Under"] >= 0.75)
df_high_conf = df_valid.loc[high_conf_mask, output_cols].copy()
high_conf_output_path = OUTPUT_DIR / "pred_goals_high_conf.csv"
df_high_conf.to_csv(high_conf_output_path, index=False)
print(f"✅ High-confidence predictions saved to {high_conf_output_path}")

# === Value Betting Analysis ===
df_valid["Model_Over_Fair_Prob"] = df_valid["Prob_Over"]
df_valid["Market_Over_Prob"] = 1 / df_valid["B365>2.5"]
df_valid["Value_Over"] = (df_valid["Model_Over_Fair_Prob"] * df_valid["B365>2.5"]) - 1

value_mask = df_valid["Value_Over"] > 0.1  # 10% edge
value_bets = df_valid[value_mask].sort_values("Value_Over", ascending=False)

value_cols = output_cols + ["Value_Over", "Model_Over_Fair_Prob", "Market_Over_Prob"]
value_output_path = OUTPUT_DIR / "pred_goals_val_bets.csv"
value_bets[value_cols].to_csv(value_output_path, index=False)
print(f"✅ Value betting opportunities saved to {value_output_path}")

# === Print Summary ===
print("\nPrediction Summary:")
print(df_valid[output_cols].head())
print("\nHigh Confidence Predictions:")
print(df_high_conf)
print("\nTop Value Betting Opportunities:")
print(value_bets[value_cols].head())
