import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# === Config ===
#MATCHES_CSV_URL = "fixtures.csv"
MATCHES_CSV_URL = "https://www.football-data.co.uk/fixtures.csv"
ROOT = Path(__file__).resolve().parent
BACKEND = ROOT.parent
RISK_SCORE_CSV_PATH = BACKEND / "test" / "updated_team_form_ref.csv"
ARTIFACT_DIR = ROOT / "artifacts_optuna_v2"
# Write to the common predictions folder so send_tele.py can find files
OUTPUT_DIR = ROOT / "predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "pred_ah.csv"

# === Load trained models & encoders ===
calibrated_model = joblib.load(ARTIFACT_DIR / "xgb_calibrated.pkl")
final_model = joblib.load(ARTIFACT_DIR / "xgb_final_model.pkl")
le_result = joblib.load(ARTIFACT_DIR / "result_encoder.pkl")
le_team = joblib.load(ARTIFACT_DIR / "team_encoder.pkl")

# === Load new matches ===
df = pd.read_csv(MATCHES_CSV_URL)

# === Rename standard columns ===
df = df.rename(columns={
    "HomeTeam": "Team",
    "AwayTeam": "Opponent",
    "B365H": "Home Odds",
    "B365D": "Draw Odds",
    "B365A": "Away Odds"
})

# Fill missing odds
for col in ["Home Odds", "Draw Odds", "Away Odds"]:
    if col not in df.columns:
        df[col] = 0
    df[col] = df[col].fillna(0)

# === Bookmaker Margin + Fair Odds ===
def compute_margin_fair_odds(row):
    h, d, a = row["Home Odds"], row["Draw Odds"], row["Away Odds"]
    if h <= 0 or d <= 0 or a <= 0:
        return pd.Series([0.0, h, d, a])
    margin = (1/h + 1/d + 1/a) - 1
    fair_h = 1 / ((1/h) / (1 + margin))
    fair_d = 1 / ((1/d) / (1 + margin))
    fair_a = 1 / ((1/a) / (1 + margin))
    return pd.Series([margin, fair_h, fair_d, fair_a])

df[["Bookmaker Margin", "Fair Odds(H)", "Fair Odds(D)", "Fair Odds(A)"]] = df.apply(compute_margin_fair_odds, axis=1)

# === Load Risk Score Data ===
risk_df = pd.read_csv(RISK_SCORE_CSV_PATH).fillna(0.0)

def lookup_risk(team_name):
    row = risk_df[risk_df["Team"] == team_name]
    if row.empty:
        return pd.Series([0.0, 0.0])
    return pd.Series([row["Risk Score"].values[0], row["Cumulative Score"].values[0]])

df[["Risk Score(H)", "Cumulative Score(H)"]] = df["Team"].apply(lookup_risk)
df[["Risk Score(A)", "Cumulative Score(A)"]] = df["Opponent"].apply(lookup_risk)

# === Filter valid teams ===
valid_mask = df["Team"].isin(le_team.classes_) & df["Opponent"].isin(le_team.classes_)
df_valid = df.loc[valid_mask].copy()

# === Encode teams ===
df_valid["Team_encoded"] = le_team.transform(df_valid["Team"])
df_valid["Opponent_encoded"] = le_team.transform(df_valid["Opponent"])

# === Prepare features ===
feature_cols = [
    "Home Odds", "Draw Odds", "Away Odds",
    "Bookmaker Margin", "Fair Odds(H)", "Fair Odds(D)", "Fair Odds(A)",
    "Risk Score(H)", "Cumulative Score(H)", "Risk Score(A)", "Cumulative Score(A)",
    "Team_encoded", "Opponent_encoded", "AHh","B365AHH","B365AHA","B365>2.5","B365<2.5"
]
feature_cols = [c for c in feature_cols if c in df_valid.columns]
X = df_valid[feature_cols].copy()
X.columns = [c.replace('<', 'lt').replace('>', 'gt') for c in X.columns]
X = X.fillna(0.0)

# === Predict ===
y_probs = calibrated_model.predict_proba(X)
y_pred = calibrated_model.predict(X)
df_valid.loc[:, "Predicted Result"] = le_result.inverse_transform(y_pred)
df_valid.loc[:, "Prob_HomeWin"] = y_probs[:, le_result.transform(["W"])[0]]
df_valid.loc[:, "Prob_Draw"] = y_probs[:, le_result.transform(["D"])[0]]
df_valid.loc[:, "Prob_AwayWin"] = y_probs[:, le_result.transform(["L"])[0]]

# === Implied probabilities from odds ===
def implied_probs(row):
    h, d, a = row["Home Odds"], row["Draw Odds"], row["Away Odds"]
    inv = np.array([1/h if h>0 else 0, 1/d if d>0 else 0, 1/a if a>0 else 0])
    return inv / inv.sum() if inv.sum() > 0 else inv

implied = df_valid.apply(implied_probs, axis=1)
df_valid[["Imp_H","Imp_D","Imp_A"]] = pd.DataFrame(implied.tolist(), index=df_valid.index)

# === Suspicious / bait-match detection ===
FAV_THRESHOLD = 0.80
MODEL_LOW_THRESHOLD = 0.50
GAP_THRESHOLD = 0.35

sus_mask = (
    ((df_valid["Imp_H"] >= FAV_THRESHOLD) & (df_valid["Prob_HomeWin"] < MODEL_LOW_THRESHOLD)) |
    ((df_valid["Imp_A"] >= FAV_THRESHOLD) & (df_valid["Prob_AwayWin"] < MODEL_LOW_THRESHOLD)) |
    ((abs(df_valid["Imp_H"] - df_valid["Prob_HomeWin"]) >= GAP_THRESHOLD) |
     (abs(df_valid["Imp_D"] - df_valid["Prob_Draw"]) >= GAP_THRESHOLD) |
     (abs(df_valid["Imp_A"] - df_valid["Prob_AwayWin"]) >= GAP_THRESHOLD))
)

df_valid["Suspicious"] = sus_mask

# === Save outputs ===
output_cols = ["Date","Team","Opponent","Predicted Result","Prob_HomeWin","Prob_Draw","Prob_AwayWin","Imp_H","Imp_D","Imp_A","Suspicious"]
df_valid[output_cols].to_csv(OUTPUT_PATH, index=False)
print(f"✅ Predictions saved to {OUTPUT_PATH}")

# High-confidence predictions
high_conf_mask = (df_valid["Prob_HomeWin"] >= 0.8) | \
                 (df_valid["Prob_Draw"] >= 0.8) | \
                 (df_valid["Prob_AwayWin"] >= 0.8)
df_high_conf = df_valid.loc[high_conf_mask, output_cols]
high_conf_path = OUTPUT_DIR / "pred_ah_high_conf.csv"
df_high_conf.to_csv(high_conf_path, index=False)
print(f"✅ High-confidence predictions saved to {high_conf_path}")

print("\nSuspicious matches flagged:", df_valid["Suspicious"].sum())

# === Draw-heavy selection (draw probability >= threshold) ===
DRAW_THRESHOLD = 0.33
df_draw_2x = df_valid.loc[df_valid["Prob_Draw"] >= DRAW_THRESHOLD, output_cols]
draw_path = OUTPUT_DIR / "pred_ah_draw_2x.csv"
df_draw_2x.to_csv(draw_path, index=False)
print(f"✅ Draw-leaning predictions (Prob_Draw >= {DRAW_THRESHOLD}) saved to {draw_path} ({len(df_draw_2x)} rows)")
