# train_optuna_calibration_bait_detector.py
import os
import joblib
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, precision_recall_curve, auc, confusion_matrix
from xgboost import XGBClassifier

# ---------------------------
# Config
# ---------------------------
ROOT = Path(__file__).resolve().parent
BACKEND = ROOT.parent
DATA_PATH = BACKEND / "train" / "cleaned_filled_dataset.csv"
ARTIFACT_DIR = ROOT / "artifacts_optuna_v2"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
MLFLOW_EXPERIMENT = "Football_XGB_Optuna_Calibrated"
RANDOM_STATE = 42
N_TRIALS = 50   # Adjust depending on compute
CV_FOLDS = 4

# ---------------------------
# Load & preprocess
# ---------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
if "League" in df.columns:
    df = df.drop(columns=["League"])

# Encode target
le_result = LabelEncoder()
df["Result_enc"] = le_result.fit_transform(df["Result"])
y = df["Result_enc"]

# Encode teams
teams = pd.concat([df["Team"], df["Opponent"]]).unique()
le_team = LabelEncoder()
le_team.fit(teams)
df["Team_encoded"] = le_team.transform(df["Team"])
df["Opponent_encoded"] = le_team.transform(df["Opponent"])

# Features
features = [
    "Home Odds","Draw Odds","Away Odds",
    "Bookmaker Margin",
    "Fair Odds(H)","Fair Odds(D)","Fair Odds(A)",
    "Risk Score(H)","Cumulative Score(H)",
    "Risk Score(A)","Cumulative Score(A)",
    "Team_encoded","Opponent_encoded",
    "AHh","B365AHH","B365AHA","B365>2.5","B365<2.5"
]
features = [f for f in features if f in df.columns]
df = df.rename(columns={c: c.replace('<','lt').replace('>','gt') for c in df.columns})
features = [c.replace('<','lt').replace('>','gt') for c in features]

X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Train / test split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# ---------------------------
# Optuna objective
# ---------------------------
def objective(trial):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE,
        "verbosity": 0,
        "n_jobs": 1,
    }
    model = XGBClassifier(**param)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    losses = []
    for train_idx, val_idx in skf.split(X_train_full, y_train_full):
        X_tr, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        probs = model.predict_proba(X_val)
        losses.append(log_loss(y_val, probs))
    return np.mean(losses)

# ---------------------------
# Run Optuna
# ---------------------------
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
print("Best params:", study.best_params)
best_params = study.best_params

# ---------------------------
# Train final model with early stopping
# ---------------------------
final_params = dict(best_params)
final_params.update({
    "use_label_encoder": False,
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
    "verbosity": 1,
    "n_jobs": -1
})
final_model = XGBClassifier(**final_params)

# Split small validation for early stopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, stratify=y_train_full, random_state=RANDOM_STATE
)
final_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=50
)

# ---------------------------
# Calibrate probabilities
# ---------------------------
calib_method = "isotonic" if len(X_tr) > 2000 else "sigmoid"
calibrated = CalibratedClassifierCV(final_model, method=calib_method, cv=5)
calibrated.fit(X_train_full, y_train_full)

# ---------------------------
# Evaluate calibrated model
# ---------------------------
y_probs_cal = calibrated.predict_proba(X_test)
y_pred_cal = calibrated.predict(X_test)
acc_cal = accuracy_score(y_test, y_pred_cal)
ll_cal = log_loss(y_test, y_probs_cal)
print(f"Calibrated test accuracy: {acc_cal:.4f}, log_loss: {ll_cal:.4f}")

# ---------------------------
# Save artifacts
# ---------------------------
os.makedirs(ARTIFACT_DIR, exist_ok=True)
joblib.dump(calibrated, ARTIFACT_DIR / "xgb_calibrated.pkl")
joblib.dump(final_model, ARTIFACT_DIR / "xgb_final_model.pkl")
joblib.dump(le_result, ARTIFACT_DIR / "result_encoder.pkl")
joblib.dump(le_team, ARTIFACT_DIR / "team_encoder.pkl")
pd.Series(final_params).to_csv(ARTIFACT_DIR / "best_params.csv")

# ---------------------------
# MLflow logging
# ---------------------------
mlflow.set_experiment(MLFLOW_EXPERIMENT)
with mlflow.start_run():
    mlflow.log_params({k: final_params[k] for k in final_params if k in ["n_estimators","learning_rate","max_depth","subsample","colsample_bytree","reg_lambda","reg_alpha","min_child_weight"]})
    mlflow.log_metric("test_accuracy", acc_cal)
    mlflow.log_metric("test_log_loss", ll_cal)

    # Precision-recall curves
    plt.figure(figsize=(8,6))
    for i, label in enumerate(le_result.classes_):
        precision, recall, _ = precision_recall_curve(y_test == i, y_probs_cal[:, i])
        pr_auc = auc(recall, precision)
        mlflow.log_metric(f"{label}_pr_auc", pr_auc)
        plt.plot(recall, precision, label=f"{label} (AUC={pr_auc:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
    plt.legend()
    pr_path = ARTIFACT_DIR / "precision_recall.png"
    plt.savefig(pr_path); plt.close()
    mlflow.log_artifact(str(pr_path))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_cal)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=le_result.classes_, yticklabels=le_result.classes_)
    plt.title("Confusion Matrix")
    cm_path = ARTIFACT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path); plt.close()
    mlflow.log_artifact(str(cm_path))

# ---------------------------
# High-confidence predictions
# ---------------------------
high_conf_df = X_test.copy()
high_conf_df["Actual"] = le_result.inverse_transform(y_test)
high_conf_df["Predicted"] = le_result.inverse_transform(y_pred_cal)
high_conf_df["Prob_Max"] = y_probs_cal.max(axis=1)
high_conf_df = high_conf_df[high_conf_df["Prob_Max"] >= 0.8]
high_conf_path = ARTIFACT_DIR / "high_confidence_predictions.csv"
high_conf_df.to_csv(high_conf_path, index=False)

# ---------------------------
# Bait/fixing detector
# ---------------------------
implied_probs = []
model_probs = []
rows = []

for idx in X_test.index:
    row = X_test.loc[idx]
    h = row.get("Home Odds", np.nan)
    d = row.get("Draw Odds", np.nan)
    a = row.get("Away Odds", np.nan)
    inv = np.array([1.0/h if h>0 else 0.0, 1.0/d if d>0 else 0.0, 1.0/a if a>0 else 0.0])
    imp = inv / inv.sum() if inv.sum() > 0 else np.array([np.nan, np.nan, np.nan])
    implied_probs.append(imp)
    model_probs.append(y_probs_cal[list(X_test.index).index(idx)])
    rows.append(df.loc[idx, ["Date","Team","Opponent"]])

implied_probs = np.vstack(implied_probs)
model_probs = np.vstack(model_probs)
rows_df = pd.DataFrame(rows).reset_index(drop=True)
comp = pd.concat([rows_df, pd.DataFrame(implied_probs, columns=["Imp_H","Imp_D","Imp_A"]),
                  pd.DataFrame(model_probs, columns=["Mod_H","Mod_D","Mod_A"])], axis=1)

FAV_THRESHOLD = 0.8
MODEL_LOW_THRESHOLD = 0.5
GAP_THRESHOLD = 0.35
sus_mask = (
    ((comp["Imp_H"]>=FAV_THRESHOLD) & (comp["Mod_H"]<MODEL_LOW_THRESHOLD)) |
    ((comp["Imp_A"]>=FAV_THRESHOLD) & (comp["Mod_A"]<MODEL_LOW_THRESHOLD)) |
    ((abs(comp["Imp_H"]-comp["Mod_H"])>=GAP_THRESHOLD) |
     (abs(comp["Imp_D"]-comp["Mod_D"])>=GAP_THRESHOLD) |
     (abs(comp["Imp_A"]-comp["Mod_A"])>=GAP_THRESHOLD))
)
comp["suspicious"] = sus_mask
suspicious_matches = comp[comp["suspicious"]]
suspicious_path = ARTIFACT_DIR / "suspicious_matches.csv"
suspicious_matches.to_csv(suspicious_path, index=False)

# Save comparison
comp_path = ARTIFACT_DIR / "implied_vs_model_probs.csv"
comp.to_csv(comp_path, index=False)

print(f"Suspicious matches flagged: {len(suspicious_matches)}")
print(suspicious_matches.head())

print("All artifacts saved to", ARTIFACT_DIR)
