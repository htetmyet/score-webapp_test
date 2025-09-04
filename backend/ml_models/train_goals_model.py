import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    accuracy_score,
    log_loss
)

from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT.parent

# === Load Dataset ===
df = pd.read_csv(BACKEND / "train" / "cleaned_filled_dataset.csv", parse_dates=["Date"])

# Create target variable for Over/Under 2.5 goals
df['Tot_Goals'] = df['FTHG'] + df['FTAG']
df['Target'] = (df['Tot_Goals'] > 2).astype(int)
y = df['Target']

# === Encode Team Names ===
teams = pd.concat([df["Team"], df["Opponent"]]).unique()
le_team = LabelEncoder()
le_team.fit(teams)
df["Team_encoded"] = le_team.transform(df["Team"])
df["Opponent_encoded"] = le_team.transform(df["Opponent"])

# === Select Important Features ===
X = df[[
    "Home Odds", "Draw Odds", "Away Odds",
    "Bookmaker Margin",
    "Fair Odds(H)", "Fair Odds(D)", "Fair Odds(A)",
    "Risk Score(H)", "Cumulative Score(H)",
    "Risk Score(A)", "Cumulative Score(A)",
    "Team_encoded", "Opponent_encoded", "AHh",
    "B365AHH", "B365AHA", "B365>2.5", "B365<2.5"
]]
X.columns = [col.replace('<', 'lt').replace('>', 'gt') for col in X.columns]

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === MLflow Experiment ===
mlflow.set_experiment("Football Over/Under 2.5 Goals Prediction")
with mlflow.start_run():
    # === XGBoost Model with Regularization ===
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2,
        reg_alpha=1,
        min_child_weight=3,
        use_label_encoder=False,
        eval_metric="logloss",  # Changed to logloss for binary classification
        random_state=42
    )

    # === Train Model ===
    model.fit(X_train, y_train)

    # === Predictions ===
    y_probs = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # === Metrics ===
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_probs)

    # Log Metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("log_loss", loss)
    mlflow.log_metric("precision_over2.5", report['1']['precision'])
    mlflow.log_metric("recall_over2.5", report['1']['recall'])
    mlflow.log_metric("f1_over2.5", report['1']['f1-score'])

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Log Loss: {loss:.4f}")
    print(f"Over 2.5 Precision: {report['1']['precision']:.2%}")
    print(f"Over 2.5 Recall: {report['1']['recall']:.2%}")
    print(f"Over 2.5 F1-Score: {report['1']['f1-score']:.2%}")

    # === Save Artifacts ===
    ARTIFACT_DIR = ROOT / "artifacts_goals"
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "over_under_model.pkl")
    model.save_model(str(ARTIFACT_DIR / "over_under_model.json"))
    joblib.dump(le_team, ARTIFACT_DIR / "team_encoder.pkl")

    mlflow.log_artifact(str(ARTIFACT_DIR / "over_under_model.pkl"))
    mlflow.log_artifact(str(ARTIFACT_DIR / "over_under_model.json"))
    mlflow.log_artifact(str(ARTIFACT_DIR / "team_encoder.pkl"))

    # === Precision-Recall Curve ===
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_test, y_probs[:, 1])
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f"Over 2.5 Goals (AUC={pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for Over 2.5 Goals")
    plt.legend()
    plt.tight_layout()
    pr_path = ARTIFACT_DIR / "precision_recall_curve.png"
    plt.savefig(pr_path)
    mlflow.log_artifact(str(pr_path))
    plt.close()

    # === Confusion Matrix ===
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=["Under 2.5", "Over 2.5"],
                yticklabels=["Under 2.5", "Over 2.5"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cm_path = ARTIFACT_DIR / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    mlflow.log_artifact(str(cm_path))
    plt.close()

    # === Save High-Confidence Predictions (≥ 0.8) ===
    high_conf_df = X_test.copy()
    high_conf_df["Actual"] = ["Over 2.5" if x == 1 else "Under 2.5" for x in y_test]
    high_conf_df["Predicted"] = ["Over 2.5" if x == 1 else "Under 2.5" for x in y_pred]
    high_conf_df["Confidence"] = y_probs.max(axis=1)
    high_conf_df = high_conf_df[high_conf_df["Confidence"] >= 0.8]
    high_conf_path = ARTIFACT_DIR / "high_confidence_predictions.csv"
    high_conf_df.to_csv(high_conf_path, index=False)
    mlflow.log_artifact(str(high_conf_path))

    # Log Parameters
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("target", "Over/Under 2.5 Goals")
    mlflow.log_param("n_estimators", 500)
    mlflow.log_param("learning_rate", 0.03)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("subsample", 0.8)
    mlflow.log_param("colsample_bytree", 0.8)

    # Log Full Model
    mlflow.sklearn.log_model(model, "over_under_model")
