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
csv_path = BACKEND / "train" / "cleaned_filled_dataset.csv"
df = pd.read_csv(csv_path, parse_dates=["Date"])

# Drop unnecessary columns
df = df.drop(columns=["League"])

# === Encode Target Variable ===
le_result = LabelEncoder()
df["Result"] = le_result.fit_transform(df["Result"])
y = df["Result"]

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
    "Team_encoded", "Opponent_encoded","AHh","B365AHH","B365AHA","B365>2.5","B365<2.5"
]]
X.columns = [col.replace('<', 'lt').replace('>', 'gt') for col in X.columns]

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === MLflow Experiment ===
mlflow.set_experiment("Football Match Result Prediction")
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
        eval_metric="mlogloss",
        random_state=42
    )

    # === Train Model ===
    model.fit(X_train, y_train)

    # === Predictions ===
    y_probs = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # === Metrics ===
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    macro_metrics = report["macro avg"]
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_probs)

    # Log Metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("log_loss", loss)
    mlflow.log_metric("macro_precision", macro_metrics["precision"])
    mlflow.log_metric("macro_recall", macro_metrics["recall"])
    mlflow.log_metric("macro_f1", macro_metrics["f1-score"])

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Log Loss: {loss:.4f}")
    print(f"Macro Precision: {macro_metrics['precision']:.2%}")
    print(f"Macro Recall: {macro_metrics['recall']:.2%}")
    print(f"Macro F1-Score: {macro_metrics['f1-score']:.2%}")

    # === Save Artifacts ===
    ARTIFACT_DIR = ROOT / "artifacts"
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "xgb_model_2.pkl")
    model.save_model(str(ARTIFACT_DIR / "xgb_model_2.json"))
    joblib.dump(le_result, ARTIFACT_DIR / "result_encoder_2.pkl")
    joblib.dump(le_team, ARTIFACT_DIR / "team_encoder_2.pkl")

    # Log the actual files we created
    mlflow.log_artifact(str(ARTIFACT_DIR / "xgb_model_2.pkl"))
    mlflow.log_artifact(str(ARTIFACT_DIR / "xgb_model_2.json"))
    mlflow.log_artifact(str(ARTIFACT_DIR / "result_encoder_2.pkl"))
    mlflow.log_artifact(str(ARTIFACT_DIR / "team_encoder_2.pkl"))

    # === Precision-Recall Curves ===
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(le_result.classes_):
        precision, recall, _ = precision_recall_curve(y_test == i, y_probs[:, i])
        pr_auc = auc(recall, precision)
        mlflow.log_metric(f"{label}_pr_auc", pr_auc)
        plt.plot(recall, precision, label=f"{label} (AUC={pr_auc:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
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
                xticklabels=le_result.classes_, yticklabels=le_result.classes_)
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
    high_conf_df["Actual"] = le_result.inverse_transform(y_test)
    high_conf_df["Predicted"] = le_result.inverse_transform(y_pred)
    high_conf_df["Prob_Max"] = y_probs.max(axis=1)
    high_conf_df = high_conf_df[high_conf_df["Prob_Max"] >= 0.8]
    high_conf_path = ARTIFACT_DIR / "high_confidence_predictions.csv"
    high_conf_df.to_csv(high_conf_path, index=False)
    mlflow.log_artifact(str(high_conf_path))

    # Log Parameters
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_estimators", 500)
    mlflow.log_param("learning_rate", 0.03)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("subsample", 0.8)
    mlflow.log_param("colsample_bytree", 0.8)

    # Log Full Model
    mlflow.sklearn.log_model(model, "xgb_model_sklearn")
