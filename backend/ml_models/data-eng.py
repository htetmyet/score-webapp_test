import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import shapiro, norm
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT.parent

# === 1. Load CSV ===
df = pd.read_csv(BACKEND / "train" / "all_leagues_merged_output.csv")

# === 2. Add tot_goals ===
df["tot_goals"] = df["FTHG"] + df["FTAG"]

# === 3. Check initial missing values ===
print("\n🔍 Missing values per column BEFORE filling:")
print(df.isnull().sum())

# === 4. Helper function to fill missing values ===
def fill_missing_knn(df, target_col, feature_cols, n_neighbors=5):
    """Fill missing numeric values using KNN regression."""
    valid_df = df.dropna(subset=feature_cols)
    known = valid_df[valid_df[target_col].notna()]
    missing = valid_df[valid_df[target_col].isna()]

    if known.empty or missing.empty:
        print(f"⚠️ Skipping {target_col}: insufficient data for KNN")
        return df

    X_train = known[feature_cols]
    y_train = known[target_col]
    X_missing = missing[feature_cols]

    # Use KNN Regressor for continuous columns
    model = KNeighborsRegressor(n_neighbors=min(n_neighbors, len(known)))
    model.fit(X_train, y_train)
    predicted = model.predict(X_missing)

    # Fill back predicted values
    df.loc[missing.index, target_col] = predicted

    # Fallback fill: median
    df[target_col].fillna(df[target_col].median(), inplace=True)

    print(f"✅ Filled missing values in {target_col}")
    return df

# === 5. Define features for odds-driven filling ===
odds_features = ["Home Odds", "Draw Odds", "Away Odds"]

# === 6. Fill missing columns ===
for col in ["AHh", "B365AHH", "B365AHA", "B365>2.5", "B365<2.5"]:
    df = fill_missing_knn(df, target_col=col, feature_cols=odds_features)

# === 7. Check missing values after filling ===
print("\n🔍 Missing values per column AFTER filling:")
print(df.isnull().sum())

# === 8. Save cleaned data ===
df.to_csv(BACKEND / "train" / "cleaned_filled_dataset.csv", index=False)
print("\n✅ All missing values handled successfully. File saved: cleaned_filled_dataset.csv")

# === 3. Function to examine distribution ===
def check_distribution(df, col):
    data = df[col].dropna()
    stat, p_value = shapiro(data)
    skewness = data.skew()
    normality = "Normally Distributed" if p_value > 0.05 else "Not Normally Distributed"
    print(f"\n📌 {col} Distribution:")
    print(f"   • Shapiro-Wilk p-value: {p_value:.5f} → {normality}")
    print(f"   • Skewness: {skewness:.3f}")
    return p_value, skewness

'''
# === 4. Outlier removal function ===
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    initial_rows = df.shape[0]
    df_clean = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    removed_rows = initial_rows - df_clean.shape[0]
    print(f"   • Removed {removed_rows} outliers from {col}")
    return df_clean


# === 5. Analyze FTHG & FTAG first ===
for col in ["FTHG", "FTAG"]:
    check_distribution(df, col)
    df = remove_outliers(df, col)

# === 6. Recalculate tot_goals after outlier removal ===
df["tot_goals"] = df["FTHG"] + df["FTAG"]

# === 7. Analyze tot_goals ===
check_distribution(df, "tot_goals")
df = remove_outliers(df, "tot_goals")

# === 8. Final check of distributions ===
print("\n🔍 Final Distribution Check After Outlier Removal:")
for col in ["FTHG", "FTAG", "tot_goals"]:
    check_distribution(df, col)

# === 9. Save cleaned dataset ===
df.to_csv("../train/cleaned_filled_dataset.csv", index=False)
print("\n✅ Cleaned dataset saved as cleaned_dataset_no_outliers.csv")
'''

# === 10. Plot normal distributions ===
def plot_normal_distribution(df, col):
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], bins=30, kde=False, stat="density", color="skyblue", edgecolor="black")

    # Fit normal distribution curve
    mu, sigma = norm.fit(df[col])
    x = np.linspace(df[col].min(), df[col].max(), 100)
    plt.plot(x, norm.pdf(x, mu, sigma), color="red", linewidth=2, label=f"N({mu:.2f}, {sigma:.2f})")

    plt.title(f"Normal Distribution - {col}", fontsize=14)
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{col}_distribution.png", dpi=300)
    plt.show()


print("\n📊 Generating distribution plots...")
for col in ["FTHG", "FTAG", "tot_goals"]:
    plot_normal_distribution(df, col)
print("\n✅ Distribution plots saved as PNG files")

# Load cleaned dataset
df = pd.read_csv(BACKEND / "train" / "cleaned_filled_dataset.csv")

# Select numeric columns for correlation
numeric_cols = df.select_dtypes(include='number').columns.tolist()
numeric_cols.remove('tot_goals')  # target separate
numeric_cols.remove('FTHG')
numeric_cols.remove('FTAG')

# Correlation with tot_goals
corr_tot_goals = df[numeric_cols + ['tot_goals']].corr()['tot_goals'].sort_values(ascending=False)
print("\n📌 Correlation with tot_goals:")
print(corr_tot_goals)

# Heatmap of correlations
plt.figure(figsize=(12, 8))
sns.heatmap(df[numeric_cols + ['tot_goals']].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
