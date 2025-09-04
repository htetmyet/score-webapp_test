import pandas as pd
import glob
import os
import chardet

# Function to detect file encoding automatically
def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read(50000))
    return result["encoding"]

# Path to the directory containing CSV files
input_path = "test/"

# Find all CSV files in 'test/' whose name contains '_team_performance.csv'
input_files = glob.glob(os.path.join(input_path, "*_team_performance.csv"))

if not input_files:
    print("⚠️ No CSV files found in 'test/' folder.")
    exit()

df_list = []
for f in input_files:
    try:
        encoding = detect_encoding(f)
        print(f"🔹 Reading {f} using encoding: {encoding}")
        df_temp = pd.read_csv(f, parse_dates=["Date"], dayfirst=False, encoding=encoding)
        df_list.append(df_temp)
    except Exception as e:
        print(f"❌ Error reading {f}: {e}")

# Combine all data
df = pd.concat(df_list, ignore_index=True)
df.sort_values("Date", inplace=True)

# Initialize dictionary for tracking team stats
team_stats = {}

# Loop through each match
for _, row in df.iterrows():
    team = row["Team"]
    result = row["Result"]
    fair_odds = row["Fair Odds(H)"] if row["Home/Away"] == "H" else row["Fair Odds(A)"]

    # Compute risk-adjusted hot/cold score
    score = 1 - 1 / fair_odds if result == "W" else -1 / fair_odds

    # Initialize team record if not exists
    if team not in team_stats:
        team_stats[team] = {
            "Wins": 0,
            "Draws": 0,
            "Losses": 0,
            "Matches": 0,
            "Cumulative Score": 0.0,
            "Scores": []
        }

    team_stats[team]["Matches"] += 1
    team_stats[team]["Cumulative Score"] += score
    team_stats[team]["Scores"].append(score)

    if result == "W":
        team_stats[team]["Wins"] += 1
    elif result == "D":
        team_stats[team]["Draws"] += 1
    else:
        team_stats[team]["Losses"] += 1

# Prepare output rows
output_rows = []
for team, stats in team_stats.items():
    avg_risk_score = stats["Cumulative Score"] / stats["Matches"] if stats["Matches"] else 0
    latest_score = stats["Scores"][-1] if stats["Scores"] else 0

    output_rows.append({
        "Team": team,
        "Total Matches": stats["Matches"],
        "Wins": stats["Wins"],
        "Draws": stats["Draws"],
        "Losses": stats["Losses"],
        "Cumulative Score": round(stats["Cumulative Score"], 3),
        "Avg Risk Score": round(avg_risk_score, 3),
        "Risk Score": round(latest_score, 3)
    })

# Save final output with UTF-8 BOM to fix special characters
output_df = pd.DataFrame(output_rows)
output_file = os.path.join(input_path, "updated_team_form_ref.csv")
output_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"✅ updated_team_form_ref.csv has been created successfully at {output_file}")
print("All Team Performance Forms Updated!")
