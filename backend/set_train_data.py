import pandas as pd
import os
from glob import glob

# Directory where your files are
input_dir = "train"
output_file = os.path.join(input_dir, "all_leagues_merged_output.csv")

# Get all *_team_performance.csv files
input_files = glob(os.path.join(input_dir, "*_team_performance.csv"))

all_results = []

for input_file in input_files:
    league_code = os.path.basename(input_file).split("_team_performance.csv")[0].upper()
    print(f"🔄 Processing {league_code}...")

    try:
        # Load CSV with UTF-8 first, fallback to Latin-1
        try:
            df = pd.read_csv(input_file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(input_file, encoding="latin1", errors="replace")

        # Normalize column names
        df.columns = df.columns.str.strip()

        # Check required columns
        required_cols = {"Team", "Opponent", "Home/Away", "Date"}
        if not required_cols.issubset(df.columns):
            print(f"⚠️ Skipping {league_code}: missing required columns.")
            continue

        # Generate PairKey for home-away match identification
        df["PairKey"] = df.apply(
            lambda row: "-".join(sorted([str(row["Team"]), str(row["Opponent"])])), axis=1
        )

        # Split into home and away rows
        home_df = df[df["Home/Away"].str.upper() == "H"].copy()
        away_df = df[df["Home/Away"].str.upper() == "A"].copy()

        # Merge on Date + PairKey
        merged = pd.merge(
            home_df,
            away_df,
            on=["Date", "PairKey"],
            suffixes=("_H", "_A"),
            how="inner"
        )

        # Check if merge gave valid results
        if merged.empty:
            print(f"⚠️ No matching home-away pairs for {league_code}")
            continue

        # Ensure all required columns exist before selecting
        required_merge_cols = [
            "Date", "Team_H", "Opponent_H", "Result_H",
            "Home Odds_H", "Draw Odds_H", "Away Odds_H",
            "Bookmaker Margin_H",
            "Fair Odds(H)_H", "Fair Odds(D)_H", "Fair Odds(A)_H",
            "Risk Score_H", "Cumulative Score_H",
            "Risk Score_A", "Cumulative Score_A",
            "AHh_H", "B365AHH_H", "B365AHA_H", "B365>2.5_H", "B365<2.5_H","FTHG_H", "FTAG_H"
        ]

        missing_cols = [c for c in required_merge_cols if c not in merged.columns]
        if missing_cols:
            print(f"⚠️ Skipping {league_code}: missing columns {missing_cols}")
            continue

        # Select and rename desired columns
        result_df = merged[required_merge_cols]
        result_df.columns = [
            "Date", "Team", "Opponent", "Result",
            "Home Odds", "Draw Odds", "Away Odds",
            "Bookmaker Margin",
            "Fair Odds(H)", "Fair Odds(D)", "Fair Odds(A)",
            "Risk Score(H)", "Cumulative Score(H)",
            "Risk Score(A)", "Cumulative Score(A)",
            "AHh", "B365AHH", "B365AHA", "B365>2.5", "B365<2.5", "FTHG", "FTAG"
        ]

        # Add league code
        result_df["League"] = league_code

        all_results.append(result_df)

    except Exception as e:
        print(f"❌ Error processing {league_code}: {e}")

# Merge and save
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)

    # Save in UTF-8 to preserve characters
    final_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"✅ All merged into: {output_file}")
else:
    print("⚠️ No valid merged data found.")
