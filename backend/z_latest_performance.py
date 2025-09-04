import os
import pandas as pd
import requests
from io import BytesIO
import chardet

# =============================
#  CONFIGURATION
# =============================
OUTPUT_DIR = "test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# League Codes & URLs
LEAGUES = {
    # England
    "epl2526": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "l12526": "https://www.football-data.co.uk/mmz4281/2526/E2.csv",
    "l22526": "https://www.football-data.co.uk/mmz4281/2526/E3.csv",
    "cham": "https://www.football-data.co.uk/mmz4281/2526/E1.csv",
    "conf": "https://www.football-data.co.uk/mmz4281/2526/EC.csv",

    # Spain
    "llg2526": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "lls2526": "https://www.football-data.co.uk/mmz4281/2526/SP2.csv",

    # Italy
    "sra2526": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "srb2526": "https://www.football-data.co.uk/mmz4281/2526/I2.csv",

    # Scotland
    "sc02526": "https://www.football-data.co.uk/mmz4281/2526/SC0.csv",
    "sc12526": "https://www.football-data.co.uk/mmz4281/2526/SC1.csv",
    "sc22526": "https://www.football-data.co.uk/mmz4281/2526/SC2.csv",
    "sc32526": "https://www.football-data.co.uk/mmz4281/2526/SC3.csv",

    # France
    "fr-dv1": "https://www.football-data.co.uk/mmz4281/2526/F1.csv",
    "fr-dv2": "https://www.football-data.co.uk/mmz4281/2526/F2.csv",

    # Germany
    "d22526": "https://www.football-data.co.uk/mmz4281/2526/D2.csv",

    # Belgium
    "b12526": "https://www.football-data.co.uk/mmz4281/2526/B1.csv",

    # Netherlands
    "n1": "https://www.football-data.co.uk/mmz4281/2526/N1.csv",

    # Portugal
    "p1": "https://www.football-data.co.uk/mmz4281/2526/P1.csv",

    # Turkey
    "t1": "https://www.football-data.co.uk/mmz4281/2526/T1.csv",
}

# =============================
#  FUNCTIONS
# =============================

def calculate_margin_and_fair_odds(row):
    try:
        h, d, a = row["B365H"], row["B365D"], row["B365A"]
        if h <= 1 or d <= 1 or a <= 1:
            return pd.Series([None, None, None, None])

        margin = (1/h + 1/d + 1/a) - 1
        fair_h = 1 / ((1/h) / (1 + margin))
        fair_d = 1 / ((1/d) / (1 + margin))
        fair_a = 1 / ((1/a) / (1 + margin))
        return pd.Series([margin, fair_h, fair_d, fair_a])
    except:
        return pd.Series([None, None, None, None])

def calculate_risk_score(result, fair_odd):
    if fair_odd is None or fair_odd <= 0:
        return 0
    return 1 - 1/fair_odd if result == "W" else -1/fair_odd

def detect_encoding(raw_data):
    detected = chardet.detect(raw_data)["encoding"]
    return detected if detected else "utf-8"

# =============================
#  MAIN PROCESS
# =============================

for league_code, url in LEAGUES.items():
    try:
        print(f"📌 Processing {league_code.upper()}...")

        # Download CSV file
        response = requests.get(url, timeout=15)
        raw_data = response.content
        encoding = detect_encoding(raw_data)
        print(f"   ➜ Detected encoding: {encoding}")

        # Read CSV safely
        df = pd.read_csv(BytesIO(raw_data), encoding=encoding, low_memory=False)

        # Ensure required columns exist
        if not all(col in df.columns for col in ["B365H", "B365D", "B365A"]):
            print(f"   ⚠️ Skipped {league_code} - Missing Bet365 odds")
            continue

        # Drop rows with missing odds
        df = df.dropna(subset=["B365H", "B365D", "B365A"])

        # Calculate bookmaker margin & fair odds
        df[["Bookmaker Margin", "Fair Odds (H)", "Fair Odds (D)", "Fair Odds (A)"]] = \
            df.apply(calculate_margin_and_fair_odds, axis=1)

        # Parse date safely
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

        # Prepare team-level records
        records = []
        for _, row in df.iterrows():
            for team_role in ["HomeTeam", "AwayTeam"]:
                if pd.isna(row[team_role]):
                    continue

                team = row[team_role]
                opponent = row["AwayTeam"] if team_role == "HomeTeam" else row["HomeTeam"]
                home_away = "H" if team_role == "HomeTeam" else "A"

                # Result W/D/L
                if row["FTR"] == "D":
                    result = "D"
                elif (row["FTR"] == "H" and team_role == "HomeTeam") or (row["FTR"] == "A" and team_role == "AwayTeam"):
                    result = "W"
                else:
                    result = "L"

                # Risk score
                fair_odd = row["Fair Odds (H)"] if home_away == "H" else row["Fair Odds (A)"]
                risk_score = calculate_risk_score(result, fair_odd)

                records.append([
                    row["Date"].strftime("%d/%m/%Y") if pd.notnull(row["Date"]) else "",
                    team, opponent, home_away, result,
                    row["B365H"], row["B365D"], row["B365A"],
                    row["Bookmaker Margin"], row["Fair Odds (H)"], row["Fair Odds (D)"], row["Fair Odds (A)"],
                    risk_score
                ])

        # Create DataFrame
        columns = [
            "Date", "Team", "Opponent", "Home/Away", "Result",
            "Home Odds", "Draw Odds", "Away Odds",
            "Bookmaker Margin", "Fair Odds(H)", "Fair Odds(D)", "Fair Odds(A)",
            "Risk Score"
        ]
        df_teams = pd.DataFrame(records, columns=columns)
        df_teams["Date"] = pd.to_datetime(df_teams["Date"], dayfirst=True, errors="coerce")
        df_teams = df_teams.sort_values(["Team", "Date"])
        df_teams["Cumulative Score"] = df_teams.groupby("Team")["Risk Score"].cumsum()

        # Save CSV
        output_path = os.path.join(OUTPUT_DIR, f"{league_code}_team_performance.csv")
        df_teams.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"   ✅ Saved to {output_path}")

    except Exception as e:
        print(f"   ❌ Error processing {league_code}: {e}")

print("\n🎯 All CSV files processed successfully!")
