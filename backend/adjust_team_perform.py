import os
import pandas as pd
import requests
from io import BytesIO
import chardet
from ftfy import fix_text

# Output directory
output_dir = "train"
os.makedirs(output_dir, exist_ok=True)

# Example leagues dictionary
leagues = {
    #ENGLAND (2122-2425)
    "epl": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "cham": "https://www.football-data.co.uk/mmz4281/2526/E1.csv",
    "conf": "https://www.football-data.co.uk/mmz4281/2526/EC.csv",
    "l12526": "https://www.football-data.co.uk/mmz4281/2526/E2.csv",
    "l22526": "https://www.football-data.co.uk/mmz4281/2526/E3.csv",

    #"epl2425"   :   "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    #"chp2425"   :   "https://www.football-data.co.uk/mmz4281/2425/E1.csv",
    #"l12425"    :   "https://www.football-data.co.uk/mmz4281/2425/E2.csv",
    #"l22425"    :   "https://www.football-data.co.uk/mmz4281/2425/E3.csv",
    #"conf2425"  :   "https://www.football-data.co.uk/mmz4281/2425/EC.csv",

    #"epl2324"   :   "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    #"chp2324"   :   "https://www.football-data.co.uk/mmz4281/2324/E1.csv",
    #"l12324"    :   "https://www.football-data.co.uk/mmz4281/2324/E2.csv",
    #"l22324"    :   "https://www.football-data.co.uk/mmz4281/2324/E3.csv",
    #"conf2324"  :   "https://www.football-data.co.uk/mmz4281/2324/EC.csv",

    #"epl2223"   :   "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    #"chp2223"   :   "https://www.football-data.co.uk/mmz4281/2223/E1.csv",
    #"l12223"    :   "https://www.football-data.co.uk/mmz4281/2223/E2.csv",
    #"l22223"    :   "https://www.football-data.co.uk/mmz4281/2223/E3.csv",
    #"conf2223"  :   "https://www.football-data.co.uk/mmz4281/2223/EC.csv",

    #"epl2122"   :   "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
    #"chp2122"   :   "https://www.football-data.co.uk/mmz4281/2122/E1.csv",
    #"l12122"    :   "https://www.football-data.co.uk/mmz4281/2122/E2.csv",
    #"l22222"    :   "https://www.football-data.co.uk/mmz4281/2122/E3.csv",
    #"conf2122"  :   "https://www.football-data.co.uk/mmz4281/2122/EC.csv",

    #SPAIN (2122-2526)
    "llg2526"   :   "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "lls2526"   :   "https://www.football-data.co.uk/mmz4281/2526/SP2.csv",
    #"llg2425"   :   "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
    #"lls2425"   :   "https://www.football-data.co.uk/mmz4281/2425/SP2.csv",
    #"llg2324"   :   "https://www.football-data.co.uk/mmz4281/2324/SP1.csv",
    #"lls2324"   :   "https://www.football-data.co.uk/mmz4281/2324/SP2.csv",
    #"llg2223"   :   "https://www.football-data.co.uk/mmz4281/2223/SP1.csv",
    #"lls2223"   :   "https://www.football-data.co.uk/mmz4281/2223/SP2.csv",
    #"llg2122"   :   "https://www.football-data.co.uk/mmz4281/2122/SP1.csv",
    #"lls2122"   :   "https://www.football-data.co.uk/mmz4281/2122/SP2.csv",

    #ITALY (2122-2425)
    "sra2526"   :   "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "srb2526"   :   "https://www.football-data.co.uk/mmz4281/2526/I2.csv",
    #"sra2425"   :   "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
    #"srb2425"   :   "https://www.football-data.co.uk/mmz4281/2425/I2.csv",
    #"sra2324"   :   "https://www.football-data.co.uk/mmz4281/2324/I1.csv",
    #"srb2324"   :   "https://www.football-data.co.uk/mmz4281/2324/I2.csv",
    #"sra2223"   :   "https://www.football-data.co.uk/mmz4281/2223/I1.csv",
    #"srb2223"   :   "https://www.football-data.co.uk/mmz4281/2223/I2.csv",
    #"sra2122"   :   "https://www.football-data.co.uk/mmz4281/2122/I1.csv",
    #"srb2122"   :   "https://www.football-data.co.uk/mmz4281/2122/I2.csv",

    #SCOTLAND (2122-2425)
    "sc02526": "https://www.football-data.co.uk/mmz4281/2526/SC0.csv",
    "sc12526": "https://www.football-data.co.uk/mmz4281/2526/SC1.csv",
    "sc22526": "https://www.football-data.co.uk/mmz4281/2526/SC2.csv",
    "sc32526": "https://www.football-data.co.uk/mmz4281/2526/SC3.csv",

    #"sc02425"   :   "https://www.football-data.co.uk/mmz4281/2425/SC0.csv",
    #"sc12425"   :   "https://www.football-data.co.uk/mmz4281/2425/SC1.csv",
    #"sc22425"   :   "https://www.football-data.co.uk/mmz4281/2425/SC2.csv",
    #"sc32425"   :   "https://www.football-data.co.uk/mmz4281/2425/SC3.csv",

    #"sc02324"   :   "https://www.football-data.co.uk/mmz4281/2324/SC0.csv",
    #"sc12324"   :   "https://www.football-data.co.uk/mmz4281/2324/SC1.csv",
    #"sc22324"   :   "https://www.football-data.co.uk/mmz4281/2324/SC2.csv",
    #"sc32324"   :   "https://www.football-data.co.uk/mmz4281/2324/SC3.csv",

    #"sc02223"   :   "https://www.football-data.co.uk/mmz4281/2223/SC0.csv",
    #"sc12223"   :   "https://www.football-data.co.uk/mmz4281/2223/SC1.csv",
    #"sc22223"   :   "https://www.football-data.co.uk/mmz4281/2223/SC2.csv",
    #"sc32223"   :   "https://www.football-data.co.uk/mmz4281/2223/SC3.csv",

    #"sc02122"   :   "https://www.football-data.co.uk/mmz4281/2122/SC0.csv",
    #"sc12122"   :   "https://www.football-data.co.uk/mmz4281/2122/SC1.csv",
    #"sc22122"   :   "https://www.football-data.co.uk/mmz4281/2122/SC2.csv",
    #"sc32122"   :   "https://www.football-data.co.uk/mmz4281/2122/SC3.csv",

    #FRANCE (2122-2425)
    "f1-dv1"    :   "https://www.football-data.co.uk/mmz4281/2526/F1.csv",
    "fr-dv2"    :   "https://www.football-data.co.uk/mmz4281/2526/F2.csv",
    #"f12425"    :   "https://www.football-data.co.uk/mmz4281/2425/F1.csv",
    #"f22425"    :   "https://www.football-data.co.uk/mmz4281/2425/F2.csv",
    #"f12324"    :   "https://www.football-data.co.uk/mmz4281/2324/F1.csv",
    #"f22324"    :   "https://www.football-data.co.uk/mmz4281/2324/F2.csv",
    #"f12223"    :   "https://www.football-data.co.uk/mmz4281/2223/F1.csv",
    #"f22223"    :   "https://www.football-data.co.uk/mmz4281/2223/F2.csv",
    #"f12122"    :   "https://www.football-data.co.uk/mmz4281/2122/F1.csv",
    #"f22122"    :   "https://www.football-data.co.uk/mmz4281/2122/F2.csv",

    #NETHERLAND (2021-2425)
    "n1"    :   "https://www.football-data.co.uk/mmz4281/2526/N1.csv",
    #"ne2425"    :   "https://www.football-data.co.uk/mmz4281/2425/N1.csv",
    #"ne2324"    :   "https://www.football-data.co.uk/mmz4281/2324/N1.csv",
    #"ne2223"    :   "https://www.football-data.co.uk/mmz4281/2223/N1.csv",
    #"ne2122"    :   "https://www.football-data.co.uk/mmz4281/2122/N1.csv",
    #"ne2021"    :   "https://www.football-data.co.uk/mmz4281/2021/N1.csv",

    #BELGIUM (2021-2425)
    "b12526"    :   "https://www.football-data.co.uk/mmz4281/2526/B1.csv",
    "bj2425"    :   "https://www.football-data.co.uk/mmz4281/2526/B1.csv",
    #"bj2324"    :   "https://www.football-data.co.uk/mmz4281/2324/B1.csv",
    #"bj2223"    :   "https://www.football-data.co.uk/mmz4281/2223/B1.csv",
    #"bj2122"    :   "https://www.football-data.co.uk/mmz4281/2122/B1.csv",
    #"bj2021"    :   "https://www.football-data.co.uk/mmz4281/2021/B1.csv",

    #PORTUGAL (2021-2425)
    "p1"        :   "https://www.football-data.co.uk/mmz4281/2526/P1.csv",
    #"pp2425"    :   "https://www.football-data.co.uk/mmz4281/2425/P1.csv",
    #"pp2324"    :   "https://www.football-data.co.uk/mmz4281/2324/P1.csv",
    #"pp2223"    :   "https://www.football-data.co.uk/mmz4281/2223/P1.csv",
    #"pp2122"    :   "https://www.football-data.co.uk/mmz4281/2122/P1.csv",
    #"pp2021"    :   "https://www.football-data.co.uk/mmz4281/2021/P1.csv",

    #TURKEY (2021-2425)
    "t1"        :   "https://www.football-data.co.uk/mmz4281/2526/T1.csv",
    #"tk2425"    :   "https://www.football-data.co.uk/mmz4281/2425/T1.csv",
    #"tk2324"    :   "https://www.football-data.co.uk/mmz4281/2324/T1.csv",
    #"tk2223"    :   "https://www.football-data.co.uk/mmz4281/2223/T1.csv",
    #"tk2122"    :   "https://www.football-data.co.uk/mmz4281/2122/T1.csv",
    #"tk2021"    :   "https://www.football-data.co.uk/mmz4281/2021/T1.csv",

    #GREECE (2021-2425)
    "gc2526"    :   "https://www.football-data.co.uk/mmz4281/2526/G1.csv",
    #"gc2425"    :   "https://www.football-data.co.uk/mmz4281/2425/G1.csv",
    #"gc2324"    :   "https://www.football-data.co.uk/mmz4281/2324/G1.csv",
    #"gc2223"    :   "https://www.football-data.co.uk/mmz4281/2223/G1.csv",
    #"gc2122"    :   "https://www.football-data.co.uk/mmz4281/2122/G1.csv",
    #"gc2021"    :   "https://www.football-data.co.uk/mmz4281/2021/G1.csv",

    #GERMANY (2021-2425)
    "d22526"    :   "https://www.football-data.co.uk/mmz4281/2526/D2.csv",
    #"bl2425"    :   "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
    #"bt2425"    :   "https://www.football-data.co.uk/mmz4281/2425/D2.csv",
    #"bl2324"    :   "https://www.football-data.co.uk/mmz4281/2324/D1.csv",
    #"bt2324"    :   "https://www.football-data.co.uk/mmz4281/2324/D2.csv",
    #"bl2223"    :   "https://www.football-data.co.uk/mmz4281/2223/D1.csv",
    #"bt2223"    :   "https://www.football-data.co.uk/mmz4281/2223/D2.csv",
    #"bl2122"    :   "https://www.football-data.co.uk/mmz4281/2122/D1.csv",
    #"bt2122"    :   "https://www.football-data.co.uk/mmz4281/2122/D2.csv",

    #All Others
    #"other2526": "https://www.football-data.co.uk/mmz4281/2526/Latest_Results.csv",
}

# Function to calculate bookmaker margin and fair odds
def calculate_margin_and_fair_odds(row):
    h, d, a = row["B365H"], row["B365D"], row["B365A"]
    margin = (1/h + 1/d + 1/a) - 1
    fair_h = 1 / ((1/h) / (1 + margin))
    fair_d = 1 / ((1/d) / (1 + margin))
    fair_a = 1 / ((1/a) / (1 + margin))
    return pd.Series([margin, fair_h, fair_d, fair_a])

# Function to calculate risk score
def calculate_risk_score(result, fair_odd):
    return 1 - 1/fair_odd if result == "W" else -1/fair_odd

# Process each league
for league_code, url in leagues.items():
    print(f"🔄 Processing {league_code}...")

    # Download CSV as raw bytes
    response = requests.get(url)
    raw_bytes = response.content

    # Auto-detect encoding
    detection = chardet.detect(raw_bytes)
    encoding = detection["encoding"] or "ISO-8859-1"
    print(f"Detected encoding for {league_code}: {encoding}")

    # Decode correctly using detected encoding
    decoded_content = raw_bytes.decode(encoding, errors="replace")

    # Read CSV into pandas
    data = pd.read_csv(BytesIO(decoded_content.encode("utf-8")))

    # Drop rows without odds
    data = data.dropna(subset=["B365H", "B365D", "B365A"])

    # Fix ALL text columns (handles Preußen Münster, München, etc.)
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = data[col].apply(lambda x: fix_text(str(x)) if isinstance(x, str) else x)

    # Calculate bookmaker margin & fair odds
    data[["Bookmaker Margin", "Fair Odds (H)", "Fair Odds (D)", "Fair Odds (A)"]] = data.apply(
        calculate_margin_and_fair_odds, axis=1
    )
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)

    # Build team performance dataset
    records = []
    for _, row in data.iterrows():
        for team_role in ["HomeTeam", "AwayTeam"]:
            team = row[team_role]
            opponent = row["AwayTeam"] if team_role == "HomeTeam" else row["HomeTeam"]
            home_away = "H" if team_role == "HomeTeam" else "A"

            # Determine match result
            if row["FTR"] == "D":
                result = "D"
            elif (row["FTR"] == "H" and team_role == "HomeTeam") or (row["FTR"] == "A" and team_role == "AwayTeam"):
                result = "W"
            else:
                result = "L"

            fair_odd = row["Fair Odds (H)"] if home_away == "H" else row["Fair Odds (A)"]
            risk_score = calculate_risk_score(result, fair_odd)

            records.append([
                row["Date"].strftime("%d/%m/%Y"), team, opponent, home_away, result,
                row["B365H"], row["B365D"], row["B365A"], row["AHh"], row["B365AHH"], row["B365AHA"], row["B365>2.5"], row["B365<2.5"], row["FTHG"], row["FTAG"],
                row["Bookmaker Margin"], row["Fair Odds (H)"], row["Fair Odds (D)"], row["Fair Odds (A)"],
                risk_score
            ])

    # Create DataFrame
    columns = [
        "Date", "Team", "Opponent", "Home/Away", "Result",
        "Home Odds", "Draw Odds", "Away Odds","AHh","B365AHH","B365AHA","B365>2.5","B365<2.5","FTHG","FTAG",
        "Bookmaker Margin", "Fair Odds(H)", "Fair Odds(D)", "Fair Odds(A)",
        "Risk Score"
    ]
    df_teams = pd.DataFrame(records, columns=columns)

    # Sort by date per team
    df_teams["Date"] = pd.to_datetime(df_teams["Date"], dayfirst=True)
    df_teams = df_teams.sort_values(["Team", "Date"])

    # Shift & compute cumulative score
    df_teams["Shifted Risk Score"] = df_teams.groupby("Team")["Risk Score"].shift(1, fill_value=0)
    df_teams["Cumulative Score"] = df_teams.groupby("Team")["Shifted Risk Score"].cumsum()
    df_teams["Risk Score"] = df_teams["Shifted Risk Score"]
    df_teams.drop(columns=["Shifted Risk Score"], inplace=True)

    # Save cleaned file in UTF-8 (proper!)
    output_path = os.path.join(output_dir, f"{league_code}_team_performance.csv")
    df_teams.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ Saved fixed CSV: {output_path}")

print("\n🎯 All leagues processed successfully!")
