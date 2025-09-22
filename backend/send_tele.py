import os
import pandas as pd
import requests
from io import BytesIO

# ==== CONFIGURATION ====
# Read sensitive values from environment variables
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CSV_FOLDER = os.getenv("PREDICTIONS_DIR", "ml_models/predictions/")   # Folder containing source CSV files

# ✅ List of CSV files you want to send (exact names)
CSV_FILES_TO_SEND = [
    "pred_goals_val_bets.csv",
    "pred_new_high_conf.csv",
    "pred_ah_high_conf.csv"   
]

# Add draw-specific prediction files
CSV_FILES_TO_SEND.extend([
    "pred_ah_draw_2x.csv",
    "pred_new_draw_only.csv",
])

# Columns we want to keep if available
SELECTED_COLUMNS = ["Date", "Team", "Opponent", "Prob_Over", "Prob_Under", "Prob_HomeWin","Prob_Draw","Prob_AwayWin"]

# Telegram API endpoint
TELEGRAM_API_URL = None
if BOT_TOKEN:
    TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"

def send_selected_csv_files():
    if not BOT_TOKEN or not CHAT_ID:
        print("❌ Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variables.")
        return
    if TELEGRAM_API_URL is None:
        print("❌ Cannot construct Telegram API URL. Check TELEGRAM_BOT_TOKEN.")
        return
    for file_name in CSV_FILES_TO_SEND:
        file_path = os.path.join(CSV_FOLDER, file_name)

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"⚠️ Skipping: {file_name} (file not found)")
            continue

        try:
            # Read CSV safely (handle encoding issues)
            df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False)
            except Exception as e:
                print(f"⚠️ Skipping {file_name}: Cannot read file -> {e}")
                continue
        except Exception as e:
            print(f"⚠️ Skipping {file_name}: Cannot read file -> {e}")
            continue

        # Select only available columns from the CSV
        available_columns = [col for col in SELECTED_COLUMNS if col in df.columns]
        filtered_df = df[available_columns]

        # If no matching columns, skip file
        if filtered_df.empty:
            print(f"⚠️ Skipping {file_name} (no matching columns)")
            continue

        # Save filtered CSV in memory (no temp file needed)
        buffer = BytesIO()
        filtered_df.to_csv(buffer, index=False)
        buffer.seek(0)

        # Send CSV file to Telegram
        print(f"📤 Sending: {file_name} (filtered)...")
        try:
            response = requests.post(
                TELEGRAM_API_URL,
                data={"chat_id": CHAT_ID},
                files={"document": (f"tg_{file_name}", buffer)}
            )

            if response.status_code == 200:
                print(f"✅ Sent successfully: tg_{file_name}")
            else:
                print(f"❌ Failed to send {file_name}: {response.text}")

        except Exception as e:
            print(f"⚠️ Error sending {file_name}: {e}")

if __name__ == "__main__":
    send_selected_csv_files()
