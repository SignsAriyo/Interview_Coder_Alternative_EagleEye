import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Calculate the exact directory where main.py resides
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / '.env'

# Force python-dotenv to read from the absolute path
load_dotenv(dotenv_path=ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print(f"\n[ERROR] Could not find GEMINI_API_KEY inside environment variables.")
    print(f"[DEBUG] Looked inside absolute file location: {ENV_PATH}")
    sys.exit(1)

print(f"[SUCCESS] Security check passed. Secret Key loaded from local environment storage.")

# ... (the rest of your main.py code continues here)