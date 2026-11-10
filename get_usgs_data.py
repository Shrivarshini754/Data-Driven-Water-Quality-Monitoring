
import urllib.request
import pandas as pd

# This is the official API endpoint for the Water Quality Portal
# We are requesting stream data for Iowa (US:19) focusing on physical, nutrient, and biological parameters for 2022
url = (
    "https://www.waterqualitydata.us/data/Result/search?"
    "statecode=US%3A19&"
    "siteType=Stream&"
    "characteristicType=Nutrient&"
    "characteristicType=Physical&"
    "characteristicType=Biological&"
    "startDateLo=01-01-2022&"
    "startDateHi=12-31-2022&"
    "mimeType=csv"
)

print("Connecting to USGS servers to download dataset (this may take 30-45 seconds)...")

try:
    # Download the file directly via Python
    filename = "usgs_raw.csv"
    urllib.request.urlretrieve(url, filename)
    print(f"Success! Downloaded and saved as '{filename}'")
    
    # Quick verification check
    df = pd.read_csv(filename, low_memory=False, nrows=5)
    print("\nPreview of downloaded columns:")
    print(list(df.columns))

except Exception as e:
    print(f"Download failed due to: {e}")
    print("If the server timed out, government servers might be busy. Let me know and we can shorten the date range!")