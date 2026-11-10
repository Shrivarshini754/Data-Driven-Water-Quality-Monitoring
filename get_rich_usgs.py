
import urllib.request
import pandas as pd

# Expanded USGS API query targeting multiple nutrients and physical parameters for streams in Iowa (State Code US:19)
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

print("Fetching multi-target dataset from USGS servers...")
filename = "usgs_raw_rich.csv"
urllib.request.urlretrieve(url, filename)
print(f"Download complete! Saved as '{filename}'")