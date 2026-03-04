import pandas as pd

# =============================================================================
# Step 1: Load the main dataset and extract CBSA_Code from GEO_ID
# =============================================================================
# GEO_ID format: "310M700US10180" — the last 5 digits are the CBSA code
metro = pd.read_csv('data/derived data/metro_data.csv')
metro['CBSA_Code'] = metro['GEO_ID'].str[-5:].astype(int)

# =============================================================================
# Step 2: Load and clean pollution data
# =============================================================================
pollution = pd.read_csv('data/raw data/annual_aqi_by_cbsa_2024 2.csv')

# Keep only relevant columns
pollution = pollution[['CBSA Code', 'Median AQI']].copy()

# Rename to match merge key and target column name
pollution = pollution.rename(columns={
    'CBSA Code': 'CBSA_Code',
    'Median AQI': 'Median_AQI'
})

# Convert to numeric (handle any non-numeric values as NaN)
pollution['CBSA_Code'] = pd.to_numeric(pollution['CBSA_Code'], errors='coerce')
pollution['Median_AQI'] = pd.to_numeric(pollution['Median_AQI'], errors='coerce')

# =============================================================================
# Step 3: Merge pollution data onto the metro dataset (left join)
# =============================================================================
# Left join keeps all metro areas; unmatched rows get Median_AQI = NaN
merged = metro.merge(pollution, on='CBSA_Code', how='left')

# =============================================================================
# Step 4: Verification check
# =============================================================================
matched = merged['Median_AQI'].notna().sum()
unmatched = merged['Median_AQI'].isna().sum()

print(f"Matched AQI data for {matched} metropolitan areas.")
print(f"{unmatched} metropolitan areas have no pollution match (Median_AQI = NaN).")

# =============================================================================
# Step 5: Keep only the required columns
# =============================================================================
result = merged[['GEO_ID', 'NAME', 'Median_AQI']].copy()

# =============================================================================
# Step 6: Save output (overwrites previous pollution.csv)
# =============================================================================
result.to_csv('data/derived data/pollution.csv', index=False)
print("Saved: data/derived data/pollution.csv")
