
### Clean Data: BLS 2025-Q2 Employment Dataset
# Source: U.S. Bureau of Labor Statistics (BLS)
# Geographic Coverage: Metropolitan Statistical Areas (MSAs)
### Filters Applied:
# Area Code: Only FIPS starting with "C" (CBSA/MSA codes).
# Area Name: Only titles containing the specific string "MSA" (Excludes MicroSAs and CSAs).
# Ownership (own_code): Set to "0" (Total Covered: Private + Government).
# Industry (industry_code): Set to "10" (Total, all industries).


import pandas as pd

# 1. Define File Names
input_file_employment = '2025_Q2_Employment.csv'
input_file_titles = 'area_titles.csv'
output_file = 'Cleaned_MSA_Employment_Growth.csv' 

# 2. Load the datasets
# We read FIPS as strings to preserve 'C' prefixes and leading zeros.
df_2025 = pd.read_csv(input_file_employment, dtype={'area_fips': str})
df_titles = pd.read_csv(input_file_titles, dtype={'area_fips': str})

# 3. Filter for Metropolitan Areas (MSAs)
# Step A: Keep only codes starting with 'C'.
df_msa_titles = df_titles[df_titles['area_fips'].str.startswith('C', na=False)].copy()

# Step B: Keep only titles that contain the specific string 'MSA'.
# This automatically excludes MicroSAs and CSAs.
df_msa_titles = df_msa_titles[df_msa_titles['area_title'].str.contains('MSA', na=False)]

# 4. Merge Employment data with the filtered Area Titles
df_merged = pd.merge(
    df_2025, 
    df_msa_titles[['area_fips', 'area_title']], 
    on='area_fips', 
    how='inner'
)

# 5. Filter for Ownership and Industry
# Using .astype(str) to ensure '0' (Total) and '10' (All Industries) match.
df_filtered = df_merged[
    (df_merged['own_code'].astype(str) == '0') & 
    (df_merged['industry_code'].astype(str) == '10')
].copy()

# 6. Select final columns for the analysis
# oty_month3_emplvl_pct_chg represents the 12-month growth as of June.
columns_to_keep = [
    'area_fips', 
    'area_title', 
    'agglvl_code', 
    'oty_month3_emplvl_pct_chg'
]
df_final = df_filtered[columns_to_keep]

# 7. EXPORT to CSV
# index=False prevents pandas from adding an unnecessary row index column.
df_final.to_csv(output_file, index=False)

print(f"Success! The cleaned data for {len(df_final)} MSA regions has been exported to: {output_file}")