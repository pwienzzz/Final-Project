### Clean Data: BEA 2024 Metropolitan Regional Price Parities (MARPP) Dataset
# Source: U.S. Bureau of Economic Analysis (BEA)
# Geographic Coverage: Metropolitan Statistical Areas (MSAs) within the United States
# Only need the most recent year data: Year 2024
# We want to combine three services line codes into one: We want the average from three services types - utilities, housing, and others.

import pandas as pd

# 1. Define File Names
input_file = 'MARPP_MSA_2008_2024.csv' 
output_file = 'BEA_Metro_RPP_2024_Clean.csv' 

# 2. Load the Dataset
df = pd.read_csv(input_file, na_values=['(NA)'])
df.columns = df.columns.str.strip()

# 3. Handle Services (Merge LineCodes 3.0, 4.0, 5.0)
service_codes = [3.0, 4.0, 5.0]
services_df = df[df['LineCode'].isin(service_codes)].copy()

# Calculate the mean for 2024 across the three service categories for each city
combined_services = services_df.groupby(['GeoFIPS', 'GeoName'])['2024'].mean().reset_index()
combined_services['LineCode'] = '3-5'
combined_services['Description'] = 'RPPs: Services'

# 4. Keep "All items" (1.0) and "Goods" (2.0)
others = df[df['LineCode'].isin([1.0, 2.0])][['GeoFIPS', 'GeoName', 'LineCode', 'Description', '2024']]

# 5. Combine and Save
# We stack the original 1 & 2 rows with our new merged Services row
final_df = pd.concat([others, combined_services], ignore_index=True)

# Sort by GeoName so all data for one city stays together
final_df = final_df.sort_values(by=['GeoName', 'LineCode'])

final_df.to_csv(output_file, index=False)

print(f"Done! Created '{output_file}' with merged Services category.")

