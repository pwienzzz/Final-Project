import pandas as pd

# =============================================================================
# 1. Clean income.csv
# =============================================================================
income = pd.read_csv('data/raw data/income.csv', header=1)
income = income.rename(columns={'Geography': 'GEO_ID', 'Geographic Area Name': 'NAME'})
income = income[['GEO_ID', 'NAME',
                 "Estimate!!Total:!!Bachelor's degree",
                 "Estimate!!Total:!!Graduate or professional degree"]].copy()
income = income.rename(columns={
    "Estimate!!Total:!!Bachelor's degree": 'bachelor_degree',
    "Estimate!!Total:!!Graduate or professional degree": 'graduate_degree'
})
income['bachelor_degree'] = pd.to_numeric(income['bachelor_degree'], errors='coerce')
income['graduate_degree'] = pd.to_numeric(income['graduate_degree'], errors='coerce')

# =============================================================================
# 2. Clean rent.csv
# =============================================================================
rent = pd.read_csv('data/raw data/rent.csv', header=1)
rent = rent.rename(columns={'Geography': 'GEO_ID', 'Geographic Area Name': 'NAME'})
rent = rent.drop(columns=[c for c in rent.columns if 'Margin of Error' in str(c)])
rent = rent.rename(columns={'Estimate!!Median gross rent': 'median_rent'})
rent['median_rent'] = pd.to_numeric(rent['median_rent'], errors='coerce')
rent = rent[['GEO_ID', 'median_rent']].copy()

# =============================================================================
# 3. Clean Travel Time to Work.csv
# =============================================================================
commute = pd.read_csv('data/raw data/Travel Time to Work.csv', header=1)
commute = commute.rename(columns={'Geography': 'GEO_ID', 'Geographic Area Name': 'NAME'})
commute = commute.drop(columns=[c for c in commute.columns if 'Margin of Error' in str(c)])

# Midpoints for each commute interval (minutes).
# The data splits "60+ min" into two bins; both receive midpoint 65 per spec.
commute_midpoints = {
    'Estimate!!Total:!!Less than 5 minutes': 2.5,
    'Estimate!!Total:!!5 to 9 minutes': 7,
    'Estimate!!Total:!!10 to 14 minutes': 12,
    'Estimate!!Total:!!15 to 19 minutes': 17,
    'Estimate!!Total:!!20 to 24 minutes': 22,
    'Estimate!!Total:!!25 to 29 minutes': 27,
    'Estimate!!Total:!!30 to 34 minutes': 32,
    'Estimate!!Total:!!35 to 39 minutes': 37,
    'Estimate!!Total:!!40 to 44 minutes': 42,
    'Estimate!!Total:!!45 to 59 minutes': 52,
    'Estimate!!Total:!!60 to 89 minutes': 65,
    'Estimate!!Total:!!90 or more minutes': 65,
}

for col in commute_midpoints:
    commute[col] = pd.to_numeric(commute[col], errors='coerce')

commute_cols = list(commute_midpoints.keys())
commute['avg_commute_time'] = (
    sum(commute[col] * mp for col, mp in commute_midpoints.items())
    / commute[commute_cols].sum(axis=1)
)
commute = commute[['GEO_ID', 'avg_commute_time']].copy()

# =============================================================================
# 4. Clean Tenure by Occupants per Room.csv
# =============================================================================
tenure = pd.read_csv('data/raw data/Tenure by Occupants per Room.csv', header=1)
tenure = tenure.rename(columns={'Geography': 'GEO_ID', 'Geographic Area Name': 'NAME'})
tenure = tenure.drop(columns=[c for c in tenure.columns if 'Margin of Error' in str(c)])
tenure = tenure.drop(columns=[c for c in tenure.columns if str(c).startswith('Unnamed')])

# Midpoints for renter-occupied occupants-per-room bins
opr_midpoints = {
    'Estimate!!Total:!!Renter occupied:!!0.50 or less occupants per room': 0.25,
    'Estimate!!Total:!!Renter occupied:!!0.51 to 1.00 occupants per room': 0.755,
    'Estimate!!Total:!!Renter occupied:!!1.01 to 1.50 occupants per room': 1.255,
    'Estimate!!Total:!!Renter occupied:!!1.51 to 2.00 occupants per room': 1.755,
    'Estimate!!Total:!!Renter occupied:!!2.01 or more occupants per room': 2.5,
}

for col in opr_midpoints:
    tenure[col] = pd.to_numeric(tenure[col], errors='coerce')

opr_cols = list(opr_midpoints.keys())
tenure['avg_occupants_per_room'] = (
    sum(tenure[col] * mp for col, mp in opr_midpoints.items())
    / tenure[opr_cols].sum(axis=1)
)
tenure = tenure[['GEO_ID', 'avg_occupants_per_room']].copy()

# =============================================================================
# 5. Clean Health Insurance Coverage.csv
# =============================================================================
health = pd.read_csv('data/raw data/Health Insurance Coverage.csv', header=1)
health = health.rename(columns={'Geography': 'GEO_ID', 'Geographic Area Name': 'NAME'})
health = health.drop(columns=[c for c in health.columns if 'Margin of Error' in str(c)])

total_pop_col = 'Estimate!!Total:'
insured_cols = [c for c in health.columns if '!!With health insurance coverage' in str(c)]

health[total_pop_col] = pd.to_numeric(health[total_pop_col], errors='coerce')
for col in insured_cols:
    health[col] = pd.to_numeric(health[col], errors='coerce')

health['insurance_coverage_rate'] = (
    health[insured_cols].sum(axis=1) / health[total_pop_col]
)
health = health[['GEO_ID', 'insurance_coverage_rate']].copy()

# =============================================================================
# 6. Merge all datasets (left joins on GEO_ID, starting from income)
# =============================================================================
metro_data = (
    income
    .merge(rent,    on='GEO_ID', how='left')
    .merge(commute, on='GEO_ID', how='left')
    .merge(tenure,  on='GEO_ID', how='left')
    .merge(health,  on='GEO_ID', how='left')
)

metro_data = metro_data[[
    'GEO_ID', 'NAME',
    'bachelor_degree', 'graduate_degree',
    'median_rent',
    'avg_commute_time',
    'avg_occupants_per_room',
    'insurance_coverage_rate'
]]

# =============================================================================
# 7. Output
# =============================================================================
print(metro_data.head())
