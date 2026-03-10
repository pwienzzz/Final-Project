# An Attractive Index: Evaluating Urban Attractiveness for Graduates

**Group Members:** Peiwen Zhang (GitHub: `pwienzzz`) | Han Zhang (GitHub: `hanz12138`)

## Live Dashboard

[Launch Streamlit App](https://hanz12138-final-project-app-2yhvvy.streamlit.app)

> **Note:** Streamlit Community Cloud apps are automatically put to sleep after 24 hours of inactivity. If the app shows a "This app is sleeping" screen, click **"Wake up"** and wait a few seconds for it to restart. This is normal Streamlit behavior, not a bug.

The interactive dashboard lets you distribute weights across three dimensions (Economic Opportunity, Affordability, Quality of Life) using sidebar sliders. All charts — a Pydeck bubble map, Plotly leaderboard, metrics table, and radar chart — update in real time to reflect your preferences.

## Repository Structure

```
Final-Project/
├── Final_Project.qmd        # Main writeup and analysis (renders to PDF)
├── writeup.qmd              # Shorter standalone writeup
├── requirements.txt         # Python dependencies for QMD rendering
├── streamlit-app/
│   ├── app.py               # Streamlit dashboard
│   ├── requirements.txt     # Dependencies for Streamlit Cloud
│   ├── packages.txt         # System packages (GDAL) for Streamlit Cloud
│   └── data/                # Symlinked/copied data for the app
└── data/
    ├── derived data/        # Cleaned and merged CSVs + output PNGs
    └── map/                 # Census TIGER/Line CBSA shapefiles
```

## Data Sources

| Dataset | Source | File |
|---|---|---|
| Census ACS (income, rent, commute, insurance, overcrowding) | U.S. Census Bureau American Community Survey | `metro_data.csv` |
| Air Quality Index | U.S. EPA AQS Annual Summary | `pollution.csv` |
| Regional Price Parities | U.S. Bureau of Economic Analysis | `BEA_Metro_RPP_2024_Clean.csv` |
| Employment Growth | BLS Quarterly Census of Employment and Wages | `Cleaned_MSA_Employment_Growth.csv` |
| CBSA Boundaries | Census TIGER/Line Shapefiles (2023) | `tl_2023_us_cbsa/` |

## Data Processing

All cleaning is done in Python (`pandas`). The main challenge was harmonizing CBSA identifiers across sources:

- **BLS**: `area_fips` requires `str[1:].astype(int) * 10` to produce a 5-digit CBSA code
- **BEA**: `GeoFIPS` contains embedded quotes that must be stripped
- **Census ACS**: CBSA code is the last 5 characters of `GEO_ID`

After merging on CBSA codes (~900 metro areas), each indicator is z-scored. Indicators where higher raw values are worse (rent, AQI, commute time, price parity, occupants per room) are negated so that higher z-scores always mean better outcomes. Three sub-scores are computed:

- **Economic Opportunity** = mean(income z-score, employment growth z-score)
- **Affordability** = mean(rent z-score, price parity z-score)
- **Quality of Life** = mean(AQI z-score, commute z-score, insurance z-score, overcrowding z-score)

A **Final Attractiveness Score** is the weighted average of these three sub-scores using user-defined weights.

## Rendering the Report

Render `Final_Project.qmd` to PDF using Quarto with the conda environment that has all dependencies installed:

```bash
quarto render Final_Project.qmd --to pdf
```
