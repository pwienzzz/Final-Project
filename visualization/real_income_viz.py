"""
Real Income Visualization for U.S. Metropolitan Areas
======================================================

This script calculates **real income** — nominal median income for bachelor's
degree holders adjusted for local purchasing power using BEA Regional Price
Parities (RPP) — and produces two visualizations:

  1. A horizontal bar chart ranking the Top 30 MSAs by real income (Altair)
  2. A spatial choropleth map of real income across all U.S. MSAs (GeoPandas)

Formula:
    Real Income = Nominal Median Income × (100 / RPP)

Intuition: An RPP of 110 means prices are 10% above the national average, so a
dollar goes 10% less far. Dividing by RPP/100 rescales income to a common
national price level, making incomes directly comparable across cities.

Data Sources:
  - ACS 5-Year Estimates 2024, Table B20004 (Median Income by Education)
  - BEA Regional Price Parities by MSA, 2024
  - U.S. Census TIGER/Line CBSA Shapefile, 2024
"""

import os
import urllib.request
import zipfile

import pandas as pd
import altair as alt
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolve paths relative to this script's location (project root assumed one
# level up from /visualization/).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, "data", "derived data")
SHP_DIR    = os.path.join(ROOT_DIR, "data", "shapefiles")   # gitignored large files
OUT_DIR    = os.path.join(ROOT_DIR, "output")

os.makedirs(SHP_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

METRO_CSV = os.path.join(DATA_DIR, "metro_data.csv")
RPP_CSV   = os.path.join(DATA_DIR, "BEA_Metro_RPP_2024_Clean.csv")

CBSA_ZIP  = os.path.join(SHP_DIR, "tl_2024_us_cbsa.zip")
CBSA_URL  = "https://www2.census.gov/geo/tiger/TIGER2024/CBSA/tl_2024_us_cbsa.zip"
CBSA_SHP  = os.path.join(SHP_DIR, "cbsa", "tl_2024_us_cbsa.shp")

STATE_ZIP = os.path.join(SHP_DIR, "tl_2024_us_state.zip")
STATE_URL = "https://www2.census.gov/geo/tiger/TIGER2024/STATE/tl_2024_us_state.zip"
STATE_SHP = os.path.join(SHP_DIR, "state", "tl_2024_us_state.shp")

# ── Helper: download + unzip ───────────────────────────────────────────────────
def download_and_unzip(url, zip_path, extract_dir):
    if not os.path.exists(zip_path):
        print(f"  Downloading {os.path.basename(zip_path)} …")
        urllib.request.urlretrieve(url, zip_path)
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

# =============================================================================
# 1. LOAD & MERGE DATA
# =============================================================================

# --- ACS metro data -----------------------------------------------------------
# GEO_ID format: "310M700US10180"  →  CBSA code = "10180" (digits after "US")
metro = pd.read_csv(METRO_CSV)
metro["cbsa_code"] = (
    metro["GEO_ID"]
    .str.extract(r"US(\d+)$")[0]
    .str.zfill(5)
)
# Shorten metro name for readable chart labels
metro["metro_name"] = (
    metro["NAME"]
    .str.replace(r"\s*(Metro Area|Metropolitan Statistical Area)$", "", regex=True)
    .str.strip()
)

# --- BEA Regional Price Parities ---------------------------------------------
# GeoFIPS format: ' ""10180""'  →  CBSA code = "10180"
# LineCode 1.0 = "RPPs: All items" (composite price index we want)
rpp = pd.read_csv(RPP_CSV)
rpp = rpp[rpp["LineCode"] == "1.0"].copy()
rpp["cbsa_code"] = (
    rpp["GeoFIPS"]
    .str.strip()
    .str.replace('"', "", regex=False)
    .str.strip()
    .str.zfill(5)
)
rpp = rpp.rename(columns={"2024": "rpp"})[["cbsa_code", "rpp"]]

# --- Merge -------------------------------------------------------------------
df = metro.merge(rpp, on="cbsa_code", how="inner")

print(f"\nMSAs matched after merge: {len(df)}")
print(f"MSAs dropped (no RPP match): {len(metro) - len(df)}")

# =============================================================================
# 2. CALCULATE REAL INCOME
# =============================================================================
#
# RPP = 100 means prices exactly match the national average.
# RPP > 100  →  more expensive city  →  lower real income than nominal.
# RPP < 100  →  cheaper city         →  higher real income than nominal.
#
# Real Income = Nominal Income × (100 / RPP)
#
df["real_income"] = df["bachelor_degree"] * (100.0 / df["rpp"])
df = df.dropna(subset=["real_income"])
df = df.sort_values("real_income", ascending=False).reset_index(drop=True)
df["rank"] = df.index + 1

print("\nTop 10 MSAs by Real Income:")
print(
    df[["rank", "metro_name", "bachelor_degree", "rpp", "real_income"]]
    .head(10)
    .to_string(index=False)
)

# =============================================================================
# 3. BAR CHART — Top 30 MSAs by Real Income  (Altair)
# =============================================================================
#
# We display the top 30 metropolitan areas sorted by real income.
# Bars are colored by affordability: blue = cheaper than average (RPP < 100),
# orange = more expensive (RPP ≥ 100), showing which high-income cities are
# truly affordable vs. merely nominally well-paid.
#
top30 = df.head(30).copy()
top30["affordable"] = top30["rpp"].apply(
    lambda x: "Below national avg (RPP < 100)" if x < 100 else "Above national avg (RPP ≥ 100)"
)

color_scale = alt.Scale(
    domain=["Below national avg (RPP < 100)", "Above national avg (RPP ≥ 100)"],
    range=["#2E86AB", "#E07B39"]
)

bar = (
    alt.Chart(top30)
    .mark_bar()
    .encode(
        y=alt.Y(
            "metro_name:N",
            sort=alt.EncodingSortField(field="real_income", order="descending"),
            title=None,
            axis=alt.Axis(labelFontSize=10, labelLimit=300),
        ),
        x=alt.X(
            "real_income:Q",
            title="Real Income (USD, adjusted to national price level)",
            axis=alt.Axis(format="$,.0f", labelFontSize=10),
        ),
        color=alt.Color(
            "affordable:N",
            scale=color_scale,
            title="Local Price Level",
            legend=alt.Legend(orient="bottom-right"),
        ),
        tooltip=[
            alt.Tooltip("rank:Q", title="Rank"),
            alt.Tooltip("metro_name:N", title="Metro Area"),
            alt.Tooltip("bachelor_degree:Q", title="Nominal Income ($)", format=",.0f"),
            alt.Tooltip("rpp:Q", title="RPP (100 = national avg)", format=".1f"),
            alt.Tooltip("real_income:Q", title="Real Income ($)", format=",.0f"),
        ],
    )
)

# Reference line at median real income of top 30
median_val = top30["real_income"].median()
ref_line = (
    alt.Chart(pd.DataFrame({"x": [median_val]}))
    .mark_rule(color="gray", strokeDash=[4, 4], size=1.5)
    .encode(x="x:Q")
)
ref_label = (
    alt.Chart(pd.DataFrame({"x": [median_val], "y": [""], "label": [f"Median: ${median_val:,.0f}"]}))
    .mark_text(align="left", dx=6, dy=-6, color="gray", fontSize=9)
    .encode(x="x:Q", text="label:N")
)

chart = (
    (bar + ref_line + ref_label)
    .properties(
        width=620,
        height=620,
        title=alt.TitleParams(
            text="Top 30 U.S. Metro Areas by Real Income for College Graduates (2024)",
            subtitle=[
                "Real Income = Nominal Median Income × (100 ÷ RPP)  |  "
                "Color indicates whether the city is cheaper or more expensive than the national average.",
                "Source: ACS 5-Year Estimates 2024 (B20004); BEA Regional Price Parities 2024",
            ],
            fontSize=14,
            subtitleFontSize=10,
            subtitleColor="gray",
            anchor="start",
        ),
    )
    .configure_view(stroke=None)
    .configure_axis(grid=False)
)

bar_path = os.path.join(OUT_DIR, "top30_real_income_bar.html")
chart.save(bar_path)
print(f"\nSaved bar chart → {bar_path}")

# =============================================================================
# 4. SPATIAL MAP — Real Income Choropleth  (GeoPandas + Matplotlib)
# =============================================================================
#
# We overlay MSA polygons on a U.S. state background, colored by real income.
# Alaska and Hawaii are excluded to keep the contiguous-48 view readable.
# Gray polygons indicate MSAs present in the shapefile but not in our dataset
# (typically micropolitan areas or territories).
#

# --- Download shapefiles if needed -------------------------------------------
print("\nChecking shapefiles …")
download_and_unzip(CBSA_URL, CBSA_ZIP, os.path.join(SHP_DIR, "cbsa"))
download_and_unzip(STATE_URL, STATE_ZIP, os.path.join(SHP_DIR, "state"))

# --- Load shapefiles ---------------------------------------------------------
cbsa_gdf   = gpd.read_file(CBSA_SHP).rename(columns={"GEOID": "cbsa_code"})
states_gdf = gpd.read_file(STATE_SHP)

# Drop non-contiguous territories (Alaska=02, Hawaii=15, PR=72, territories)
exclude_fips = {"02", "15", "60", "66", "69", "72", "78"}
states_gdf = states_gdf[~states_gdf["STATEFP"].isin(exclude_fips)]

# Filter MSA polygons to those within the contiguous U.S. bounding box
# (removes Puerto Rico and far-flung areas that stretch the map)
cbsa_gdf = cbsa_gdf.cx[-130:-60, 22:52]   # lon/lat bounding box (WGS84 first)

# --- Merge real income data --------------------------------------------------
geo_df = cbsa_gdf.merge(
    df[["cbsa_code", "real_income", "metro_name", "rpp", "bachelor_degree"]],
    on="cbsa_code",
    how="left",
)

# --- Reproject to Albers Equal Area (standard for U.S. choropleth maps) ------
crs = "EPSG:5070"
geo_df    = geo_df.to_crs(crs)
states_gdf = states_gdf.to_crs(crs)

# --- Plot --------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(18, 10))

# State background
states_gdf.plot(ax=ax, color="#f7f7f7", edgecolor="#aaaaaa", linewidth=0.6, zorder=1)

# MSAs with data
geo_with = geo_df[geo_df["real_income"].notna()]
geo_with.plot(
    ax=ax,
    column="real_income",
    cmap="YlOrRd",
    legend=False,
    edgecolor="white",
    linewidth=0.15,
    vmin=geo_with["real_income"].quantile(0.05),  # clip extreme low outliers
    vmax=geo_with["real_income"].quantile(0.95),  # clip extreme high outliers
    zorder=2,
)

# MSAs without data (gray)
geo_without = geo_df[geo_df["real_income"].isna()]
geo_without.plot(ax=ax, color="#cccccc", edgecolor="white", linewidth=0.15, zorder=2)

# Colorbar
sm = plt.cm.ScalarMappable(
    cmap="YlOrRd",
    norm=plt.Normalize(
        vmin=geo_with["real_income"].quantile(0.05),
        vmax=geo_with["real_income"].quantile(0.95),
    ),
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.02,
                    shrink=0.45, aspect=30, fraction=0.03)
cbar.set_label("Real Income (USD, adjusted to national price level)", fontsize=11)
cbar.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
cbar.ax.tick_params(labelsize=9)

# Annotate top 5 cities
top5 = df.head(5)
top5_geo = geo_df[geo_df["cbsa_code"].isin(top5["cbsa_code"])]
for _, row in top5_geo.iterrows():
    centroid = row.geometry.centroid
    short_name = row["metro_name"].split(",")[0]
    ax.annotate(
        short_name,
        xy=(centroid.x, centroid.y),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=7.5,
        color="#333333",
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="#666666", lw=0.5),
    )

ax.set_title(
    "Real Income for College Graduates Across U.S. Metropolitan Areas (2024)\n"
    "Adjusted for Regional Price Parities — higher values indicate greater purchasing power",
    fontsize=14,
    fontweight="bold",
    pad=14,
)
ax.text(
    0.01, -0.03,
    "Source: ACS 5-Year Estimates 2024 (B20004); BEA Regional Price Parities 2024; "
    "U.S. Census TIGER/Line CBSA Shapefile 2024",
    transform=ax.transAxes,
    fontsize=8,
    color="gray",
    va="top",
)
ax.set_axis_off()
plt.tight_layout()

map_path = os.path.join(OUT_DIR, "real_income_map.png")
fig.savefig(map_path, dpi=150, bbox_inches="tight")
print(f"Saved map       → {map_path}")
plt.show()

# =============================================================================
# 5. EXPORT MERGED DATA (optional — useful for Quarto writeup)
# =============================================================================
export_path = os.path.join(DATA_DIR, "metro_real_income.csv")
df.to_csv(export_path, index=False)
print(f"Saved merged data → {export_path}")
print("\nDone.")
