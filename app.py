import os
import pathlib

_HERE = pathlib.Path(__file__).parent
_DATA = _HERE / "data"

_proj_path = "/opt/miniconda3/envs/dap311/share/proj"
if os.path.exists(_proj_path):
    os.environ["PROJ_DATA"] = _proj_path
    os.environ["PROJ_LIB"]  = _proj_path

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

# Compatibility shim: st.cache_data was added in Streamlit 1.18; fall back to st.cache for older installs
if not hasattr(st, "cache_data"):
    st.cache_data = lambda f=None, **kw: (st.cache(allow_output_mutation=True)(f) if f else st.cache(allow_output_mutation=True))

import pandas as pd
import numpy as np
import geopandas as gpd
import pydeck as pdk
import plotly.graph_objects as go
import matplotlib.cm as mcm
from matplotlib.colors import Normalize
import plotly.express as px

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="An Attractive Index",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
STATE_TO_REGION = {
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "RI": "Northeast", "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast",
    "PA": "Northeast",
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest",
    "WI": "Midwest", "IA": "Midwest", "KS": "Midwest", "MN": "Midwest",
    "MO": "Midwest", "NE": "Midwest", "ND": "Midwest", "SD": "Midwest",
    "DE": "South", "FL": "South", "GA": "South", "MD": "South", "NC": "South",
    "SC": "South", "VA": "South", "WV": "South", "DC": "South", "AL": "South",
    "KY": "South", "MS": "South", "TN": "South", "AR": "South", "LA": "South",
    "OK": "South", "TX": "South",
    "AZ": "West", "CO": "West", "ID": "West", "MT": "West", "NV": "West",
    "NM": "West", "UT": "West", "WY": "West", "AK": "West", "CA": "West",
    "HI": "West", "OR": "West", "WA": "West",
    "PR": "Puerto Rico / Other",
}

COLOR_METRICS = {
    "Final Attractiveness Score": "final_score",
    "Median Income (Bachelor's)": "bachelor_degree",
    "Median Rent":                "median_rent",
    "Air Quality (AQI)":         "Median_AQI",
}

# ── Helper functions ───────────────────────────────────────────────────────────
def value_to_rgba(series, cmap_name="YlGnBu", alpha=200, invert=False):
    """Convert a numeric Series to a list of [R, G, B, A] for pydeck."""
    filled = series.fillna(series.median())
    norm   = Normalize(vmin=filled.min(), vmax=filled.max())
    cmap   = mcm.get_cmap(cmap_name)
    result = []
    for v in series:
        if pd.isna(v):
            result.append([160, 160, 160, 100])
        else:
            t = float(norm(v))
            if invert:
                t = 1.0 - t
            r, g, b, _ = cmap(t)
            result.append([int(r * 255), int(g * 255), int(b * 255), alpha])
    return result


def clean_name(name):
    if pd.isna(name):
        return ""
    for suffix in [" Metro Area", " Metropolitan Statistical Area",
                   " Micropolitan Statistical Area"]:
        name = name.replace(suffix, "")
    return name.strip()


def z_score(series):
    mu, sigma = series.mean(), series.std()
    if sigma == 0:
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sigma



# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_raw():
    metro = pd.read_csv(_DATA / "derived data" / "metro_data.csv")
    poll  = pd.read_csv(_DATA / "derived data" / "pollution.csv")
    rpp   = pd.read_csv(_DATA / "derived data" / "BEA_Metro_RPP_2024_Clean.csv")
    emp   = pd.read_csv(_DATA / "derived data" / "Cleaned_MSA_Employment_Growth.csv")

    metro["CBSA"] = metro["GEO_ID"].str[-5:]

    rpp["CBSA"] = rpp["GeoFIPS"].str.replace('"', "", regex=False).str.strip()
    rpp_all = (
        rpp[rpp["Description"].str.strip() == "RPPs: All items"][["CBSA", "2024"]]
        .rename(columns={"2024": "regional_price_parity"})
    )

    emp["CBSA"] = (
        emp["area_fips"].str[1:].astype(int) * 10
    ).astype(str).str.zfill(5)
    emp_clean = (
        emp[["CBSA", "oty_month3_emplvl_pct_chg"]]
        .rename(columns={"oty_month3_emplvl_pct_chg": "employment_growth"})
    )

    df = metro.merge(poll[["GEO_ID", "Median_AQI"]], on="GEO_ID", how="left")
    df = df.merge(rpp_all,   on="CBSA", how="left")
    df = df.merge(emp_clean, on="CBSA", how="left")

    df["state"]        = df["NAME"].apply(
        lambda n: n.split(",")[-1].strip().split()[0] if "," in str(n) else "Unknown"
    )
    df["region"]       = df["state"].map(STATE_TO_REGION).fillna("Other")
    df["display_name"] = df["NAME"].apply(clean_name)
    return df


@st.cache_data
def load_centroids():
    shp = gpd.read_file(_DATA / "map" / "tl_2023_us_cbsa" / "tl_2023_us_cbsa.shp")
    shp = shp.to_crs("EPSG:4326")
    shp = shp.cx[-130:-60, 23:52]
    shp["lon"] = shp.geometry.centroid.x
    shp["lat"] = shp.geometry.centroid.y
    # Return plain DataFrame — no geometry column — so pydeck can serialize it
    return pd.DataFrame({"CBSAFP": shp["CBSAFP"].values,
                         "lon":    shp["lon"].values,
                         "lat":    shp["lat"].values})


def compute_scores(df, w_opp, w_exp, w_life):
    df = df.copy()

    df["income_z"]             =  z_score(df["bachelor_degree"])
    df["employment_growth_z"]  =  z_score(df["employment_growth"])
    df["rent_z"]               = -z_score(df["median_rent"])
    df["price_parity_z"]       = -z_score(df["regional_price_parity"])
    df["AQI_z"]                = -z_score(df["Median_AQI"])
    df["commute_time_z"]       = -z_score(df["avg_commute_time"])
    df["insurance_z"]          =  z_score(df["insurance_coverage_rate"])
    df["occupants_per_room_z"] = -z_score(df["avg_occupants_per_room"])

    df["economic_opportunity_score"] = (
        df["income_z"] + df["employment_growth_z"]
    ) / 2
    df["living_expense_score"] = (
        df["rent_z"] + df["price_parity_z"]
    ) / 2
    df["living_conditions_score"] = (
        df["AQI_z"] + df["commute_time_z"] +
        df["insurance_z"] + df["occupants_per_room_z"]
    ) / 4

    total = w_opp + w_exp + w_life
    if total == 0:
        w_opp = w_exp = w_life = 1.0 / 3
        total = 1.0

    df["final_score"] = (
        w_opp  * df["economic_opportunity_score"] +
        w_exp  * df["living_expense_score"] +
        w_life * df["living_conditions_score"]
    ) / total

    df["rank"] = (
        df["final_score"]
        .rank(ascending=False, method="min")
        .fillna(999)
        .astype(int)
    )
    return df


# ── App ────────────────────────────────────────────────────────────────────────
def main():
    raw_df    = load_raw()
    centroids = load_centroids()

    # ── Sidebar Phase 1: weights / color / filters ─────────────────────────────
    with st.sidebar:
        st.markdown("## Dashboard Controls")

        st.markdown("### Personal Preferences")
        st.caption(
            "Distribute your priorities across three dimensions. "
            "Sliders are automatically normalized to 100%."
        )
        w_opp  = st.slider("Economic Opportunity (%)", 0, 100, 33)
        w_exp  = st.slider("Affordability (%)",        0, 100, 33)
        w_life = st.slider("Quality of Life (%)",      0, 100, 34)

        total_raw = w_opp + w_exp + w_life
        if total_raw == 0:
            w_opp_n = w_exp_n = w_life_n = 1.0 / 3
        else:
            w_opp_n  = w_opp  / total_raw
            w_exp_n  = w_exp  / total_raw
            w_life_n = w_life / total_raw

        st.caption(
            f"Normalized — Opportunity: **{w_opp_n:.0%}** | "
            f"Affordability: **{w_exp_n:.0%}** | "
            f"Quality of Life: **{w_life_n:.0%}**"
        )

        st.markdown("---")
        st.markdown("### Map Color Metric")
        color_label = st.selectbox("Select Color Metric", list(COLOR_METRICS.keys()))
        color_col   = COLOR_METRICS[color_label]

        st.markdown("---")
        st.markdown("### Leaderboard")
        top_n = st.slider("Number of cities shown", 5, 30, 20)

    # Compute scores (needs weights from sidebar Phase 1)
    df = compute_scores(raw_df, w_opp_n, w_exp_n, w_life_n)

    # ── Sidebar Phase 2: city explorer + radar chart ───────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("### City Explorer")

        all_regions = ["All"] + sorted(df["region"].dropna().unique().tolist())
        sel_region  = st.selectbox("Filter by Region", all_regions)

        city_pool = (
            df["display_name"].dropna().sort_values().tolist()
            if sel_region == "All"
            else df[df["region"] == sel_region]["display_name"]
                   .dropna().sort_values().tolist()
        )

        # Session state keeps map-click in sync with selectbox
        if "selected_city" not in st.session_state:
            st.session_state["selected_city"] = city_pool[0] if city_pool else ""
        if st.session_state["selected_city"] not in city_pool:
            st.session_state["selected_city"] = city_pool[0] if city_pool else ""

        default_idx = (
            city_pool.index(st.session_state["selected_city"])
            if st.session_state["selected_city"] in city_pool else 0
        )
        sel_city = st.selectbox("Select City", city_pool, index=default_idx)
        st.session_state["selected_city"] = sel_city

        # City row lookup (used later in main panel)
        city_row = df[df["display_name"] == sel_city]

    # ── Main panel ─────────────────────────────────────────────────────────────
    st.title("An Attractive Index: Evaluating Urban Attractiveness for Graduates")
    st.markdown(
        "This dashboard visualizes a **City Attractiveness Index** that compares "
        "U.S. metropolitan areas based on opportunity, affordability, and quality "
        "of life. Use the sidebar sliders to personalize the index weights, select "
        "a region and city to explore its profile, and filter charts to discover "
        "the best metro area for you."
    )

    # ── Pre-compute top-N set for map highlighting ─────────────────────────────
    top_cbsa_set = set(
        df.dropna(subset=["final_score"])
        .nlargest(top_n, "final_score")["CBSA"]
        .tolist()
    )

    # ── Chart 1: Pydeck bubble map ─────────────────────────────────────────────
    st.subheader("City Attractiveness Map")
    st.caption(
        f"**Bubble size** = Final Attractiveness Score (larger = better fit for "
        f"your preferences) | **Color** = {color_label} | "
        f"**Red rings** = Top {top_n} leaderboard cities | "
        f"Hover over a bubble for details."
    )

    map_df = df.merge(centroids, left_on="CBSA", right_on="CBSAFP", how="inner")
    map_df = map_df.dropna(subset=["lon", "lat", "final_score"])

    # Radius: scale final_score linearly to [12 km, 77 km]
    s_min, s_max = map_df["final_score"].min(), map_df["final_score"].max()
    denom = s_max - s_min if s_max > s_min else 1.0
    map_df["radius"] = 12000 + (map_df["final_score"] - s_min) / denom * 65000

    # Color
    c_col    = color_col if color_col in map_df.columns else "final_score"
    invert_c = c_col in ("median_rent", "Median_AQI")
    map_df["fill_color"] = value_to_rgba(map_df[c_col], "YlGnBu", 210, invert=invert_c)

    # Pre-format tooltip values
    map_df["score_str"] = map_df["final_score"].round(3).astype(str)
    map_df["rank_str"]  = map_df["rank"].astype(str)

    # Build plain dict list for pydeck
    layer_records = [
        {
            "lon":          float(row["lon"]),
            "lat":          float(row["lat"]),
            "radius":       float(row["radius"]),
            "fill_color":   [int(c) for c in row["fill_color"]],
            "display_name": str(row["display_name"]),
            "score_str":    str(row["score_str"]),
            "rank_str":     str(row["rank_str"]),
        }
        for _, row in map_df.iterrows()
    ]

    # Red ring records for top-N leaderboard cities
    top_records = [
        {
            "lon":          float(row["lon"]),
            "lat":          float(row["lat"]),
            "radius":       float(row["radius"]) + 6000,
            "display_name": str(row["display_name"]),
            "score_str":    str(row["score_str"]),
            "rank_str":     str(row["rank_str"]),
        }
        for _, row in map_df.iterrows()
        if str(row["CBSA"]) in top_cbsa_set
    ]

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=layer_records,
        get_position="[lon, lat]",
        get_radius="radius",
        get_fill_color="fill_color",
        get_line_color=[20, 20, 20, 60],
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 210, 0, 230],
    )

    top_layer = pdk.Layer(
        "ScatterplotLayer",
        data=top_records,
        get_position="[lon, lat]",
        get_radius="radius",
        get_fill_color=[0, 0, 0, 0],
        get_line_color=[220, 30, 30, 255],
        line_width_min_pixels=3,
        stroked=True,
        filled=False,
        pickable=True,
        auto_highlight=False,
    )

    view_state = pdk.ViewState(
        latitude=38.5, longitude=-96.0, zoom=3.8, pitch=0, min_zoom=3.2
    )

    tooltip = {
        "html": (
            "<span style='font-size:13px;font-weight:bold'>{display_name}</span><br/>"
            "Rank: <b>#{rank_str}</b> &nbsp;|&nbsp; Score: <b>{score_str}</b>"
        ),
        "style": {
            "background":    "rgba(20,30,50,0.88)",
            "color":         "white",
            "padding":       "8px 12px",
            "borderRadius":  "6px",
            "fontSize":      "12px",
        },
    }

    deck = pdk.Deck(
        layers=[scatter_layer, top_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="light",
    )

    st.pydeck_chart(deck, use_container_width=True)

    # Color-scale legend
    low_label, high_label = ("Lower", "Higher")
    if invert_c:
        low_label, high_label = "Higher (worse)", "Lower (better)"
    st.caption(
        f"Color scale: {low_label} {color_label} "
        f"(light yellow) → {high_label} (dark blue)"
    )

    st.markdown("---")

    # ── Chart 4: Leaderboard (below map, full width) ───────────────────────────
    st.subheader(f"Attractiveness Leaderboard — Top {top_n} Metro Areas")
    st.caption("Ranked by Final Attractiveness Score based on your current weight preferences.")

    top_df = (
        df.dropna(subset=["final_score"])
        .nlargest(top_n, "final_score")
        .sort_values("final_score", ascending=True)
        .copy()
    )
    top_df["name_short"] = top_df["display_name"].apply(
        lambda x: x[:30] + "…" if len(x) > 30 else x
    )
    top_df["rank_label"] = top_df["rank"].astype(str).apply(lambda r: f"#{r}")

    fig_b = go.Figure(go.Bar(
        y=top_df["name_short"],
        x=top_df["final_score"],
        orientation="h",
        marker_color="#636EFA",
        text=top_df["rank_label"],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Final Score: %{x:.3f}<extra></extra>",
    ))
    fig_b.update_layout(
        height=max(350, top_n * 22),
        margin=dict(l=10, r=60, t=10, b=50),
        xaxis=dict(title="Final Attractiveness Score", title_standoff=10),
        yaxis=dict(title="", tickfont=dict(size=9), automargin=True),
        showlegend=False,
    )
    st.plotly_chart(fig_b, use_container_width=True)

    # ── Detailed metrics table (ranked by attractiveness) ──────────────────────
    st.subheader(f"City Metrics Table — Top {top_n} Metro Areas")
    st.caption("Same cities as the leaderboard above, ranked by Final Attractiveness Score. Raw indicator values shown.")

    # Build display table from top_df (already ranked)
    table_df = top_df.sort_values("final_score", ascending=False)[[
        "rank", "display_name",
        "bachelor_degree", "employment_growth",
        "median_rent", "regional_price_parity",
        "Median_AQI", "avg_commute_time",
        "insurance_coverage_rate", "avg_occupants_per_room",
    ]].copy()

    table_df.columns = [
        "Rank", "Metro Area",
        "Median Income ($)", "Emp. Growth (%)",
        "Median Rent ($)", "Price Parity",
        "Median AQI", "Avg Commute (min)",
        "Insurance Coverage", "Occupants/Room",
    ]

    table_df["Rank"]               = table_df["Rank"].apply(lambda x: f"#{int(x)}")
    table_df["Median Income ($)"]  = table_df["Median Income ($)"].apply(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
    table_df["Emp. Growth (%)"]    = table_df["Emp. Growth (%)"].apply(
        lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
    table_df["Median Rent ($)"]    = table_df["Median Rent ($)"].apply(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
    table_df["Price Parity"]       = table_df["Price Parity"].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    table_df["Median AQI"]         = table_df["Median AQI"].apply(
        lambda x: f"{x:.0f}" if pd.notna(x) else "—")
    table_df["Avg Commute (min)"]  = table_df["Avg Commute (min)"].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    table_df["Insurance Coverage"] = table_df["Insurance Coverage"].apply(
        lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    table_df["Occupants/Room"]     = table_df["Occupants/Room"].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) else "—")

    html = table_df.reset_index(drop=True).to_html(index=False, escape=False)
    scrollable_html = f"""
    <div style="overflow-x: auto; overflow-y: auto; max-height: 420px; border: 1px solid #e0e0e0; border-radius: 6px;">
        <style>
            .metrics-table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
            .metrics-table th {{
                background-color: #f0f2f6; position: sticky; top: 0; z-index: 1;
                padding: 8px 12px; text-align: center; border-bottom: 2px solid #ccc;
                white-space: nowrap;
            }}
            .metrics-table td {{
                padding: 7px 12px; text-align: center; border-bottom: 1px solid #eee;
                white-space: nowrap;
            }}
            .metrics-table tr:hover td {{ background-color: #f7f9fc; }}
        </style>
        {html.replace('<table border="1" class="dataframe">', '<table class="metrics-table">')}
    </div>
    """
    st.write(scrollable_html, unsafe_allow_html=True)

    # ── City Profile + Radar Chart (bottom center) ─────────────────────────────
    st.markdown("---")
    st.subheader(f"City Profile — {sel_city}")

    if not city_row.empty:
        cd = city_row.iloc[0]

        prof_left, prof_right = st.columns([1, 1])

        with prof_left:
            st.markdown(f"##### {sel_city}")
            st.metric("Final Rank",          f"#{int(cd['rank'])}")
            st.metric("Final Score",         f"{cd['final_score']:.3f}")
            st.metric("Opportunity Score",   f"{cd['economic_opportunity_score']:.3f}")
            st.metric("Affordability Score", f"{cd['living_expense_score']:.3f}")
            st.metric("Quality of Life",     f"{cd['living_conditions_score']:.3f}")

        with prof_right:
            dim_labels = ["Economic\nOpportunity", "Affordability", "Quality\nof Life"]
            dim_vals   = [
                cd["economic_opportunity_score"],
                cd["living_expense_score"],
                cd["living_conditions_score"],
            ]
            score_cols = [
                "economic_opportunity_score",
                "living_expense_score",
                "living_conditions_score",
            ]
            r_min = df[score_cols].min().min() - 0.3
            r_max = df[score_cols].max().max() + 0.3

            fig_radar = go.Figure(go.Scatterpolar(
                r     = dim_vals + [dim_vals[0]],
                theta = dim_labels + [dim_labels[0]],
                fill  = "toself",
                fillcolor = "rgba(99,110,250,0.25)",
                line  = dict(color="rgb(99,110,250)", width=2),
                name  = sel_city,
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[r_min, r_max],
                        tickfont=dict(size=8),
                    )
                ),
                showlegend=False,
                margin=dict(l=40, r=40, t=50, b=20),
                height=320,
                title=dict(
                    text="Z-Score Dimensions",
                    font=dict(size=13),
                    x=0.5,
                ),
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("Select a city in the sidebar to view its profile.")


if __name__ == "__main__":
    main()
