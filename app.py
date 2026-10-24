import time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="PCA Analysis of Urban Heat", page_icon="🌳", layout="wide")

st.title("PCA Analysis of Urban Heat")
st.caption("Impact of Green Cover and Urbanization | SOFTWARE SIMULATION — NOT LIVE SENSOR DATA")

st.warning(
    "SIMULATED / ILLUSTRATIVE DATA: This app generates synthetic readings for demonstration. "
    "It does not measure actual temperatures, use satellite imagery, or connect to hardware."
)

# These distributions are illustrative assumptions for a classroom demonstration, not measured evidence.
FEATURES = [
    "air_temp_c", "humidity_pct", "surface_temp_c",
    "green_cover_pct", "built_up_pct", "ndvi_proxy", "albedo_proxy"
]

def make_simulated_data(n=120, seed=7):
    rng = np.random.default_rng(seed)
    zones = rng.choice(["Urban", "Green"], size=n, p=[0.5, 0.5])
    rows = []
    start = pd.Timestamp.now().floor("min") - pd.Timedelta(minutes=n)
    for i, zone in enumerate(zones):
        if zone == "Urban":
            green = rng.normal(15, 5)
            built = rng.normal(78, 7)
            air = rng.normal(34.5, 2.0)
            surface = rng.normal(41, 3.0)
            humidity = rng.normal(48, 7)
            ndvi = rng.normal(0.20, 0.07)
            albedo = rng.normal(0.17, 0.025)
        else:
            green = rng.normal(55, 10)
            built = rng.normal(32, 9)
            air = rng.normal(30.5, 1.8)
            surface = rng.normal(33, 2.8)
            humidity = rng.normal(63, 8)
            ndvi = rng.normal(0.68, 0.10)
            albedo = rng.normal(0.22, 0.025)
        rows.append({
            "timestamp": start + pd.Timedelta(minutes=i),
            "zone": zone,
            "air_temp_c": air,
            "humidity_pct": humidity,
            "surface_temp_c": surface,
            "green_cover_pct": green,
            "built_up_pct": built,
            "ndvi_proxy": ndvi,
            "albedo_proxy": albedo,
        })
    df = pd.DataFrame(rows)
    df["green_cover_pct"] = df["green_cover_pct"].clip(0, 100)
    df["built_up_pct"] = df["built_up_pct"].clip(0, 100)
    df["humidity_pct"] = df["humidity_pct"].clip(0, 100)
    df["ndvi_proxy"] = df["ndvi_proxy"].clip(-1, 1)
    df["albedo_proxy"] = df["albedo_proxy"].clip(0, 1)
    return df.sort_values("timestamp").reset_index(drop=True)

def analyze(df, contamination=0.08):
    X = df[FEATURES].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    out = df.copy()
    out["PC1"] = pcs[:, 0]
    out["PC2"] = pcs[:, 1]
    # This is a demonstration anomaly score, not a validated heat-risk model.
    detector = IsolationForest(contamination=contamination, random_state=42)
    out["anomaly"] = detector.fit_predict(X_scaled) == -1
    loadings = pd.DataFrame(
        pca.components_.T,
        index=FEATURES,
        columns=["PC1 loading", "PC2 loading"]
    )
    return out, pca, loadings

with st.sidebar:
    st.header("Simulation controls")
    n_rows = st.slider("Number of illustrative observations", 40, 300, 120, 20)
    seed = st.number_input("Simulation seed", min_value=0, max_value=99999, value=7)
    contamination = st.slider("Expected anomaly fraction (demo setting)", 0.02, 0.20, 0.08, 0.01)
    st.caption("Changing the seed creates a different synthetic dataset.")
    generate = st.button("Generate simulated dataset", type="primary")

if "sim_data" not in st.session_state or generate:
    st.session_state.sim_data = make_simulated_data(int(n_rows), int(seed))
df = st.session_state.sim_data.copy()
# Keep PCA analysis consistent with the selected demonstration settings.
result, pca, loadings = analyze(df, float(contamination))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Observations", len(result))
c2.metric("Illustrative mean air temp.", f"{result.air_temp_c.mean():.1f} °C")
c3.metric("Illustrative mean surface temp.", f"{result.surface_temp_c.mean():.1f} °C")
c4.metric("Flagged anomalies", int(result.anomaly.sum()))

st.subheader("1. Compare zones (simulated values)")
zone_summary = result.groupby("zone", as_index=False).agg(
    mean_air_temp_c=("air_temp_c", "mean"),
    mean_surface_temp_c=("surface_temp_c", "mean"),
    mean_humidity_pct=("humidity_pct", "mean"),
    mean_green_cover_pct=("green_cover_pct", "mean"),
    mean_built_up_pct=("built_up_pct", "mean"),
)
st.dataframe(zone_summary.round(2), use_container_width=True)
fig_temp = px.line(
    result.sort_values("timestamp"), x="timestamp", y="air_temp_c", color="zone",
    title="Illustrative air-temperature readings over simulated time",
    labels={"air_temp_c": "Air temperature (°C)", "timestamp": "Simulated timestamp"}
)
st.plotly_chart(fig_temp, use_container_width=True)

st.subheader("2. PCA scatter plot")
fig_pca = px.scatter(
    result, x="PC1", y="PC2", color="zone", symbol="anomaly",
    hover_data=["timestamp", "air_temp_c", "surface_temp_c", "humidity_pct", "green_cover_pct", "built_up_pct"],
    title="PCA of standardized synthetic environmental variables",
    labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"}
)
st.plotly_chart(fig_pca, use_container_width=True)
st.write(
    f"Explained variance: PC1 = {pca.explained_variance_ratio_[0]*100:.1f}%, "
    f"PC2 = {pca.explained_variance_ratio_[1]*100:.1f}%, "
    f"combined = {pca.explained_variance_ratio_.sum()*100:.1f}%."
)
st.caption(
    "PC1/PC2 meanings must be interpreted from the loadings for this run; they are not automatically "
    "equivalent to 'urbanization' or 'vegetation'. PCA shows association/patterns, not causation."
)

st.subheader("3. PCA loadings")
st.dataframe(loadings.round(3), use_container_width=True)
st.caption("Loadings show each standardized input variable's contribution to the two principal-component directions.")

st.subheader("4. Demonstration anomaly flags")
flagged = result[result["anomaly"]].sort_values("surface_temp_c", ascending=False)
if flagged.empty:
    st.info("No observations were flagged in this run. Try changing the seed or anomaly fraction.")
else:
    st.dataframe(
        flagged[["timestamp", "zone", "air_temp_c", "surface_temp_c", "humidity_pct",
                 "green_cover_pct", "built_up_pct", "PC1", "PC2", "anomaly"]].round(2),
        use_container_width=True
    )
st.caption(
    "Anomaly flags are generated by Isolation Forest over the selected synthetic variables. "
    "A flagged point is unusual relative to this simulated dataset; it is not automatically a dangerous heat event."
)

st.subheader("5. Download the dataset")
csv = result.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download simulated readings and PCA results as CSV",
    data=csv,
    file_name="urban_heat_simulated_results.csv",
    mime="text/csv"
)
st.success("Demo status: software simulation is running. No real sensor or field measurements are being used.")
