import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    page_title="Delhi AQI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; }

section[data-testid="stSidebar"] {
    background: #0a0a0a;
    border-right: 1px solid #1a1a1a;
}
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.82rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

[data-testid="metric-container"] {
    background: #0f0f0f;
    border: 1px solid #1e1e1e;
    border-radius: 3px;
    padding: 1rem 1.2rem;
}
[data-testid="metric-container"] label {
    font-size: 0.68rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #555 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace;
    font-size: 1.5rem !important;
    color: #e0e0e0 !important;
}

.stButton button {
    background: #e0e0e0;
    color: #000;
    border: none;
    border-radius: 2px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.6rem 2rem;
}
.stButton button:hover { background: #fff; }

.stSelectbox > div > div {
    background: #0f0f0f;
    border: 1px solid #222;
    border-radius: 2px;
    color: #ccc;
}

.section-label {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #444;
    border-bottom: 1px solid #181818;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
    margin-top: 2rem;
}

.aqi-block {
    padding: 1.8rem 2rem;
    margin-top: 1rem;
    background: #080808;
    border-left: 3px solid #fff;
}
.aqi-num {
    font-family: 'DM Mono', monospace;
    font-size: 4.5rem;
    font-weight: 400;
    line-height: 1;
}
.aqi-cat {
    font-size: 0.78rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.6rem;
}
.aqi-adv {
    font-size: 0.82rem;
    color: #666;
    margin-top: 0.8rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    model        = joblib.load('delhi_aqi_model.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    scaler       = joblib.load('scaler.pkl')
    kmeans       = joblib.load('kmeans_model.pkl')
    return model, feature_cols, scaler, kmeans

@st.cache_data
def load_data():
    df = pd.read_csv('delhi_clustered.csv')
    df['timestamp_hour'] = pd.to_datetime(df['timestamp_hour'])
    return df

model, feature_cols, scaler, kmeans = load_models()
df = load_data()

with st.sidebar:
    st.markdown(
        "<div style='font-size:1rem;font-weight:600;letter-spacing:0.05em;margin-bottom:1rem'>Delhi AQI</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    page = st.radio("", ["Home", "Predictor", "Station Map", "Analysis"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.7rem;color:#3a3a3a;line-height:1.8'>"
        "Random Forest · R² 0.79<br>14 stations · Live data<br>"
        "OpenAQ + Open-Meteo"
        "</div>",
        unsafe_allow_html=True
    )

station_info = {
    "Aya Nagar — IMD":             {"id": 5570,  "lat": 28.4744, "lon": 77.1316, "agency": "IMD"},
    "Sirifort — CPCB":             {"id": 5586,  "lat": 28.5494, "lon": 77.2197, "agency": "CPCB"},
    "ITO — CPCB":                  {"id": 5613,  "lat": 28.6289, "lon": 77.2412, "agency": "CPCB"},
    "Sector-62, Noida — IMD":      {"id": 5616,  "lat": 28.6272, "lon": 77.3692, "agency": "IMD"},
    "NSIT Dwarka — CPCB":          {"id": 5622,  "lat": 28.6090, "lon": 77.0305, "agency": "CPCB"},
    "Lodhi Road — IMD":            {"id": 5634,  "lat": 28.5918, "lon": 77.2273, "agency": "IMD"},
    "IGI Airport T3 — IMD":        {"id": 5650,  "lat": 28.5562, "lon": 77.0999, "agency": "IMD"},
    "Vasundhara — UPPCB":          {"id": 5665,  "lat": 28.6612, "lon": 77.3476, "agency": "UPPCB"},
    "Alipur — DPCC":               {"id": 6932,  "lat": 28.7969, "lon": 77.1336, "agency": "DPCC"},
    "Jahangirpuri — DPCC":         {"id": 8235,  "lat": 28.7279, "lon": 77.1620, "agency": "DPCC"},
    "Bawana — DPCC":               {"id": 8472,  "lat": 28.7930, "lon": 77.0420, "agency": "DPCC"},
    "Sector-51, Gurugram — HSPCB": {"id": 10825, "lat": 28.4421, "lon": 77.0580, "agency": "HSPCB"},
    "Rohini — DPCC":               {"id": 10831, "lat": 28.7041, "lon": 77.1025, "agency": "DPCC"},
    "Chandni Chowk — IITM":        {"id": 11603, "lat": 28.6508, "lon": 77.2311, "agency": "IITM"},
}

def get_calibration(agency, month):
    table = {
        "DPCC":  [([ 12,1,2],0.65),([ 10,11],0.80),([3,4,5],0.55),([6,7,8,9],0.70)],
        "CPCB":  [([ 12,1,2],0.80),([ 10,11],0.90),([3,4,5],0.75),([6,7,8,9],0.85)],
        "IMD":   [([ 12,1,2],0.88),([ 10,11],0.95),([3,4,5],0.85),([6,7,8,9],0.90)],
        "HSPCB": [([ 12,1,2],0.85),([ 10,11],0.92),([3,4,5],0.85),([6,7,8,9],0.88)],
        "UPPCB": [([ 12,1,2],0.82),([ 10,11],0.90),([3,4,5],0.80),([6,7,8,9],0.85)],
        "IITM":  [([ 12,1,2],0.80),([ 10,11],0.90),([3,4,5],0.75),([6,7,8,9],0.85)],
    }
    for months, val in table.get(agency, table["CPCB"]):
        if month in months:
            return val
    return 0.80

def aqi_meta(val):
    if val <= 50:   return "Good",      "#4caf50", "Air quality is satisfactory. Safe for all activities."
    if val <= 100:  return "Moderate",  "#ffeb3b", "Acceptable. Sensitive groups should reduce prolonged outdoor exertion."
    if val <= 200:  return "Poor",      "#ff9800", "Sensitive groups should avoid outdoor activities."
    if val <= 300:  return "Very Poor", "#f44336", "Limit outdoor exposure. Wear masks when outside."
    return             "Severe",        "#9c27b0", "Health emergency. Stay indoors. Avoid all outdoor activity."

PLOT = dict(
    paper_bgcolor="#080808", plot_bgcolor="#080808",
    font=dict(family="DM Sans", color="#555", size=11),
    xaxis=dict(gridcolor="#141414", linecolor="#1e1e1e", zerolinecolor="#1e1e1e"),
    yaxis=dict(gridcolor="#141414", linecolor="#1e1e1e", zerolinecolor="#1e1e1e"),
    margin=dict(l=40, r=20, t=30, b=40)
)

# ─── HOME ────────────────────────────────────────────────────────────────────
if page == "Home":
    st.markdown("## Delhi Air Quality Predictor")
    st.markdown(
        "<div style='color:#555;font-size:0.88rem;margin-bottom:2rem;line-height:1.6'>"
        "Hyperlocal AQI prediction across 14 NCR monitoring stations using live sensor data, "
        "real-time weather, and a Random Forest model trained on historical CPCB data."
        "</div>", unsafe_allow_html=True
    )

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Training Rows", "47,672")
    c2.metric("Stations",      "14")
    c3.metric("Model R²",      "0.79")
    c4.metric("Parameters",    "5")

    st.markdown("<div class='section-label'>Model Pipeline</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#555;font-size:0.85rem;line-height:2;max-width:720px'>"
        "Raw pollution readings → outlier removal → pivot to wide format → "
        "lag features (t−1, t−2, t−3) → rolling averages (6h, 24h) → "
        "weather merge → festival & season flags → Random Forest training → "
        "post-hoc seasonal calibration → live prediction"
        "</div>", unsafe_allow_html=True
    )

    st.markdown("<div class='section-label'>Data Sources</div>", unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            "<div style='color:#555;font-size:0.82rem;line-height:2'>"
            "OpenAQ API — live pollution readings<br>"
            "CPCB via Kaggle — historical 2015–2020"
            "</div>", unsafe_allow_html=True
        )
    with d2:
        st.markdown(
            "<div style='color:#555;font-size:0.82rem;line-height:2'>"
            "Open-Meteo — weather & boundary layer height<br>"
            "NASA FIRMS — satellite fire detection"
            "</div>", unsafe_allow_html=True
        )

    st.markdown("<div class='section-label'>Sample Data</div>", unsafe_allow_html=True)
    st.dataframe(
        df[['timestamp_hour','station_name','pm25','pm10','no2','AQI']].head(20),
        use_container_width=True
    )

# ─── PREDICTOR ───────────────────────────────────────────────────────────────
elif page == "Predictor":
    st.markdown("## AQI Predictor")
    st.markdown(
        "<div style='color:#555;font-size:0.85rem;margin-bottom:1.5rem'>"
        "Select a station. Live sensor readings and weather are fetched automatically."
        "</div>", unsafe_allow_html=True
    )

    selected = st.selectbox("", list(station_info.keys()), label_visibility="collapsed")

    if st.button("Fetch Live Data"):
        import requests
        with st.spinner("Fetching live data..."):
            try:
                info        = station_info[selected]
                location_id = info["id"]
                lat, lon    = info["lat"], info["lon"]
                agency      = info["agency"]
                api_key     = os.getenv("OPENAQ_API_KEY")

                res     = requests.get(
                    f"https://api.openaq.org/v3/locations/{location_id}/sensors",
                    headers={"X-API-Key": api_key, "accept": "application/json"}
                )
                sensors = res.json()

                live_values, sensor_ids = {}, {}
                for sensor in sensors["results"]:
                    param = sensor["parameter"]["name"]
                    if param in ["pm25","pm10","no2","co","o3"]:
                        if sensor["latest"]["value"] is not None:
                            live_values[param] = sensor["latest"]["value"]
                            sensor_ids[param]  = sensor["id"]

                if len(live_values) >= 3:
                    historical = {}
                    for param, sid in sensor_ids.items():
                        h = requests.get(
                            f"https://api.openaq.org/v3/sensors/{sid}/measurements",
                            headers={"X-API-Key": api_key, "accept": "application/json"},
                            params={"limit": 24, "page": 1}
                        ).json()
                        if "results" in h:
                            historical[param] = [r["value"] for r in h["results"] if r["value"] > 0]

                    wj = requests.get(
                        "https://api.open-meteo.com/v1/forecast",
                        params={
                            "latitude": lat, "longitude": lon,
                            "current": ["temperature_2m","relative_humidity_2m",
                                        "wind_speed_10m","wind_direction_10m",
                                        "precipitation","surface_pressure"],
                            "hourly": ["boundary_layer_height"],
                            "timezone": "Asia/Kolkata", "forecast_days": 1
                        }
                    ).json()
                    weather = wj["current"]
                    now     = datetime.now()
                    bl      = wj["hourly"]["boundary_layer_height"][now.hour]

                    st.markdown("<div class='section-label'>Live Sensor Readings</div>", unsafe_allow_html=True)
                    c1,c2,c3,c4,c5 = st.columns(5)
                    c1.metric("PM2.5", f"{live_values.get('pm25',0):.1f}" if 'pm25' in live_values else "—")
                    c2.metric("PM10",  f"{live_values.get('pm10',0):.1f}" if 'pm10' in live_values else "—")
                    c3.metric("NO2",   f"{live_values.get('no2',0):.1f}"  if 'no2'  in live_values else "—")
                    c4.metric("CO",    f"{live_values.get('co',0):.2f}"   if 'co'   in live_values else "—")
                    c5.metric("O3",    f"{live_values.get('o3',0):.1f}"   if 'o3'   in live_values else "—")

                    st.markdown("<div class='section-label'>Current Weather</div>", unsafe_allow_html=True)
                    w1,w2,w3,w4 = st.columns(4)
                    w1.metric("Temperature",   f"{weather['temperature_2m']} °C")
                    w2.metric("Humidity",       f"{weather['relative_humidity_2m']} %")
                    w3.metric("Wind Speed",     f"{weather['wind_speed_10m']} km/h")
                    w4.metric("Boundary Layer", f"{bl:.0f} m")

                    def get_lag(p,n):
                        v = historical.get(p,[])
                        return v[n] if len(v)>n else live_values.get(p,0)
                    def get_roll(p,n):
                        v = historical.get(p,[])
                        s = v[:n] if len(v)>=n else v
                        return np.mean(s) if s else live_values.get(p,0)

                    pm25=live_values.get('pm25',80); pm10=live_values.get('pm10',150)
                    no2=live_values.get('no2',40);   co=live_values.get('co',1.0)
                    o3=live_values.get('o3',50)
                    hour=now.hour; month=now.month

                    inp = pd.DataFrame([{
                        'hour':hour,'day_of_week':now.weekday(),'month':month,
                        'year':now.year,'day_of_year':now.timetuple().tm_yday,
                        'is_weekend':1 if now.weekday()>=5 else 0,
                        'is_peak_traffic':1 if(8<=hour<=10 or 17<=hour<=20)else 0,
                        'is_monsoon':1 if 6<=month<=9 else 0,
                        'is_winter':1 if month in[12,1,2]else 0,
                        'is_stubble_season':1 if month in[10,11]else 0,
                        'is_summer':1 if 3<=month<=5 else 0,
                        'is_diwali':0,'is_holi':0,'is_dussehra':0,
                        'temperature_2m':weather['temperature_2m'],
                        'relative_humidity_2m':weather['relative_humidity_2m'],
                        'wind_speed_10m':weather['wind_speed_10m'],
                        'wind_direction_10m':weather.get('wind_direction_10m',180),
                        'precipitation':weather.get('precipitation',0),
                        'surface_pressure':weather.get('surface_pressure',990),
                        'boundary_layer_height':bl,
                        'wind_dispersion':weather['wind_speed_10m']*bl,
                        'co':co,'no2':no2,'o3':o3,'pm10':pm10,'pm25':pm25,
                        'pm25_lag1':get_lag('pm25',1),'pm25_lag2':get_lag('pm25',2),'pm25_lag3':get_lag('pm25',3),
                        'pm10_lag1':get_lag('pm10',1),'pm10_lag2':get_lag('pm10',2),'pm10_lag3':get_lag('pm10',3),
                        'no2_lag1':get_lag('no2',1),'no2_lag2':get_lag('no2',2),'no2_lag3':get_lag('no2',3),
                        'co_lag1':get_lag('co',1),'co_lag2':get_lag('co',2),'co_lag3':get_lag('co',3),
                        'o3_lag1':get_lag('o3',1),'o3_lag2':get_lag('o3',2),'o3_lag3':get_lag('o3',3),
                        'pm25_roll6':get_roll('pm25',6),'pm25_roll24':get_roll('pm25',24),
                        'pm10_roll6':get_roll('pm10',6),'pm10_roll24':get_roll('pm10',24),
                        'no2_roll6':get_roll('no2',6),'no2_roll24':get_roll('no2',24),
                        'co_roll6':get_roll('co',6),'co_roll24':get_roll('co',24),
                        'o3_roll6':get_roll('o3',6),'o3_roll24':get_roll('o3',24),
                    }])

                    raw  = model.predict(inp[feature_cols])[0]
                    pred = raw * get_calibration(agency, month)
                    # K-Means cluster prediction
                    try:
                        cluster_features = np.array([[
                            pm25, pm10, no2, co, o3,
                            hour, month,
                            1 if now.weekday() >= 5 else 0,
                            weather['temperature_2m'],
                            weather['wind_speed_10m'],
                            weather['relative_humidity_2m']
                        ]])
                        cluster_scaled = scaler.transform(cluster_features)
                        cluster_id = kmeans.predict(cluster_scaled)[0]
                        cluster_names = {0: "Moderate", 1: "Clean", 2: "Severe", 3: "High"}
                        cluster_label = cluster_names.get(cluster_id, "Unknown")
                        cluster_colors = {
                            "Clean": "#4caf50",
                            "Moderate": "#ffeb3b", 
                            "High": "#ff9800",
                            "Severe": "#f44336"
                        }
                        cluster_color = cluster_colors.get(cluster_label, "#888")
                    except:
                        cluster_label = "Unknown"
                        cluster_color = "#888"
                    cat, color, advice = aqi_meta(pred)

                    st.markdown("<div class='section-label'>Prediction</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='aqi-block' style='border-left-color:{color}'>"
                        f"<div style='font-size:0.65rem;letter-spacing:0.2em;text-transform:uppercase;color:#444;margin-bottom:0.4rem'>Predicted AQI · {selected}</div>"
                        f"<div class='aqi-num' style='color:{color}'>{pred:.0f}</div>"
                        f"<div class='aqi-cat' style='color:{color}'>{cat}</div>"
                        f"<div class='aqi-adv'>{advice}</div>"
                        f"<div style='margin-top:1.2rem;padding-top:1rem;border-top:1px solid #1a1a1a'>"
                        f"<div style='font-size:0.65rem;letter-spacing:0.2em;text-transform:uppercase;color:#444;margin-bottom:0.3rem'>Pollution Pattern — K-Means Cluster</div>"
                        f"<div style='font-size:1rem;letter-spacing:0.1em;text-transform:uppercase;color:{cluster_color};font-family:DM Mono,monospace'>{cluster_label}</div>"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("Insufficient live sensor data for this station.")

            except Exception as e:
                st.error(f"Error: {e}")

# ─── STATION MAP ─────────────────────────────────────────────────────────────
elif page == "Station Map":
    st.markdown("## Monitoring Stations")
    st.markdown(
        "<div style='color:#555;font-size:0.85rem;margin-bottom:1.5rem'>"
        "14 government monitoring stations across Delhi NCR."
        "</div>", unsafe_allow_html=True
    )

    map_data = pd.DataFrame([
        {"Station": name, "Agency": info["agency"], "lat": info["lat"], "lon": info["lon"]}
        for name, info in station_info.items()
    ])

    color_map = {"DPCC":"#f44336","CPCB":"#2196f3","IMD":"#4caf50",
                 "HSPCB":"#ff9800","UPPCB":"#9c27b0","IITM":"#00bcd4"}

    fig = px.scatter_mapbox(
        map_data, lat="lat", lon="lon",
        hover_name="Station",
        hover_data={"Agency":True,"lat":False,"lon":False},
        color="Agency", color_discrete_map=color_map,
        size=[12]*len(map_data), zoom=10, height=560
    )
    fig.update_layout(
        mapbox_style="carto-darkmatter",
        paper_bgcolor="#080808",
        margin=dict(l=0,r=0,t=0,b=0),
        legend=dict(bgcolor="#0f0f0f",bordercolor="#1e1e1e",borderwidth=1,
                    font=dict(size=11,color="#666"))
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── ANALYSIS ────────────────────────────────────────────────────────────────
elif page == "Analysis":
    st.markdown("## Pollution Analysis")

    st.markdown("<div class='section-label'>Pollution Category Distribution</div>", unsafe_allow_html=True)
    counts = df['pollution_category'].value_counts().reset_index()
    counts.columns = ['Category','Count']
    cmap = {'Clean':'#4caf50','Moderate':'#ffeb3b','High':'#ff9800','Severe':'#f44336'}
    fig1 = px.bar(counts, x='Category', y='Count', color='Category', color_discrete_map=cmap)
    fig1.update_traces(marker_line_width=0)
    fig1.update_layout(**PLOT, showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-label'>Monthly PM2.5 Average</div>", unsafe_allow_html=True)
        monthly = df.groupby('month')['pm25'].mean().reset_index()
        fig2 = px.line(monthly, x='month', y='pm25', markers=True)
        fig2.update_traces(line_color='#aaa', marker_color='#aaa', marker_size=5)
        fig2.update_layout(**PLOT, xaxis_title="Month", yaxis_title="PM2.5 (µg/m³)")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("<div class='section-label'>Hourly PM2.5 Pattern</div>", unsafe_allow_html=True)
        hourly = df.groupby('hour')['pm25'].mean().reset_index()
        fig3 = px.bar(hourly, x='hour', y='pm25')
        fig3.update_traces(marker_color='#2a2a2a', marker_line_color='#333', marker_line_width=1)
        fig3.update_layout(**PLOT, xaxis_title="Hour of Day", yaxis_title="PM2.5 (µg/m³)")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='section-label'>Parameter Correlations</div>", unsafe_allow_html=True)
    corr = df[['pm25','pm10','no2','co','o3','AQI']].corr()
    fig4 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig4.update_layout(**PLOT)
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<div class='section-label'>Station-wise Average PM2.5</div>", unsafe_allow_html=True)
    savg = df.groupby('station_name')['pm25'].mean().sort_values().reset_index()
    fig5 = px.bar(savg, x='pm25', y='station_name', orientation='h')
    fig5.update_traces(marker_color='#1e1e1e', marker_line_color='#2e2e2e', marker_line_width=1)
    fig5.update_layout(**PLOT, height=420, xaxis_title="Average PM2.5 (µg/m³)", yaxis_title="")
    st.plotly_chart(fig5, use_container_width=True)