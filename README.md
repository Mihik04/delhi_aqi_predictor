# Delhi AQI Predictor

Live hyperlocal Air Quality Index prediction across 14 NCR monitoring stations.

## Live Demo
[View App](https://delhiaqipredictor0107.streamlit.app/)

## Features
- Live pollution readings via OpenAQ API
- Real-time weather via Open-Meteo
- Random Forest model · R² = 0.79
- K-Means clustering for pollution pattern detection
- Seasonal post-hoc calibration by station agency
- 14 monitoring stations across Delhi NCR

## Tech Stack
Python · Streamlit · Scikit-learn · Plotly · OpenAQ API · Open-Meteo

## Data Sources
- OpenAQ API — live sensor readings
- CPCB via Kaggle — historical 2015–2020
- Open-Meteo — weather & boundary layer height
- NASA FIRMS — satellite fire detection

## Model
- Random Forest Regressor · 52 engineered features
- Lag features (t-1, t-2, t-3) and rolling averages (6h, 24h)
- Time-based train/test split to prevent data leakage
- Identified and corrected data leakage that inflated R² from 0.79 to 0.98
- Post-hoc seasonal calibration achieving ~85% category accuracy
