# COVID-19 Forecasting and Global Case Analysis

A Flask-based web application for analyzing global COVID-19 cases and forecasting future cases using **Facebook Prophet**. The application provides an interactive global choropleth map and a 30-day COVID-19 case forecast.

---

## 🚀 Features

- 🌍 Interactive global COVID-19 case map
- 📊 Country-wise COVID-19 case range visualization
- 🔎 Interactive Plotly map with zoom, pan, hover, and legend controls
- 📈 30-day COVID-19 case forecasting
- 🤖 Time-series forecasting using Prophet
- 🎨 Premium glassmorphism and 3D dashboard UI
- 📱 Responsive web interface
- ⚡ Flask-based backend
- 📉 Forecast visualization with prediction bounds

---

## 🛠️ Technologies Used

### Backend
- Python
- Flask
- Pandas
- NumPy

### Machine Learning
- Facebook Prophet
- Scikit-learn

### Data Visualization
- Plotly
- Matplotlib

### Frontend
- HTML
- CSS
- JavaScript

---

## 📂 Project Structure

```text
covid_forecast_app/
│
├── app.py
│
├── requirements.txt
│
├── .gitignore
├── README.md
│
├── CONVENIENT_global_confirmed_cases.csv
├── CONVENIENT_global_deaths.csv
├── continents2.csv
│
├── utils/
│   └── forecast_utils.py
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
