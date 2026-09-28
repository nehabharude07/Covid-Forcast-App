from flask import Flask, render_template
from utils.forecast_utils import preprocess_world_data, plot_choropleth, prepare_daily_data, FbProphetModel
import pandas as pd
from prophet import Prophet

app = Flask(__name__)

df0 = pd.read_csv("CONVENIENT_global_confirmed_cases.csv")
df1 = pd.read_csv("CONVENIENT_global_deaths.csv")
continent = pd.read_csv("continents2.csv")
continent["name"] = continent["name"].str.upper()

@app.route("/")
def index():
    # Choropleth
    world = preprocess_world_data(df0, continent)
    choropleth_html = plot_choropleth(world)

    # Time-series and Forecasting
    df = prepare_daily_data(df0, df1)
    model = FbProphetModel(df)
    model.train()
    forecast_img = model.plot_forecast()

    return render_template("index.html", choropleth_html=choropleth_html, forecast_img=forecast_img)


if __name__ == "__main__":
    app.run(debug=True, port=5161)
