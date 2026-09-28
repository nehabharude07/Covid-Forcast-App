import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import r2_score
import io
import base64

plt.switch_backend('Agg')  # Use non-GUI backend

def preprocess_world_data(df0, continent_df):
    world = pd.DataFrame({"Country": df0.iloc[:, 1:].columns})
    cases = [pd.to_numeric(df0[c][1:]).sum() for c in world["Country"]]
    world["Cases"] = cases

    # Clean country names
    cleaned = []
    for country in world["Country"]:
        if "." in country:
            country = country.split(".")[0]
        elif "(" in country:
            country = country.split("(")[0].strip()
        cleaned.append(country)
    world["Country"] = cleaned
    world = world.groupby("Country")["Cases"].sum().reset_index()

    world["Cases Range"] = pd.cut(world["Cases"], [-150000, 50000, 200000, 800000, 1500000, 15000000],
                                  labels=["U50K", "50Kto200K", "200Kto800K", "800Kto1.5M", "1.5M+"])

    alpha = []
    for country in world["Country"].str.upper():
        if country == "BRUNEI":
            country = "BRUNEI DARUSSALAM"
        elif country == "US":
            country = "UNITED STATES"
        match = continent_df[continent_df["name"] == country]["alpha-3"]
        alpha.append(np.nan if match.empty else match.values[0])
    world["Alpha3"] = alpha

    return world.dropna()

def plot_choropleth(world):
    fig = px.choropleth(
        world,
        locations="Alpha3",
        color="Cases Range",
        projection="natural earth",

        # Hover information
        hover_name="Country",
        hover_data={
            "Alpha3": False,
            "Cases Range": True,
            "Cases": ":,.0f"
        },

        color_discrete_map={
            "U50K": "#f1f5b8",
            "50Kto200K": "#f5df00",
            "200Kto800K": "#ffb000",
            "800Kto1.5M": "#ff5a36",
            "1.5M+": "#ef1d1d"
        }
    )

    # ---------------------------------------------------------
    # MAP SETTINGS
    # ---------------------------------------------------------

    fig.update_geos(
        fitbounds="locations",
        visible=False,

        showland=True,
        landcolor="#18243a",

        showcountries=True,
        countrycolor="rgba(255,255,255,0.18)",

        showocean=True,
        oceancolor="#081525",

        showlakes=True,
        lakecolor="#0c2138",

        projection_scale=1.05
    )

    # ---------------------------------------------------------
    # HOVER STYLE
    # ---------------------------------------------------------

    fig.update_traces(
        marker_line_color="rgba(255,255,255,0.25)",
        marker_line_width=0.5,

        hovertemplate=(
            "<b>%{hovertext}</b>"
            "<br>Cases: %{customdata[1]:,.0f}"
            "<br>Range: %{customdata[0]}"
            "<extra></extra>"
        )
    )

    # ---------------------------------------------------------
    # LAYOUT
    # ---------------------------------------------------------

    fig.update_layout(

        # IMPORTANT:
        # Extra right space prevents legend + toolbar overlap
        margin=dict(
            l=10,
            r=190,
            t=75,
            b=15
        ),

        height=560,

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        # -----------------------------------------------------
        # TITLE
        # -----------------------------------------------------

        title=dict(
            text="🌍 Global COVID-19 Cases",
            x=0.50,
            xanchor="center",
            y=0.97,
            yanchor="top",

            font=dict(
                family="Segoe UI, Arial",
                size=25,
                color="#eaf2ff"
            )
        ),

        # -----------------------------------------------------
        # LEGEND
        # -----------------------------------------------------

        legend=dict(

            title=dict(
                text="<b>Cases Range</b>",
                font=dict(
                    size=14,
                    color="#ffffff"
                )
            ),

            # Move legend away from toolbar
            x=1.04,
            y=0.50,

            xanchor="left",
            yanchor="middle",

            orientation="v",

            bgcolor="rgba(8,18,35,0.92)",

            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,

            font=dict(
                size=12,
                color="#eaf2ff"
            ),

            itemsizing="constant",

            tracegroupgap=8
        ),

        # -----------------------------------------------------
        # INTERACTION
        # -----------------------------------------------------

        hoverlabel=dict(
            bgcolor="#0b1729",
            bordercolor="#60a5fa",
            font=dict(
                family="Segoe UI",
                size=13,
                color="white"
            )
        ),

        dragmode="zoom",

        # Better responsive behaviour
        autosize=True
    )

    # ---------------------------------------------------------
    # PLOTLY CONFIG
    # ---------------------------------------------------------

    config = {
        "responsive": True,

        "displayModeBar": True,

        "displaylogo": False,

        "scrollZoom": True,

        "doubleClick": "reset",

        "modeBarButtonsToRemove": [
            "lasso2d",
            "select2d"
        ]
    }

    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config=config
    )

def prepare_daily_data(df0, df1):
    cases = [pd.to_numeric(df0.iloc[i, 1:].values).sum() for i in range(1, len(df0))]
    deaths = [pd.to_numeric(df1.iloc[i, 1:].values).sum() for i in range(1, len(df1))]

    df = pd.DataFrame({
        "Date": df0["Country/Region"][1:],
        "Cases": cases,
        "Deaths": deaths
    }).set_index("Date")
    return df

class FbProphetModel:
    def __init__(self, df):
        self.df = df
        self.model = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False)

    def train(self):
        df_fb = pd.DataFrame({"ds": pd.to_datetime(self.df.index), "y": self.df["Cases"].values})
        self.model.fit(df_fb)
        future = self.model.make_future_dataframe(periods=30, freq='D')
        self.forecast = self.model.predict(future)

    def plot_forecast(self):
        df_forecast = self.forecast[["ds", "yhat_lower", "yhat_upper", "yhat"]].tail(30)
        df_forecast.set_index("ds", inplace=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_forecast.index, df_forecast["yhat"], marker='.')
        ax.fill_between(df_forecast.index, df_forecast["yhat_lower"], df_forecast["yhat_upper"], color="gray", alpha=0.3)
        ax.set_title("Forecasting of Next 30 Days Cases")
        ax.legend(["Forecast", "Bound"])

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return encoded
