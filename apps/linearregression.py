import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model


def app():
    st.title("Linear Regression")
    data = pd.read_csv("./onceki_lig_siralamalari.csv")
    pred_data = pd.read_csv("./play_off_takimlari_lig_siralamasi.csv")

    data = data.drop(columns=["Rk", "O_MP", "L", "PW", "PL", "Arena"]).rename(
        columns={"eFG%.1": "O_eFG%", "TOV%.1": "O_TOV%", "FT/FGA.1": "O_FT/FGA"}
    )
    pred_data = pred_data.drop(columns=["Rk", "O_MP", "L", "PW", "PL", "Arena"]).rename(
        columns={"O_eFG%_1": "O_eFG%", "O_TOV%_2": "O_TOV%"}
    )

    corr = data.corr().abs()
    corr = corr.loc[corr["Playoff Wins"] > 0.25]

    variables = list(corr.index)

    ######
    X = data[variables].drop("Playoff Wins", 1)
    y = data["Playoff Wins"]

    pred_X = pred_data[variables].drop("Playoff Wins", 1)

    X.head()

    # splitting data into training and testing sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    lr_model = linear_model.LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_wins_predicted = lr_model.predict(X_valid)
    st.text(
        "Mean Absolute Error: " + str(mean_absolute_error(lr_wins_predicted, y_valid))
    )
    lr_wins_predicted = lr_model.predict(pred_X)

    lr_wins_predicted_df = pred_data[["Team", "Playoff Wins"]]
    i = 0
    while i < 16:
        lr_wins_predicted_df.at[i, "Playoff Wins"] = lr_wins_predicted[i]
        i += 1
    st.table(lr_wins_predicted_df.sort_values(by="Playoff Wins", ascending=False))

