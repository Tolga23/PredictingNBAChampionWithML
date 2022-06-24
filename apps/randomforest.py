import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def app():
    st.title("Random Forest")
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

    # using cross validation score to build first model
    from sklearn.model_selection import cross_val_score

    def rfr_cvs(n_estimators, cv):
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
        scores = -1 * cross_val_score(
            model, X, y, cv=cv, scoring="neg_mean_absolute_error"
        )
        return (
            "score = "
            + str(scores.mean())
            + " with n_estimators: "
            + str(n_estimators)
            + " and cv: "
            + str(cv)
        )

    # comparing cross validation scores across different numbers of trees and testing if 5 or 10 k-folds is optimal
    for n_estimators in np.arange(start=100, stop=400, step=50):
        print(rfr_cvs(n_estimators, 10))

    rfr_model = RandomForestRegressor(n_estimators=350, random_state=0)
    rfr_model.fit(X_train, y_train)
    rfr_wins_predicted = rfr_model.predict(X_valid)
    st.text(
        "Mean Absolute Error: " + str(mean_absolute_error(rfr_wins_predicted, y_valid))
    )

    rfr_wins_predicted = rfr_model.predict(pred_X)
    rfr_wins_predicted

    rfr_wins_predicted_df = pred_data[["Team", "Playoff Wins"]]
    i = 0
    while i < 16:
        rfr_wins_predicted_df.at[i, "Playoff Wins"] = rfr_wins_predicted[i]
        i += 1
    st.table(rfr_wins_predicted_df.sort_values(by="Playoff Wins", ascending=False))

