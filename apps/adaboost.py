from ntpath import join
from unittest import result
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
    

from sklearn.model_selection import cross_val_score


def app():
    st.title('AdaBoost')
    data = pd.read_csv('./onceki_lig_siralamalari.csv')
    pred_data = pd.read_csv('./play_off_takimlari_lig_siralamasi.csv')

    data = data.drop(columns = ['Rk','O_MP','L','PW','PL','Arena']).rename(columns={'eFG%.1':'O_eFG%','TOV%.1':'O_TOV%','FT/FGA.1':'O_FT/FGA'})
    pred_data = pred_data.drop(columns = ['Rk','O_MP','L','PW','PL','Arena']).rename(columns={'O_eFG%_1':'O_eFG%','O_TOV%_2':'O_TOV%'})

    corr = data.corr().abs()
    corr = corr.loc[corr['Playoff Wins']>.25]

    variables = list(corr.index)
    corr_df = data[variables].groupby('Playoff Wins').mean()




    X = data[variables].drop('Playoff Wins',1)
    y = data['Playoff Wins']

    pred_X = pred_data[variables].drop('Playoff Wins',1)

        #splitting data into training and testing sets
    X_train, X_valid, y_train, y_valid = train_test_split(X,y)

    ada_model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=2), n_estimators=100, learning_rate=0.1)
    ada_model.fit(X_train, y_train)
    ada_wins_predicted = ada_model.predict(X_valid)
    st.text('Mean Absolute Error: ' + str(mean_absolute_error(ada_wins_predicted, y_valid)))
    ada_wins_predicted = ada_model.predict(pred_X)

    ada_wins_predicted_df = pred_data[['Team','Playoff Wins']]
    i = 0
    while i < 16:
        ada_wins_predicted_df.at[i,'Playoff Wins'] = ada_wins_predicted[i]
        i += 1
    st.table(ada_wins_predicted_df.sort_values(by="Playoff Wins", ascending=False))

   







   
    