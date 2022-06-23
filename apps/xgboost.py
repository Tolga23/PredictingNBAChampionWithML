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
from openpyxl import Workbook

from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score


def app():
    st.title('XGBoost')
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

    X.head()

        #splitting data into training and testing sets
    X_train, X_valid, y_train, y_valid = train_test_split(X,y)



    def xgb_cvs(n_estimators, learning_rate, cv):
        model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
        scores = -1*cross_val_score(model,X,y,cv=cv,scoring='neg_mean_absolute_error')
        return "score = "+str(scores.mean())+" with n_estimators: "+str(n_estimators)+" and cv: "+str(cv)


    xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective='reg:linear', nthread=-1, scale_pos_weight=1, seed=27)
    xgb_model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_valid, y_valid)], verbose=False)
    xgb_predicted_wins = xgb_model.predict(X_valid)
    st.text("Mean Absolute Error: " + str(mean_absolute_error(xgb_predicted_wins, y_valid)))
        #####

    xgb_predicted_wins = xgb_model.predict(pred_X)
    xgb_predicted_wins 

    xgb_predicted_wins_df = pred_data[['Team','Playoff Wins']]
    i=0
    while i<16:
        xgb_predicted_wins_df.at[i, 'Playoff Wins'] = xgb_predicted_wins[i]
        i+=1
    st.table(xgb_predicted_wins_df.sort_values(by='Playoff Wins',ascending=False))

    if st.button('Download'):
        xgb_predicted_wins_df.to_excel('./xgb_wins_predicted.xlsx', index=False)
        st.success('Excel File downloaded successfully')