import streamlit as st
from multiapp import MultiApp
from apps import (
    main,
    randomforest,
    xgboost,
    linearregression,
    adaboost
)  # import your app modules here

app = MultiApp()

st.title("Makina Öğrenmesi Yöntemiyle NBA Şampiyonunu Tahmin Etme")

# Add all your application here
app.add_app("Ana Sayfa", main.app)
app.add_app("Linear Regression", linearregression.app)
app.add_app("Random Forest", randomforest.app)
app.add_app("XGBoost", xgboost.app)
app.add_app("AdaBoost", adaboost.app)

# The main app
app.run()
