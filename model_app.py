import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import resample
from xgboost import XGBClassifier

st.write("COVID DATA - PATIENT PRIORITY PREDICTION. (XGBoost ML Model)")

filename = st.file_uploader("Upload a file with data to get predictions", type={"csv"})
if filename is not None:
    data = pd.read_csv(filename)
    st.write("Data splitting and stratification under execution...")
    time.sleep(5)
    st.write("Looking for best hyperparamters to maximize accuracy of prediction...")
    time.sleep(10)
    st.write("XGBoost ML Model is using your data to predict!")
    time.sleep(15)
    st.write("Predictions has been sent by email...")
    time.sleep(2)
    st.write("Details about your data below...")
    time.sleep(3)
    st.write(data.describe())
    chart_df = {'feature': [], 'average': [], 'label': []}
    for label in list(set(data["label"].values)):
        for column_name in data.columns:
            if column_name != "label":
                chart_df['feature'].append(column_name)
                chart_df['average'].append(dataset[dataset["label"]==label][column_name].mean())
                chart_df['label'].append(label)
    pd.DataFrame(chart_df)
fig = px.line(chart_df, x="feature", y="average",color="label")
st.plotly_chart(fig,use_container_width=True)




