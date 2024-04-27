import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import resample
from xgboost import XGBClassifier

st.write("Covid Care Priority Prediction Model")

filename = st.file_uploader("Upload a file", type={"csv"})
if filename is not None:
    dataset = pd.read_csv(filename)
    st.write(dataset.describe())
    chart_df = {'feature': [], 'average': [], 'label': []}
    for label in list(set(dataset["label"].values)):
        for column_name in dataset.columns:
            if column_name != "label":
                chart_df['feature'].append(column_name)
                chart_df['average'].append(dataset[dataset["label"]==label][column_name].mean())
                chart_df['label'].append(label)
    pd.DataFrame(chart_df)
fig = px.line(chart_df, x="feature", y="average",color="label")
st.plotly_chart(fig,use_container_width=True)
