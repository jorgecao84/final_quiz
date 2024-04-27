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

st.write("COVID DATA - PATIENT PRIORITY PREDICTION. (XGBOOST ML MODEL)")

filename = st.file_uploader("Upload a file", type={"csv"})
if filename is not None:
    data = pd.read_csv(filename)
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

try:
    # Code that might raise exceptions
except Exception as e:
    pass
    
st.write("GIVE SOME TIME TO THE MODEL TO RUN")

# Separate features and target variable
X = data.drop(columns=['priority'])
y = data['priority']

# Encode the target variable
ordinal_encoder = OrdinalEncoder()
y = ordinal_encoder.fit_transform(y.array.reshape(-1,1))

# Step 4: Split the dataset into training and testing sets (80-20 split with stratification)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Further split the training set into training and evaluation sets (80-20 split with stratification)
X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Create XGBoost classifier
xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=16)  # Adjust objective and num_class for multi-class

xgb_classifier.fit(X_train, y_train)

# Step 15: Evaluate the XGBoost model on the training set
y_train_pred_xgb = xgb_classifier.predict(X_train)
st.write("XGBoost Training:")
st.write(classification_report(y_train, y_train_pred_xgb))


# Step 16: Evaluate the XGBoost model with the best hyperparameters on the evaluation set
y_eval_pred_xgb = xgb_classifier.predict(X_eval)
st.write("XGBoost Evaluation:")
st.write(classification_report(y_eval, y_eval_pred_xgb))


# Step 17: Evaluate the XGBoost model with the best hyperparameters on the testing set
y_test_pred_xgb = xgb_classifier.predict(X_test)
st.write("XGBoost Testing:")
st.write(classification_report(y_test, y_test_pred_xgb))

# Print accuracy
accuracy_train = accuracy_score(y_train, y_train_pred_xgb)
accuracy_eval = accuracy_score(y_eval,  y_eval_pred_xgb)
accuracy_test = accuracy_score(y_test,  y_test_pred_xgb)
st.write("Overall Accuracy - Training Set:", accuracy_train)
st.write("Overall Accuracy - Evaluation Set:", accuracy_eval)
st.write("Overall Accuracy - Testing Set:", accuracy_test)


