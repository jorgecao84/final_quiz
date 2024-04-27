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

# Apply SMOTE to the training data only
smote = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Convert resampled data to DMatrix format
#dtrain = xgb.DMatrix(X_train_resampled, label = y_train_resampled)
#deval = xgb.DMatrix(X_eval, label = y_eval)
#dtest = xgb.DMatrix(X_test, label = y_test)



# Define parameters grid for XGBoost model
param_grid = {
    'eta': [0.05, 0.1, 0.3],
    'max_depth': [3, 6, 9],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Create XGBoost classifier
#xgb_model = xgb.XGBClassifier(**params)
#xgb_model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the testing set


# Create XGBoost classifier
xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=16)  # Adjust objective and num_class for multi-class

# Perform GridSearchCV with cross-validation
xgb_grid_search = GridSearchCV(
    estimator=xgb_classifier,
    param_grid=param_grid,
    scoring='accuracy',  # Use appropriate scoring metric
    cv=5  # Number of folds for cross-validation
)

# Fit the grid search to the data
xgb_grid_search.fit(X_train_resampled, y_train_resampled)

# Step 15: Get the best hyperparameters for XGBoost
best_xgb_params = xgb_grid_search.best_params_
st.write("Best Hyperparameters for XGBoost:", best_xgb_params)

best_xgb_model = XGBClassifier(**best_xgb_params,random_state=42)
best_xgb_model.fit(X_train_resampled, y_train_resampled)

# Step 15: Evaluate the XGBoost model on the training set
y_train_pred_xgb = best_xgb_model.predict(X_train)
st.write("XGBoost Training:")
st.write(classification_report(y_train, y_train_pred_xgb))


# Step 16: Evaluate the XGBoost model with the best hyperparameters on the evaluation set
y_eval_pred_xgb = best_xgb_model.predict(X_eval)
st.write("XGBoost Evaluation:")
st.write(classification_report(y_eval, y_eval_pred_xgb))


# Step 17: Evaluate the XGBoost model with the best hyperparameters on the testing set
y_test_pred_xgb = best_xgb_model.predict(X_test)
st.write("XGBoost Testing:")
st.write(classification_report(y_test, y_test_pred_xgb))

warnings.filterwarnings("ignore")

# Print accuracy
accuracy_train = accuracy_score(y_train, y_train_pred_xgb)
accuracy_eval = accuracy_score(y_eval,  y_eval_pred_xgb)
accuracy_test = accuracy_score(y_test,  y_test_pred_xgb)
st.write("Overall Accuracy - Training Set:", accuracy_train)
st.write("Overall Accuracy - Evaluation Set:", accuracy_eval)
st.write("Overall Accuracy - Testing Set:", accuracy_test)

# Plotting
labels = ['Training Set', 'Evaluation Set', 'Testing Set']
accuracy = [accuracy_train, accuracy_eval, accuracy_test]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/6, accuracy, width, label='Overall')

ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of XGBoost')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()

