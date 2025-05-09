import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="Customer Satisfaction Analysis", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f7f7f9; }
    h1, h2, h3 { color: #003366; }
    .stMetricLabel { font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("Identifying Key Drivers of Customer Satisfaction through Survey Data Analysis")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("1. Raw Data Preview")
    st.dataframe(df.head())

    # Preprocessing
    df.drop(['id', 'Unnamed: 0'], axis=1, errors='ignore', inplace=True)
    df['satisfaction'] = df['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})

    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)

    # Metrics summary
    satisfied_pct = round((df['satisfaction'].sum() / len(df)) * 100, 2)
    dissatisfied_pct = 100 - satisfied_pct

    col1, col2 = st.columns(2)
    col1.metric("Satisfied Passengers", f"{satisfied_pct}%", delta_color="normal")
    col2.metric("Neutral/Dissatisfied", f"{dissatisfied_pct}%", delta_color="inverse")

    # Correlation Heatmap
    st.subheader("2. Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Split Data
    X = df.drop('satisfaction', axis=1)
    y = df['satisfaction']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model Training
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict(X_test)
    y_prob_log = log_model.predict_proba(X_test)[:, 1]

    # Logistic Metrics
    st.subheader("3. Logistic Regression Results")
    st.code(classification_report(y_test, y_pred_log), language="text")

    # Confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.title("Confusion Matrix - Logistic Regression")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob_log)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob_log):.2f}")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    # Logistic Feature Importance
    st.subheader("4. Feature Importance - Logistic Regression")
    coef_df = pd.Series(log_model.coef_[0], index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    coef_df.plot(kind='barh', ax=ax, color='teal')
    st.pyplot(fig)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    st.subheader("5. Random Forest Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax, palette='viridis')
    st.pyplot(fig)

    st.success("Top 10 Most Important Features")
    st.dataframe(feature_importance_df.head(10))

else:
    st.info("Awaiting input file... Please upload a customer satisfaction dataset in CSV format to continue.")