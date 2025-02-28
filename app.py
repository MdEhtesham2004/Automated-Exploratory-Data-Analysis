import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Streamlit App Title
st.title("📊 AutoEDA: Automated Exploratory Data Analysis")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load dataset
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
    
    st.subheader("📌 Dataset Overview")
    st.write("**First 5 Rows:**")
    st.dataframe(df.head())
    
    st.write("**Dataset Shape:**", df.shape)
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())
    
    # Data Cleaning Options
    if st.checkbox("Remove Duplicates"):
        df = df.drop_duplicates()
        st.success("Duplicates removed!")
    
    if st.checkbox("Drop Missing Values"):
        df = df.dropna()
        st.success("Missing values dropped!")
    
    # Univariate Analysis
    st.subheader("📊 Univariate Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if numeric_cols:
        selected_num_col = st.selectbox("Select a numerical column", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_num_col], kde=True, ax=ax)
        st.pyplot(fig)
    
    if categorical_cols:
        selected_cat_col = st.selectbox("Select a categorical column", categorical_cols)
        fig, ax = plt.subplots()
        sns.countplot(x=df[selected_cat_col], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Bivariate Analysis
    st.subheader("📈 Bivariate Analysis")
    if len(numeric_cols) > 1:
        x_var = st.selectbox("Select X variable", numeric_cols)
        y_var = st.selectbox("Select Y variable", numeric_cols, index=1)
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x_var], y=df[y_var], ax=ax)
        st.pyplot(fig)
    
    # Correlation Analysis
    st.subheader("🔍 Correlation Heatmap")
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    
    # Download Processed Data
    st.subheader("📥 Download Cleaned Data")
    csv = df.to_csv(index=False).encode('utf-8')
    download_button = st.download_button(label="Download CSV", data=csv, file_name="cleaned_data.csv", mime="text/csv")
    if download_button:
        st.success("Your cleaned dataset is downloaded!")
else:
    st.info("Upload a dataset to start EDA!")