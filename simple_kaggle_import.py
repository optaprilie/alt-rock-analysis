import streamlit as st
import kagglehub as kh
import pandas as pd
import os

# Initialize Streamlit page configuration
st.set_page_config(page_title="Basic Kaggle Data Import", layout="wide")

st.title("Basic Streamlit & Kaggle Import Showcase")
st.write("This file demonstrates how the project looks if we only initialize Streamlit and export the data from Kaggle using their library.")

# Function to download and load data using Kaggle's library
@st.cache_data
def load_data():
    # Exporting the data from kaggle using their library
    path = kh.dataset_download("thebumpkin/800-classic-alt-rock-tracks-with-spotify-data")
    csv_file_path = os.path.join(path, "ClassicAltRock.csv")
    
    # Read the downloaded dataset using pandas
    df = pd.read_csv(csv_file_path)
    return df

with st.spinner("Loading dataset from Kaggle..."):
    df = load_data()

# Showcasing the loaded dataset
st.subheader("Raw Dataset Preview")
st.dataframe(df)

st.subheader("Dataset Info")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
