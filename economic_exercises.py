import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub as kh
import os

# Initialize Streamlit page configuration
st.set_page_config(page_title="Economic Interpretations", layout="wide")

st.title("Economic Interpretations of the Alt-Rock Dataset")
st.write("This file contains standalone exercises showcasing how to apply economic theories to the dataset.")

# Function to download and load data using Kaggle's library
@st.cache_data
def load_data():
    path = kh.dataset_download("thebumpkin/800-classic-alt-rock-tracks-with-spotify-data")
    csv_file_path = os.path.join(path, "ClassicAltRock.csv")
    df = pd.read_csv(csv_file_path)
    return df

with st.spinner("Loading dataset..."):
    df = load_data()

st.divider()

# ==========================================
# EXERCISE 1: Attention Economy
# ==========================================
st.subheader("Economic Exercise 1: The Attention Economy")
st.write("Does the market reward shorter songs under the streaming payout model?")

if st.button("Analyze Duration vs. Attention (Popularity)"):
    # Convert ms to minutes for readability
    df['Duration_Mins'] = df['Duration'] / 60000 
    
    fig_econ1, ax_econ1 = plt.subplots(figsize=(10, 6))
    ax_econ1.scatter(df['Duration_Mins'], df['Popularity'], alpha=0.5, color='teal')
    
    # Filter for finite values to safely calculate trendline
    clean_df = df.dropna(subset=['Duration_Mins', 'Popularity'])
    if not clean_df.empty:
        z = np.polyfit(clean_df['Duration_Mins'], clean_df['Popularity'], 1)
        p = np.poly1d(z)
        ax_econ1.plot(clean_df['Duration_Mins'], p(clean_df['Duration_Mins']), "r--", linewidth=2)
    
    ax_econ1.set_xlabel("Song Duration (Minutes)", fontweight='bold')
    ax_econ1.set_ylabel("Market Demand (Popularity)", fontweight='bold')
    ax_econ1.set_title("Scarcity of Attention: Duration vs. Popularity", fontsize=14, fontweight='bold')
    st.pyplot(fig_econ1)
    
    st.info("💡 **Economic Interpretation:** If the red trendline slopes downward, it proves that in the modern digital 'attention economy', shorter products successfully capture higher market demand.")

st.divider()

# ==========================================
# EXERCISE 2: IP Asset Depreciation
# ==========================================
st.subheader("Economic Exercise 2: IP Asset Depreciation")
st.write("Do older intellectual property assets lose their market value over time?")

if st.button("Map Residual IP Value by Year"):
    # Group by Year and get the average Popularity
    yearly_value = df.groupby('Year')['Popularity'].mean().reset_index()
    
    # Filter out weird outliers or years with almost no data (like pre-1970)
    yearly_value = yearly_value[yearly_value['Year'] > 1975]
    
    fig_econ2, ax_econ2 = plt.subplots(figsize=(12, 5))
    ax_econ2.plot(yearly_value['Year'], yearly_value['Popularity'], marker='o', color='goldenrod', linewidth=2)
    
    ax_econ2.set_xlabel("Asset Vintage (Release Year)", fontweight='bold')
    ax_econ2.set_ylabel("Average Residual Value (Current Popularity)", fontweight='bold')
    ax_econ2.set_title("Residual Market Value of Classic Alt-Rock IPs", fontsize=14, fontweight='bold')
    ax_econ2.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_econ2)
    
    st.success("💡 **Economic Interpretation:** This charts how Intellectual Property depreciates. Spikes in certain years show that cultural 'vintages' behave like fine wine, artificially holding their economic value far better than surrounding years.")

st.divider()

# ==========================================
# EXERCISE 3: Hedonic Pricing Model
# ==========================================
st.subheader("Economic Exercise 3: Hedonic Pricing Model")
st.write("Which specific product attribute does the consumer market value the most right now?")

if st.button("Calculate Hedonic Market Valuations"):
    # Define the "product attributes" we want to test
    features = ['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Liveness', 'Valence', 'Tempo']
    
    # Ensure all columns are numeric before calculating correlation
    numeric_df = df[features + ['Popularity']].apply(pd.to_numeric, errors='coerce').dropna()
    
    # Calculate the Pearson correlation of these features strictly against 'Popularity'
    correlations = numeric_df[features].corrwith(numeric_df['Popularity']).sort_values(ascending=False)
    
    fig_econ3, ax_econ3 = plt.subplots(figsize=(10, 5))
    
    # Create a bar chart showing positive vs negative correlations
    colors = ['green' if c > 0 else 'red' for c in correlations]
    correlations.plot(kind='bar', color=colors, ax=ax_econ3, edgecolor='black')
    
    ax_econ3.set_ylabel("Correlation with Market Demand", fontweight='bold')
    ax_econ3.set_title("Hedonic Valuation of Musical Attributes", fontsize=14, fontweight='bold')
    ax_econ3.axhline(0, color='black', linewidth=1)
    
    # Rotate x labels for readability
    plt.xticks(rotation=45, ha='right')
    
    st.pyplot(fig_econ3)
    
    st.warning("💡 **Economic Interpretation:** Green bars show features the market currently demands and rewards (Positive ROI). Red bars show features that actively destroy market value. A record label using this data would force their artists to maximize the top green attributes.")
