import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub as kh
import os
from sklearn.cluster import KMeans

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
st.subheader("Economic Exercise 10: The Attention Economy")
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

st.divider()

# ==========================================
# EXERCISE 11: Market Segmentation & Encoding
# ==========================================
st.subheader("Economic Exercise 11: Market Segmentation & Encoding")
st.write("Encoding track lengths into product categories to predict market demand and popularity over time.")

if st.button("Run Encode & Sort Analysis"):
    with st.spinner("Processing encoding and sorting..."):
        sorted_df = df.dropna(subset=['Duration', 'Popularity', 'Year']).sort_values('Year')
        sorted_df['Duration_Mins'] = sorted_df['Duration'] / 60000
        bins = [0, 3.5, 6, np.inf]
        labels = ['Radio Edit', 'Album Cut', 'Extended Mix']
        sorted_df['Format'] = pd.cut(sorted_df['Duration_Mins'], bins=bins, labels=labels)
        encoded_formats = pd.get_dummies(sorted_df['Format'], prefix='Format').astype(int)
        sorted_df = pd.concat([sorted_df, encoded_formats], axis=1)
        st.write("Notice the binary (0 and 1) columns created by the encoding method:")
        st.dataframe(sorted_df[['Year', 'Track', 'Duration_Mins', 'Format_Radio Edit', 'Format_Album Cut', 'Format_Extended Mix', 'Popularity']].head(15))
        yearly_trends = sorted_df.groupby(['Year', 'Format'])['Popularity'].mean().unstack()
        fig_econ11, ax_econ11 = plt.subplots(figsize=(12, 6))
        for col in yearly_trends.columns:
            trend = yearly_trends[col].dropna()
            ax_econ11.plot(trend.index, trend.values, marker='o', markersize=4, label=col, alpha=0.8)
        ax_econ11.set_xlabel("Release Year")
        ax_econ11.set_ylabel("Average Popularity")
        ax_econ11.legend()
        st.pyplot(fig_econ11)
        st.info("💡 **Economic Interpretation:** One-Hot Encoding helps us track how the market demand for different length formats has evolved.")

st.divider()

# ==========================================
# EXERCISE 12: Machine Learning Clustering
# ==========================================
st.subheader("Economic Exercise 12: Machine Learning Clustering")
st.write("Using KMeans to group tracks by audio characteristics (Danceability & Energy).")

if st.button("Run KMeans Clustering"):
    X = df[['Danceability', 'Energy']].dropna().values
    kmeans = KMeans(n_clusters=3, n_init=5, random_state=42).fit(X)
    
    fig_ml, ax_ml = plt.subplots(figsize=(8, 6))
    scatter = ax_ml.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='viridis', alpha=0.6)
    ax_ml.set_xlabel("Danceability")
    ax_ml.set_ylabel("Energy")
    ax_ml.set_title("KMeans Clustering of Musical Products")
    st.pyplot(fig_ml)

st.divider()

# ==========================================
# EXERCISE 13: Geographical Encoding
# ==========================================
st.subheader("Economic Exercise 13: Geographical Encoding")
st.write("How can we translate geographic regions into numbers for a computer?")

if st.button("Run Geo-Encoding Analysis"):
    # Target subset for demonstration
    sample_artists = ['Nirvana', 'Oasis', 'The Cure', 'U2']
    sample_df = df[df['Artist'].isin(sample_artists)].copy()
    mapping = {'Nirvana': 'USA', 'Oasis': 'UK', 'The Cure': 'UK', 'U2': 'Ireland'}
    sample_df['Country'] = sample_df['Artist'].map(mapping)
    encoded = pd.get_dummies(sample_df, columns=['Country'], prefix='Origin', dtype=int)
    st.write("Results of the One-Hot Encoding transformation:")
    cols_to_show = [c for c in encoded.columns if 'Origin_' in c]
    st.dataframe(encoded[['Artist', 'Track'] + cols_to_show].head(10))

