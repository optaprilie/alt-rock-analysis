import streamlit as st
import kagglehub as kh
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot
import geopandas as gpd
import statsmodels.api as sm

#X 1. Page Configuration
st.set_page_config(page_title="Classic Alt Rock Analysis", layout="wide")

st.title("Classic Alt Rock Dataset analysis")
st.write("Streamlit interface for dataset analysis.")

@st.cache_data
def load_data():
    path = kh.dataset_download("thebumpkin/800-classic-alt-rock-tracks-with-spotify-data")
    csv_file_path = os.path.join(path, "ClassicAltRock.csv")
    df = pd.read_csv(csv_file_path)
    return df

with st.spinner("Loading dataset..."):
    df = load_data()

st.subheader("1. Raw Dataset Preview")
st.dataframe(df.head(800))

st.subheader("Dataset Dimensions")
st.write(f"This dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

#X 2. Extracted Artists
st.subheader("2. Extracted Artists")

sort_option = st.radio("Sort Artists By:", ["Alphabetical (A-Z)", "Average Popularity (High to Low)"]
, horizontal=True)

if sort_option == "Average Popularity (High to Low)":
    artist_pop = df.groupby('Artist')['Popularity'].mean().sort_values(ascending=False)
    artists_list = artist_pop.index.tolist()
    artists_list = [str(artist) for artist in artists_list]
else:
    artists_list = df['Artist'].unique().tolist()
    artists_list = [str(artist) for artist in artists_list]
    artists_list.sort(key=str.lower)

st.write(f"There are a total of {len(artists_list)} unique artists present in the initial database:")

num_cols = 5
cols = st.columns(num_cols)

for i, artist in enumerate(artists_list):
    cols[i % num_cols].write(f"- {artist}")

st.divider()

#X 3. Filtered Dataset
st.subheader("3. Filtered Dataset")
st.write("A newly created dataset strictly isolating the targeted bands:")

selected_bands = [
    "3 Doors Down", "Alice In Chains", "Blur", "Counting Crows", "Dead Kennedys", 
    "Deftones", "Depeche Mode", "Disturbed", "Elvis Costello", "Everclear", 
    "Foo Fighters", "Green Day", "Incubus", "Joy Division", "King Crimson", "Korn", 
    "Linkin Park", "Muse", "my bloody valentine", "My Chemical Romance", "New Order", 
    "Nine Inch Nails", "Nirvana", "Oasis", "Papa Roach", "Pearl Jam", "Pet Shop Boys", 
    "Red Hot Chili Peppers", "Rob Zombie", "Sex Pistols", "Soundgarden", 
    "System Of A Down", "Talking Heads", "The Cars", 
    "The Clash", "The Cure", "The Smashing Pumpkins", "The Smiths", "TOOL", "Weezer"
]

filtered_df = df[df['Artist'].isin(selected_bands)].reset_index(drop=True)

st.dataframe(filtered_df)
st.write(f"The beautifully filtered dataset now has exactly {filtered_df.shape[0]} rows.")

st.write(f"**The {len(selected_bands)} Officially Filtered Artists:**")

sort_option_filtered = st.radio("Sort Filtered Artists By:", ["Alphabetical (A-Z)", "Average Popularity (High to Low)"], horizontal=True, key="sort_filtered")

if sort_option_filtered == "Average Popularity (High to Low)":
    artist_pop_f = filtered_df.groupby('Artist')['Popularity'].mean().sort_values(ascending=False)
    display_bands = artist_pop_f.index.tolist()
else:
    display_bands = sorted(selected_bands, key=str.lower)

num_cols_f = 5
cols_f = st.columns(num_cols_f)

for i, artist in enumerate(display_bands):
    cols_f[i % num_cols_f].write(f"- {artist}")

st.write("---")
st.write("**March 2026 Estimated Artist Revenue Comparison**")

if st.button("Show Revenue Chart"):
    with st.spinner("Generating revenue chart..."):
        fig_rev, ax_rev = matplotlib.pyplot.subplots(figsize=(8, 5))
        bands = ['Red Hot Chili Peppers', 'Sex Pistols']
        
        rhcp_revenue = 45844488 * 0.005
        pistols_revenue = 1258867 * 0.005
        revenues = [rhcp_revenue, pistols_revenue]
        
        bar_colors = ['red', 'black']
        
        ax_rev.bar(bands, revenues, color=bar_colors, edgecolor='black')
        
        ax_rev.set_ylabel("Revenue ($)", fontweight='bold')
        ax_rev.set_title("March 2026 Estimated Spotify Revenue", fontsize=14, fontweight='bold')
        ax_rev.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, v in enumerate(revenues):
            ax_rev.text(i, v + 5000, f"${v:,.2f}", ha='center', fontweight='bold', fontsize=11)
            
        st.pyplot(fig_rev)

st.write("---")
st.write("**Average Popularity Ranking of the 40 Selected Bands**")

if st.button("Show Popularity Chart for 40 Bands"):
    with st.spinner("Generating popularity comparison chart..."):
        band_popularity = filtered_df.groupby('Artist')['Popularity'].mean().sort_values(ascending=False)
        
        fig_pop, ax_pop = matplotlib.pyplot.subplots(figsize=(14, 6))
        
        import matplotlib.cm as cm
        bar_colors = cm.viridis(np.linspace(0, 1, len(band_popularity)))
        
        ax_pop.bar(band_popularity.index, band_popularity.values, color=bar_colors, edgecolor='black')
        
        ax_pop.set_xticks(range(len(band_popularity)))
        ax_pop.set_xticklabels(band_popularity.index, rotation=90, fontsize=9)
        ax_pop.set_ylabel("Average Spotify Popularity", fontweight='bold')
        ax_pop.set_title("Average Spotify Popularity of the 40 Filtered Artists", fontsize=16, fontweight='bold')
        ax_pop.grid(axis='y', linestyle='--', alpha=0.6)
        
        st.pyplot(fig_pop)

st.divider()

#X 4. Statistical Processing & Aggregation
st.subheader("4. Statistical Processing & Aggregation")
st.write("Using Pandas `.groupby()` to calculate the **sum** and **mean** of `Danceability` for each artist's tracks:")


df_agg = filtered_df.groupby(['Artist']).agg({'Danceability': [sum, "mean"], 
                                              'Tempo': "mean",               
                                              'Track': 'count'})             

st.dataframe(df_agg)

if st.button("Show Aggregated Data Graph"):
    with st.spinner("Rendering Matplotlib chart..."):
        artist_names = df_agg.index
        avg_tempo = df_agg[('Tempo', 'mean')]
        
        fig3, ax3 = matplotlib.pyplot.subplots(figsize=(14, 6))
        
        import matplotlib.colors as mcolors
        
        all_colors = list(mcolors.cnames.keys())
        
        ax3.bar(artist_names, avg_tempo, color=all_colors[:len(artist_names)], edgecolor="black")
        
        ax3.set_xticks(range(len(artist_names)))
        ax3.set_xticklabels(artist_names, rotation=90, fontsize=9)
        ax3.set_xlabel("Artist", fontsize=12, fontweight='bold')
        ax3.set_ylabel("Average Tempo (BPM)", fontsize=12, fontweight='bold')
        ax3.set_title("Average Track Tempo per Artist", fontsize=16)
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        
        st.pyplot(fig3)

st.write("---")
st.write("**Correlation Analysis: Danceability, Tempo, and Popularity**")

if st.button("Show Correlation Matrix"):
    with st.spinner("Calculating Pearson correlations..."):
        corr_data = filtered_df[['Danceability', 'Tempo', 'Popularity']].dropna()
        
        corr_matrix = corr_data.corr()
        
        st.write("Raw Mathematical Matrix (Color Coded):")
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1).format("{:.3f}"))
        
        fig_corr, ax_corr = matplotlib.pyplot.subplots(figsize=(6, 5))
        
        cax = ax_corr.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        fig_corr.colorbar(cax, shrink=0.8)
        
        labels = ['Danceability', 'Tempo', 'Popularity']
        ax_corr.set_xticks(range(len(labels)))
        ax_corr.set_yticks(range(len(labels)))
        ax_corr.set_xticklabels(labels, fontsize=10, fontweight='bold')
        ax_corr.set_yticklabels(labels, fontsize=10, fontweight='bold')
        ax_corr.xaxis.set_ticks_position('bottom')
        
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = corr_matrix.iloc[i, j]
                text_color = "white" if abs(val) > 0.5 else "black"
                ax_corr.text(j, i, f"{val:.2f}",
                             ha="center", va="center", color=text_color, fontweight='bold', fontsize=12)
                
        ax_corr.set_title("Pearson Feature Correlation Heatmap", pad=20, fontsize=14, fontweight='bold')
        
        st.pyplot(fig_corr)
        
        st.info("💡 **Statistical Insight:** A correlation value of exactly `1.00` is perfect (e.g., Tempo vs Tempo). If Popularity strongly scales with Danceability, the number will be firmly positive. If it is close to `0.00`, there is zero linear mathematical relationship between them in this dataset.")

st.divider()

#X 5. Merge / Join Datasets
st.subheader("5. Processing Datasets with Merge / Join")
st.write("We create a secondary standalone dataset containing the 'Country of Origin' for each artist, and use Pandas `pd.merge()` to mathematically join them:")

country_mapping = pd.DataFrame({
    'Artist': selected_bands,
    'Country': [
        "USA", "USA", "UK", "USA", "USA", 
        "USA", "UK", "USA", "UK", "USA", 
        "USA", "USA", "USA", "UK", "UK", "USA", 
        "USA", "UK", "Ireland", "USA", "UK", 
        "USA", "USA", "UK", "USA", "USA", "UK", 
        "USA", "USA", "UK", "USA", 
        "USA", "USA", "USA", 
        "UK", "UK", "USA", "UK", "USA", "USA"
    ]
})

st.dataframe(country_mapping)

merged_df = pd.merge(filtered_df, country_mapping, on='Artist', how='left')

st.dataframe(merged_df)

st.write("Visualizing the Distribution of Tracks by Country of Origin using a Pie Chart:")

if st.button("Show Country Pie Chart"):
    with st.spinner("Generating pie chart..."):
        country_distribution = merged_df.groupby('Country')['Track'].count()

        fig4, ax4 = matplotlib.pyplot.subplots(figsize=(8, 8))
        
        if not country_distribution.empty:
            ax4.pie(country_distribution, labels=country_distribution.index, autopct='%1.1f%%', startangle=90)
            
            ax4.set_title('Track Distribution by Country of Origin', fontsize=16, fontweight='bold')
            ax4.axis('equal') 
            st.pyplot(fig4)

st.write("Visualizing the mapped regions geographically using **GeoPandas**:")

if st.button("Extract Geographic Boundaries"):
    with st.spinner("Extracting and drawing polygons..."):
        
        world = gpd.read_file("https://raw.githubusercontent.com/python-visualization/folium/main/examples/data/world-countries.json")
        
        world['name'] = world['name'].replace({
            'United States of America': 'USA',
            'United Kingdom': 'UK'
        })
        
        alt_rock_countries = world[world['name'].isin(['USA', 'UK', 'Ireland'])]
        
        fig5, ax5 = matplotlib.pyplot.subplots(figsize=(10, 6))
        
        world.plot(ax=ax5, color='#e9ecef', edgecolor='white')
        
        world[world['name'] == 'USA'].plot(ax=ax5, color='red', edgecolor='black')
        world[world['name'] == 'UK'].plot(ax=ax5, color='blue', edgecolor='black')
        world[world['name'] == 'Ireland'].plot(ax=ax5, color='green', edgecolor='black')
        
        ax5.set_title("Geographic Origins of Classic Alt-Rock Artists", fontsize=14, fontweight='bold')
        ax5.axis("off")

        import matplotlib.patches as mpatches
        usa_patch = mpatches.Patch(color='red', label='USA')
        uk_patch = mpatches.Patch(color='blue', label='UK')
        ireland_patch = mpatches.Patch(color='green', label='Ireland')
        ax5.legend(handles=[usa_patch, uk_patch, ireland_patch], loc='lower left', title="Artist Origins")
        
        st.pyplot(fig5)

st.divider()

#X 6. Matplotlib Graphical Representation
st.subheader("6. Graphical Representation (`matplotlib`)")
st.write("Using the full `matplotlib` package to create a bar chart of the top 10 most popular tracks:")

if st.button("Show Matplotlib Chart"):
    with st.spinner("Rendering bar chart..."):
        top_10 = filtered_df.sort_values(by="Popularity", ascending=False).head(10)

        fig2, ax2 = matplotlib.pyplot.subplots(figsize=(12, 6))

        import matplotlib.colors as mcolors

        all_colors_top = list(mcolors.cnames.keys())
        ax2.bar(top_10["Track"], top_10["Popularity"], color=all_colors_top[:len(top_10)], edgecolor="black")

        ax2.set_xticks(range(len(top_10)))
        ax2.set_xticklabels(top_10["Track"], rotation=45, ha='right')

        ax2.set_xlabel("Track Name", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Spotify Popularity Score", fontsize=12, fontweight='bold')
        ax2.set_title("Top 10 Most Popular Alt-Rock Tracks", fontsize=16)

        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        st.pyplot(fig2)

st.divider()

#X 7. Statistical Modeling: Multiple Regression (`statsmodels`)
st.subheader("7. Statistical Modeling: Multiple Regression (`statsmodels`)")
st.write("Using the mathematical `statsmodels.api` package to analyze how significantly `Danceability` and `Energy` statistically predict a track's ultimate `Popularity` on Spotify:")

if st.button("Run Multiple Regression Analysis"):
    with st.spinner("Running OLS mathematical regression..."):
        
        regression_df = filtered_df[['Popularity', 'Danceability', 'Energy']].dropna()
        
        Y = regression_df['Popularity']
        X = regression_df[['Danceability', 'Energy']]
        X = sm.add_constant(X)
        
        model = sm.OLS(Y, X).fit()
        
        st.write("**Comprehensive Statsmodels Regression Summary:**")
        st.text(model.summary().as_text())
        
        st.write("**Actual vs Predicted Popularity Scatter:**")
        predictions = model.predict(X)
        
        fig_reg, ax_reg = matplotlib.pyplot.subplots(figsize=(10, 6))
        ax_reg.scatter(Y, predictions, color='magenta', alpha=0.6, edgecolor='black')
        
        min_val = min(Y.min(), predictions.min())
        max_val = max(Y.max(), predictions.max())
        ax_reg.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2)
        
        ax_reg.set_xlabel("Actual Algorithm Popularity", fontweight='bold')
        ax_reg.set_ylabel("Predicted Popularity (from audio features)", fontweight='bold')
        ax_reg.set_title("OLS Multiple Regression Accuracy", fontsize=15, fontweight='bold')
        ax_reg.grid(linestyle='--', alpha=0.5)
        
        st.pyplot(fig_reg)

st.divider()

#X 8. Advanced Encoding: Track Length & Popularity Trends
st.subheader("8. Advanced Encoding: Track Length & Popularity Trends")
st.write("Encoding track lengths into categorical 'Formats' and visualizing popularity trends sorted by Year.")

if st.button("Run Format Analysis"):
    with st.spinner("Processing encoding and sorting by Year..."):
        # Select exactly 30 tracks for the analysis
        # 1. The 10 shortest tracks in the collection
        shortest_10 = merged_df.sort_values('Duration').head(10)
        # 2. The 10 longest tracks in the collection
        longest_10 = merged_df.sort_values('Duration', ascending=False).head(10)
        # 3. The 10 tracks closest to the median duration (The Middle)
        sorted_all = merged_df.sort_values('Duration')
        mid_idx = len(sorted_all) // 2
        middle_10 = sorted_all.iloc[mid_idx-5 : mid_idx+5]

        # Combine these into a single representative 30-track dataframe
        adv_df = pd.concat([shortest_10, longest_10, middle_10]).drop_duplicates()
        adv_df = adv_df.dropna(subset=['Duration', 'Popularity', 'Year']).sort_values('Year')

        adv_df['Duration_Mins'] = adv_df['Duration'] / 60000

        bins = [0, 3.5, 6, np.inf]
        labels = ['Radio Edit', 'Album Cut', 'Extended Mix']
        adv_df['Format'] = pd.cut(adv_df['Duration_Mins'], bins=bins, labels=labels)
        format_encoded = pd.get_dummies(adv_df['Format'], prefix='Type', dtype=int)
        adv_df = pd.concat([adv_df, format_encoded], axis=1)
        st.write("### The One-Hot Encoded Dataset (Sorted by Year)")
        st.dataframe(adv_df[['Year', 'Artist', 'Track', 'Duration_Mins', 'Type_Radio Edit', 'Type_Album Cut', 'Type_Extended Mix', 'Popularity']].head(15))
        yearly_pop = adv_df.groupby(['Year', 'Format'], observed=True)['Popularity'].mean().unstack()
        fig_final, ax_final = matplotlib.pyplot.subplots(figsize=(12, 6))
        for format_type in yearly_pop.columns:
            data = yearly_pop[format_type].dropna()
            ax_final.plot(data.index, data.values, marker='o', markersize=4, label=format_type, alpha=0.8)
        ax_final.set_xlabel("Release Year")
        ax_final.set_ylabel("Average Popularity")
        ax_final.set_title("Historical Popularity Trends by Encoded Track Format")
        ax_final.legend()
        st.pyplot(fig_final)
        st.info("**Final Context:** This encoding exercise proves how track format and length influences consumer demand across history.")


