import streamlit as st
import kagglehub as kh
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot
import geopandas as gpd

st.set_page_config(page_title="Classic Alt Rock Analysis", layout="wide")

st.title("Classic Alt Rock Dataset Analysis")
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
st.dataframe(df.head(1000))

st.subheader("Dataset Dimensions")
st.write(f"This dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

#X 2. Extracted Artists
st.subheader("2. Extracted Artists")
# Extract a unique list of all artists in the dataset
artists_list = df['Artist'].unique().tolist()
artists_list = [str(artist) for artist in artists_list]
artists_list.sort(key=str.lower)

st.write(f"There are a total of {len(artists_list)} unique artists present in the initial database:")

# We will use Streamlit's built-in column layout to display the artists cleanly without any table headers!
num_cols = 5
cols = st.columns(num_cols)

for i, artist in enumerate(artists_list):
    # This automatically distributes the artist names left-to-right into the 5 columns
    cols[i % num_cols].write(f"- {artist}")

st.divider()

#X 3. Filtered Dataset
st.subheader("3. Filtered Dataset")
st.write("A newly created dataset strictly isolating the targeted bands:")

# The user specifically requested to shrink a NEW dataset down to only these targeted bands:
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

# Create a brand new dataframe (leaving the initial 'df' intact)
filtered_df = df[df['Artist'].isin(selected_bands)].reset_index(drop=True)

st.dataframe(filtered_df)
st.write(f"The beautifully filtered dataset now has exactly {filtered_df.shape[0]} rows.")

st.divider()

#X 4. Statistical Processing & Aggregation
st.subheader("4. Statistical Processing & Aggregation")
st.write("Using Pandas `.groupby()` to calculate the **sum** and **mean** of `Danceability` for each artist's tracks:")

# Group by Artist; compute statistics for each group, keeping it purely in Streamlit's display!
df_agg = filtered_df.groupby(['Artist']).agg({'Danceability': [sum, "mean"], # sum and mean of danceability per group
                                              'Tempo': "mean",               # mean of tempo
                                              'Track': 'count'})             # count of tracks for each group

st.dataframe(df_agg)

# Generate a visually distinct red Matplotlib button specific to this aggregated data
if st.button("Show Aggregated Data Graph"):
    with st.spinner("Rendering Matplotlib chart..."):
        # Plotting the Average Tempo per Artist from the aggregated table
        artist_names = df_agg.index
        avg_tempo = df_agg[('Tempo', 'mean')]
        
        # Create a Matplotlib figure
        fig3, ax3 = matplotlib.pyplot.subplots(figsize=(14, 6))
        
        import matplotlib.colors as mcolors
        
        # Generate a list of all distinct named colors natively from matplotlib (inspired by your code!)
        all_colors = list(mcolors.cnames.keys())
        
        # Plot a beautiful multi-colored bar chart pulling from the exact length required
        ax3.bar(artist_names, avg_tempo, color=all_colors[:len(artist_names)], edgecolor="black")
        
        # Format the axes and labels
        ax3.set_xticks(range(len(artist_names)))
        ax3.set_xticklabels(artist_names, rotation=90, fontsize=9)
        ax3.set_xlabel("Artist", fontsize=12, fontweight='bold')
        ax3.set_ylabel("Average Tempo (BPM)", fontsize=12, fontweight='bold')
        ax3.set_title("Average Track Tempo per Artist", fontsize=16)
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Render the specific Matplotlib chart
        st.pyplot(fig3)

st.divider()

#X 5. Merge / Join Datasets
st.subheader("5. Processing Datasets with Merge / Join")
st.write("We create a secondary standalone dataset containing the 'Country of Origin' for each artist, and use Pandas `pd.merge()` to mathematically join them:")

# Create a small, secondary complementary dataset
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

# Display the secondary mini-dataset
st.write("Here is the secondary 'Country Mapping' dataset we created:")
st.dataframe(country_mapping)

# Execute the Merge (Left Join) combining our 800 tracks with the country mapping
st.write("Here is the final dataset after we apply `pd.merge(filtered_df, country_mapping, on='Artist', how='left')`:")
merged_df = pd.merge(filtered_df, country_mapping, on='Artist', how='left')

# Output the result
st.dataframe(merged_df)

# Create a Pie Chart based on the exact syntax logic from the User's inspired code example!
st.write("Visualizing the Distribution of Tracks by Country of Origin using a Pie Chart:")

if st.button("Show Country Pie Chart"):
    with st.spinner("Generating pie chart..."):
        # We group by 'Country' and calculate the size (track count) of each region dynamically
        country_distribution = merged_df.groupby('Country')['Track'].count()

        # Create the matplotlib figure using their exact figsize parameters
        fig4, ax4 = matplotlib.pyplot.subplots(figsize=(8, 8))
        
        if not country_distribution.empty:
            # Replicating the exact logic: pie(data, labels=..., autopct='%1.1f%%', startangle=90)
            ax4.pie(country_distribution, labels=country_distribution.index, autopct='%1.1f%%', startangle=90)
            
            ax4.set_title('Track Distribution by Country of Origin', fontsize=16, fontweight='bold')
            ax4.axis('equal') # Equal aspect ratio ensures that pie is drawn as a flawless circle.
            
            # Show the plot
            st.pyplot(fig4)

st.divider()

#X 6. Matplotlib Graphical Representation
st.subheader("6. Graphical Representation (`matplotlib`)")
st.write("Using the full `matplotlib` package to create a bar chart of the top 10 most popular tracks:")

# Injecting HTML/CSS to force the button's background to be exactly #FF0000
st.markdown("""
<style>
div.stButton > button {
    background-color: #FF0000 !important;
    color: white !important;
    border-color: #CC0000 !important;
}
</style>""", unsafe_allow_html=True)

if st.button("Show Matplotlib Chart"):
    with st.spinner("Rendering bar chart..."):
        # Sort by popularity to isolate the top 10 tracks
        top_10 = filtered_df.sort_values(by="Popularity", ascending=False).head(10)

        # Create a beautiful Matplotlib figure and axis
        fig2, ax2 = matplotlib.pyplot.subplots(figsize=(12, 6))

        import matplotlib.colors as mcolors

        # Generate a distinct array list of native named colors
        all_colors_top = list(mcolors.cnames.keys())

        # Plot a multi-colored bar chart using the named color list
        ax2.bar(top_10["Track"], top_10["Popularity"], color=all_colors_top[:len(top_10)], edgecolor="black")

        # Rotate the track names on the X-axis so they don't overlap
        ax2.set_xticks(range(len(top_10)))
        ax2.set_xticklabels(top_10["Track"], rotation=45, ha='right')

        # Add official academic labels and titles
        ax2.set_xlabel("Track Name", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Spotify Popularity Score", fontsize=12, fontweight='bold')
        ax2.set_title("Top 10 Most Popular Alt-Rock Tracks", fontsize=16)

        # Adding horizontal grid lines to match typical visual standards
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        # Send the static Matplotlib figure to Streamlit for display
        st.pyplot(fig2)

st.divider()

#X 7. Geographic Polygon Mapping (GeoPandas)
st.subheader("7. Geographic Polygons (GeoPandas)")
st.write("A simpler, computational illustration using GeoPandas purely to isolate the mathematical boundary vectors (polygons) of the USA, UK, and Ireland:")

if st.button("Extract Geographic Boundaries"):
    with st.spinner("Extracting and drawing polygons..."):
        
        # Load the world geometry map
        world = gpd.read_file("https://raw.githubusercontent.com/python-visualization/folium/main/examples/data/world-countries.json")
        
        # Standardize names
        world['name'] = world['name'].replace({
            'United States of America': 'USA',
            'United Kingdom': 'UK'
        })
        
        # Isolate the precise coordinates for our 3 specific dataset countries
        alt_rock_countries = world[world['name'].isin(['USA', 'UK', 'Ireland'])]
        
        fig5, ax5 = matplotlib.pyplot.subplots(figsize=(10, 6))
        
        # Plot the entire world faintly in the background
        world.plot(ax=ax5, color='#e9ecef', edgecolor='white')
        
        # Plot our 3 specific Alt-Rock country polygons dynamically in bright #FF0000 Red
        alt_rock_countries.plot(ax=ax5, color='#FF0000', edgecolor='black')
        
        ax5.set_title("Literal Geometry Arrays for Selected Countries", fontsize=14, fontweight='bold')
        ax5.axis("off")
        
        st.pyplot(fig5)

st.divider()

#X 8. KMeans Clustering (Machine Learning)
st.subheader("8. Machine Learning: KMeans Clustering")
st.write("Using `sklearn.cluster.KMeans` to group our Classic Alt-Rock tracks into 3 distinct mathematical clusters based on their `Danceability` and `Energy` levels:")

if st.button("Generate KMeans Clustering Diagrams"):
    with st.spinner("Running KMeans algorithm..."):
        from sklearn.cluster import KMeans

        # 1. Prepare the Data array 'X' just like the example
        # We will extract 2 numerical features from our filtered dataset to cluster!
        cluster_data = filtered_df[['Danceability', 'Energy']].dropna()
        X = cluster_data.values

        # 2. Initialize and fit the KMeans algorithm (from the provided example!)
        kmeans = KMeans(n_clusters=3, n_init=5, random_state=42)
        kmeans.fit(X)

        # 3. Print the Centers and Labels (to Streamlit instead of terminal print())
        st.write("**Calculated Cluster Centers:**")
        st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=['Danceability Center', 'Energy Center']))
        
        # 4. Generate the 3 beautiful scatter plots based strictly on the user's example format
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**1. Raw Data Scatter (`f1`)**")
            f1, ax1 = matplotlib.pyplot.subplots(figsize=(5,5))
            ax1.scatter(X[:,0], X[:,1], color='gray', alpha=0.6)
            ax1.set_xlabel("Danceability", fontsize=10)
            ax1.set_ylabel("Energy", fontsize=10)
            st.pyplot(f1)
            
        with col2:
            st.write("**2. Clustered Data (`f2`)**")
            f2, ax2 = matplotlib.pyplot.subplots(figsize=(5,5))
            ax2.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow', alpha=0.6)
            ax2.set_xlabel("Danceability", fontsize=10)
            ax2.set_ylabel("Energy", fontsize=10)
            st.pyplot(f2)
            
        with col3:
            st.write("**3. Isolated Cluster Centers (`f3`)**")
            f3, ax3 = matplotlib.pyplot.subplots(figsize=(5,5))
            ax3.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='black', s=150, marker='X')
            ax3.set_xlabel("Danceability Center", fontsize=10)
            ax3.set_ylabel("Energy Center", fontsize=10)
            ax3.set_xlim(ax2.get_xlim())
            ax3.set_ylim(ax2.get_ylim())
            st.pyplot(f3)

st.divider()

#X 9. Data Cleaning (Extreme Outliers)
st.subheader("9. Data Cleaning (Handling Extreme Values)")
st.write("Using the mathematical Interquartile Range (IQR) limits to actively detect and cleanly cap extreme statistical anomalies (outliers) hiding inside the dataset:")

if st.button("Detect & Clean Extreme Values"):
    with st.spinner("Calculating Interquartile Bounds..."):
        
        # Calculate IQR strictly for the 'Popularity' column
        Q1 = filtered_df['Popularity'].quantile(0.25)
        Q3 = filtered_df['Popularity'].quantile(0.75)
        IQR_value = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR_value
        upper_bound = Q3 + 1.5 * IQR_value
        
        # 1. Identify raw outliers mathematically exceeding the 1.5 mathematical limits
        outliers = filtered_df[(filtered_df['Popularity'] < lower_bound) | (filtered_df['Popularity'] > upper_bound)]
        
        st.write(f"**Step 1. Detection:** My algorithm definitively identified **{len(outliers)} extreme outliers** in the song `Popularity` metric that fell significantly outside normal bounds!")
        st.dataframe(outliers)
        
        st.write("**Step 2. Data Capping (Winsorization):** We mathematically compress those extreme outliers tightly back into the mathematical limits to clean the predictive model!")
        
        # 2. Clean the dataset strictly by capping the upper/lower extreme values
        cleaned_df = filtered_df.copy()
        
        cleaned_df.loc[cleaned_df['Popularity'] > upper_bound, 'Popularity'] = upper_bound
        cleaned_df.loc[cleaned_df['Popularity'] < lower_bound, 'Popularity'] = lower_bound
        
        # Verify cleaning completely eliminated the outliers
        new_outliers = cleaned_df[(cleaned_df['Popularity'] < lower_bound) | (cleaned_df['Popularity'] > upper_bound)]
        
        # Output the Success metrics
        st.success(f"**Success!** The dataset has been rigorously cleansed. There are now {len(new_outliers)} extreme values remaining.")
        st.dataframe(cleaned_df.describe())
