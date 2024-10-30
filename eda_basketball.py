import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Title for the Streamlit app
st.title('# EDA Basketball')

# Sidebar header for input parameters
st.sidebar.header("Input Parameters")
selected_year = st.sidebar.selectbox("Year", list(reversed(range(1950, 2020))))

"""
Parsing basketball players' info from https://www.basketball-reference.com
"""
@st.cache_data
def parse_data(year: str):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    parsed_df = pd.read_html(url, header=0)[0]
    parsed_df = parsed_df.drop(parsed_df[parsed_df['Age'] == 'Age'].index)  # Remove header rows within data
    parsed_df = parsed_df.fillna(0)
    parsed_df = parsed_df.drop(['Rk'], axis=1)  # Drop 'Rk' column
    # Convert datatype to work with age filter
    parsed_df['Age'] = pd.to_numeric(parsed_df['Age'], errors='coerce').fillna(0).astype(int)
    return parsed_df

# Load player statistics for the selected year
df_player_stat_dataset = parse_data(str(selected_year))

# Convert 'Tm' column to strings before sorting
sorted_dataset_by_team = sorted(df_player_stat_dataset['Tm'].astype(str).unique())

# Team filter
selected_team = st.sidebar.multiselect("Team", sorted_dataset_by_team, sorted_dataset_by_team)

# Position filter
player_positions = ['C', 'PF', 'SF', 'PG', 'SG']
selected_position = st.sidebar.multiselect("Position", player_positions, player_positions)

# Unique age values for the slider
unique_age_values = df_player_stat_dataset.Age.unique()
minValue, maxValue = int(min(unique_age_values)), int(max(unique_age_values))

# Age filter
selected_age = st.sidebar.slider("Age", minValue, maxValue, (minValue, maxValue), 1)
min_age, max_age = selected_age

# Filtered dataset based on selections
df_selected_dataset = df_player_stat_dataset[
    (df_player_stat_dataset['Tm'].isin(selected_team) &
     df_player_stat_dataset['Pos'].isin(selected_position) &
     df_player_stat_dataset['Age'].between(min_age, max_age))]

# Display dataframe
st.header('Display Player Stats of Selected Team(s)')
st.write(f'Data Dimension: Rows: {df_selected_dataset.shape[0]}, Columns: {df_selected_dataset.shape[1]}')
st.dataframe(df_selected_dataset)

# Function to download the dataset as CSV
def download_dataset(dataset):
    csv = dataset.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Strings to bytes conversions
    href_link = f'<a href="data:file/csv;base64,{b64}" download="player_stats.csv">Download CSV File</a>'
    return href_link

# Link for downloading the filtered dataset
st.markdown(download_dataset(df_selected_dataset), unsafe_allow_html=True)

# Inter-correlation heatmap button
if st.button("Inter-correlation Heatmap"):
    st.header("Inter-correlation Heatmap")

    # Generate correlation matrix and plot
    corr = df_selected_dataset.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    # Plot the inter-correlation heatmap
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        try:
            sns.heatmap(corr, mask=mask, vmax=1, square=True, annot=True, fmt=".2f")
        except ValueError:
            st.write("Error in generating heatmap.")
    st.pyplot(f)
