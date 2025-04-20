# Library imports
import streamlit as st
import pandas as pd
import argparse
import tiktoken
import os

from classes.data_source import Season2018 
from classes.visual import ShotMap, ThreatMap, AttackerRadarAndDistPlot
from classes.chat import PassChat
from classes.description import PassDescription

from utils.page_components import add_common_page_elements

from utils.utils import (
    create_chat
)



# UI Setup
sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.title("Attacker Analysis: xG and xT")
st.subheader(f"Attacker analysis during the season 2018/19")

url = "https://figshare.com/collections/Soccer_match_event_dataset/4415000/2"
st.markdown("Data Sourced from WyScout [https://figshare.com/collections/Soccer_match_event_dataset/4415000/2](%s)" % url)
# st.markdown("check out this [link](%s)" % url)

seasonData = Season2018()

shotData_England = seasonData.getEnglandShotData()

possessionChain_England = seasonData.getEnglandPossessionChains()

possessionChain_Europe = seasonData.getEuropePossessionChains()

playerStats_England = seasonData.getEnglandPlayerStats()

playerStats_Europe = seasonData.getEuropePlayerStats()

with st.expander("2018 Season England ShotData Dataframe"):
    st.write(shotData_England)

with st.expander("2018 Season England Possession Chains Dataframe"):
    st.write(possessionChain_England)
    
with st.expander("2018 Season Europe Possession Chains Dataframe"):
    st.write(possessionChain_Europe)
    
with st.expander("2018 Season England Player Stats Dataframe"):
    st.write(playerStats_England)

with st.expander("2018 Season Europe Player Stats Dataframe"):
    st.write(playerStats_Europe)
    
st.divider()
st.title("Expected Goals (xG) Visualisation")


# Create shot map instance
shot_map = ShotMap(shotData_England)

# Display the filtered shot map
shot_map.display_shot_map()


st.divider()
st.title("Expected Threat (xT) Visualisation")

# Create shot map instance
threat_map = ThreatMap(possessionChain_England)

# Display the filtered shot map
threat_map.display_threat_map()

st.divider()
st.title("Display Player Stats - England")

player_radar = AttackerRadarAndDistPlot(playerStats_England)
player_radar.display_player_radar()




# # Load Data
# DATA_DIR = "data/2018_season"

# events_files = ["events_England_part1.json", "events_England_part2.json"]
# matches_file = "matches_England.json"
# players_file = "players.json"
# teams_file = "teams.json"

# # Load event data
# events = []
# for file in events_files:
#     with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
#         events.extend(json.load(f))

# # Load other data
# matches = load_data(os.path.join(DATA_DIR, matches_file))
# players = load_data(os.path.join(DATA_DIR, players_file))
# teams = load_data(os.path.join(DATA_DIR, teams_file))

# # Convert event data to DataFrame
# events_df = pd.DataFrame(events)

# # Calculate xG
# st.subheader("Expected Goals (xG) Calculation")
# xg_data = calculate_xG(events_df, players)
# st.dataframe(xg_data.head())

# # Calculate xT
# st.subheader("Expected Threat (xT) Calculation")
# xT_data = calculate_xT(events_df)
# st.dataframe(xT_data.head())

# # Calculate xCarry
# st.subheader("Expected Carry Value (xCarry) Calculation")
# xCarry_data = calculate_xCarry(events_df, xT_data, xg_data, players)
# st.dataframe(xCarry_data.head())

# # Train Model on English Data
# st.subheader("Training Model on English League Data")
# # Assume a training function exists
# model = train_xCarry_model(xCarry_data)
# st.write("Model training completed.")

# # Test Model on European Data
# st.subheader("Testing Model on European Leagues")
# EUROPEAN_LEAGUES = ["France", "Germany", "Italy", "Spain"]
# european_data = load_european_data(EUROPEAN_LEAGUES)
# model_results = model.predict(european_data)
# st.dataframe(model_results.head())

# # Player Analysis
# st.subheader("Player Performance Analysis")
# selected_player = st.selectbox("Select a player to analyze:", model_results["player_name"].unique())
# player_data = model_results[model_results["player_name"] == selected_player]
# st.dataframe(player_data)
# # plot_player_analysis(player_data)

# st.write("### Conclusion")
# st.write("This analysis helps identify attackers who create and capitalize on scoring opportunities through carrying, positioning, and passing contributions.")
