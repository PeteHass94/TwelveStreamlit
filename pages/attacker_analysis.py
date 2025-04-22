# Library imports
import streamlit as st
import pandas as pd
import argparse
import tiktoken
import os

from classes.data_source import Season2018 
from classes.visual import ShotMap, ThreatMap, AttackerRadarAndDistPlot
from classes.chat import AttackingChat
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
st.text("Showing only Players with 1 or more goals from open play.")

player_radar1 = AttackerRadarAndDistPlot(playerStats_England)
player_radar1.display_player_radar()

player_radar1.get_similar_players()

st.title("Display Player Stats - Europe")
st.text("Showing only Players with 1 or more goals from open play.")

player_radar2 = AttackerRadarAndDistPlot(playerStats_Europe)
player_radar2.display_player_radar()

player_radar2.get_similar_players()
