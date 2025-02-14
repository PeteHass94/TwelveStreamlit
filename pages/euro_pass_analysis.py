# Library imports
import streamlit as st
import pandas as pd
import argparse
import tiktoken
import os
import pickle

from utils.utils import normalize_text

from classes.data_source import EuroPasses 

from utils.page_components import add_common_page_elements

# UI Setup
sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.title("Euro Pass Analysis")
st.divider()


euroPasses = EuroPasses()

with st.expander("Raw Euro Passes Dataframe"):
    st.write(euroPasses.df)

# # Load datasets
# @st.cache_data
# def load_data():
#     with open('../Data/EuroEvents.pkl', 'rb') as file:
#         euro_events = pickle.load(file)
#     with open('./Data/EuroMatches.pkl', 'rb') as file:
#         euro_matches = pickle.load(file)
#     with open('./Data/EuroEventsShotSequence.pkl', 'rb') as file:
#         euro_events_shot_sequence = pickle.load(file)
#     return euro_events, euro_matches, euro_events_shot_sequence

# Load the data
# euro_events, euro_matches, euro_events_shot_sequence = load_data()



# Player selection
# players = euro_events['player_name'].unique()
# selected_player = st.selectbox("Select a player", players)

# Filter relevant data
# player_pass_data = euro_events[(euro_events['player_name'] == selected_player) & 
                            #    ((euro_events['pass_shot_assist'] == True) | (euro_events['pass_goal_assist'] == True))]

# # Display statistics
# st.write(f"### Passing Analysis for {selected_player}")
# shot_assists = player_pass_data[player_pass_data['pass_shot_assist'] == True]
# goal_assists = player_pass_data[player_pass_data['pass_goal_assist'] == True]

# st.write(f"**Shot Assists:** {len(shot_assists)}")
# st.write(f"**Goal Assists:** {len(goal_assists)}")

# # Display passing details
# st.dataframe(player_pass_data[['match_id', 'team_name', 'pass_recipient', 'pass_shot_assist', 'pass_goal_assist']])