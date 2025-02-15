# Library imports
import streamlit as st
import pandas as pd
import argparse
import tiktoken
import os
import pickle

from utils.utils import normalize_text

from classes.data_source import EuroPasses 
from classes.visual import PitchPlot

from utils.page_components import add_common_page_elements

# UI Setup
sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.title("Euro Pass Analysis")
st.divider()


euro_passes = EuroPasses()

with st.expander("Raw Euro Passes Dataframe"):
    st.write(euro_passes.df)

# Team selection
teams = euro_passes.df['team'].unique()
selected_team = st.sidebar.selectbox("Select a country", teams)

# Filter players based on selected team
team_players = euro_passes.df[euro_passes.df['team'] == selected_team]
players = team_players['player_name'].unique() #.apply(lambda row: row['player_nickname'] if pd.notna(row['player_nickname']) else row['player'], axis=1).unique()
selected_player = st.sidebar.selectbox("Select a player", players)

# st.text(type(euro_passes.df)) 

# Filter relevant data
player_pass_data = euro_passes.get_player_pass_data(selected_player)
player_pass_chance_data = player_pass_data[(player_pass_data['pass_shot_assist'] == True) | (player_pass_data['pass_goal_assist'] == True)]

st.header(f"{selected_player}'s Data")

jerseyNumber, totalPasses, totalPassesComplete, totalShotAssists, totalGoalAssists, totalxA = euro_passes.get_player_metrics(player_pass_data)


with st.expander(f"{selected_player}'s Euro Passes Dataframe ({totalPasses})"):
    st.write(player_pass_data)

with st.expander(f"{selected_player}'s Euro Completed Passes Dataframe ({totalPassesComplete})"):
    st.write(player_pass_data[pd.isna(player_pass_data['pass_outcome'])])

if (totalShotAssists + totalGoalAssists) > 0:
    with st.expander(f"{selected_player}'s Euro Shot and Goal Assisted Passes Dataframe ({totalShotAssists + totalGoalAssists}, xA: {totalxA:.2f})"):
        st.write(player_pass_chance_data)


st.text(f"Total Passes: {totalPasses}, Completed Passes: {totalPassesComplete}, Shot Assists: {totalShotAssists}, Goal Assists: {totalGoalAssists}, xA: {totalxA:.2f}")

recipient_stats = euro_passes.get_recipient_metrics(player_pass_data)

if len(recipient_stats) > 0:
    with st.expander(f"{selected_player}'s Recipient Data ({len(recipient_stats)} players)"):
        st.write(recipient_stats)

st.divider()
st.header(f"{selected_player}'s Pitch Plot, #{jerseyNumber} Chances Created at Euro 2024")

import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
import numpy as np

import matplotlib.font_manager as font_manager
import matplotlib.patheffects as path_effects

font_path = 'data/fonts/futura/futura.ttf'
font_path_light = 'data/fonts/futura/Futura Light font.ttf'     
font_props = font_manager.FontProperties(fname=font_path)
font_props_light = font_manager.FontProperties(fname=font_path_light)

path_eff = [path_effects.Stroke(linewidth=0.5, foreground='black'),
            path_effects.Normal()]

# Set up the figure and axes
fig = plt.figure(figsize=(8, 12), dpi=300)
fig.patch.set_facecolor('#f3f3f3')

# [left, bottom, width, height]
ax1 = fig.add_axes([0.1, 0.82, 0.9, 0.22])
# ax1.set_facecolor('#f3f3f3')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)


ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

ax1.axis('off')



# Add Title and Subtitle
ax1.text(0.5, 0.85, f'{selected_player} #{jerseyNumber}', fontsize=20, fontproperties=font_props,
         fontweight='bold', color='black', ha='center')
ax1.text(0.5, 0.72, 'Chances Created, at Euro 2024',
         fontsize=14, fontweight='bold', fontproperties=font_props, color='black', ha='center')



yPos1 = 0.55
yPos2 = yPos1 + 0.02

# Define key items for plotting
key_items = [
    # Pass Key
    {'x_text': 0.15, 'y_text': yPos1, 'text': 'Pass Key:', 'fontprops': font_props},
    
    {'x_text': 0.3, 'y_text': yPos1, 'text': 'Shot Assists', 'fontprops': font_props_light,
     'x_scatter': 0.4, 'y_scatter': yPos2, 'color': 'seagreen'},
    
    {'x_text': 0.55, 'y_text': yPos1, 'text': 'Goal Assists', 'fontprops': font_props_light,
     'x_scatter': 0.65, 'y_scatter': yPos2, 'color': 'goldenrod'}

]

# Function to add key items
def add_key_item(ax, item):
    if 'x_text' in item:
        ax.text(
            x=item['x_text'], 
            y=item['y_text'], 
            s=item['text'], 
            fontsize=12,
            fontproperties=item['fontprops'], 
            color='black', ha='left')
    if 'x_scatter' in item:
        ax.scatter(
            x=item['x_scatter'], 
            y=item['y_scatter'], 
            s=item.get('size', 150),
            color=item.get('color', 'white'), 
            edgecolor=item.get('color', 'black'),
            linewidth=0.8)
        
        ax.arrow(item['x_scatter'], item['y_scatter'], dx=0.075, dy=0, width=0.02,
                  head_length=0.02, color=item.get('color', 'white'), linewidth=1
                  
                  )
        ax.text(
            x=item['x_scatter'], 
            y=item['y_scatter']-0.009, 
            s="19", 
            fontsize=10,
            fontproperties=item['fontprops'], path_effects=path_eff,
            color='black', ha='center', va='center')   
# Add key items to ax1
for item in key_items:
    add_key_item(ax1, item)

# ax1.text(0.5, yPos1-0.2, additionalText, fontsize=10, fontproperties=font_props_light,
#          fontweight='bold', color='white', ha='center')


# [left, bottom, width, height]
ax2 = fig.add_axes([0.05, 0.4, 0.95, 0.6])
# ax2.set_facecolor('#f3f3f3')

# Draw the pitch
pitch = pitch = VerticalPitch(
    pitch_type='statsbomb', 
    line_zorder=2,
    half=True, 
    pitch_color='#0C0D0E', 
    # pad_bottom=.5, 
    line_color='black',
    linewidth=.75,
    axis=True, label=True
)

pitch.draw(ax=ax2)
ax2.axis('off')

# Heatmap for completed passes
bin_statistic = pitch.bin_statistic(
    player_pass_data['location'].str[0],
    player_pass_data['location'].str[1],
    statistic='count', bins=(12, 8), normalize=False)
pcm = pitch.heatmap(bin_statistic, cmap='Reds', edgecolor='dimgrey', ax=ax2, alpha=0.7)



# Function to plot events
def plot_events(events, event_type):
    for x in events.to_dict(orient='records'):
        color = ('goldenrod' if x['pass_goal_assist'] == True else 'seagreen') 
        # if x.get('pass_shot_assist', False) else
                 #'lightgrey')
                 
        if event_type == 'pass':
            pitch.scatter(x['location'][0], x['location'][1], s=125, color=color, 
                          ax=ax2, alpha=1, linewidth=0.8, edgecolor=color)                         

            pitch.arrows(x['location'][0], x['location'][1],
                x['pass_end_location'][0], x['pass_end_location'][1],
                color=color, alpha=1, ax=ax2, width=2,
                headwidth= 5, headlength=5,
                pivot='tail'
            )
            
        
            pitch.annotate(jerseyNumber, xy=(x['location'][0]-0.1, x['location'][1]),
                           ax=ax2, fontsize=8, fontproperties=font_props_light,
                           path_effects=path_eff,
                           color='black', ha='center', va='center')
        
        

# Plot passes, carries, and shots
plot_events(player_pass_chance_data, 'pass')

# Add colorbar # [left, bottom, width, height]
ax_cbar = fig.add_axes([0.1, 0.405, 0.39, 0.02])
cbar = plt.colorbar(pcm, cax=ax_cbar, orientation='horizontal')
cbar.outline.set_edgecolor('black')
cbar.ax.xaxis.set_tick_params(color='black')
plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='black')
# ax_cbar.set_facecolor(background_color)
#cbar.set_label('Completed Passes', color='white', fontsize=12, fontproperties=font_props_light)

max_value = np.max(bin_statistic['statistic'])
ax_cbar.text(x=max_value/2, y=1.4, s='Completed Passes', fontsize=12,
         fontproperties=font_props, color='black', ha='center')


yPos1 = 52.75
yPos2 = yPos1 - 4

# Stats annotations
stats = [
    {'x': 38, 'y1': yPos1, 'y2': yPos2, 'title': 'Passes',
     'value': f"{totalPassesComplete}/{totalPasses}",
     'color': 'red'},    
    {'x': 50, 'y1': yPos1, 'y2': yPos2, 'title': 'Shot Asts.',
     'value': f"{totalShotAssists}",
     'color': 'seagreen'},
    {'x': 61, 'y1': yPos1, 'y2': yPos2, 'title': 'Goal Asts.',
     'value': f"{totalGoalAssists}",
     'color': 'goldenrod'},
    {'x': 73.5, 'y1': yPos1, 'y2': yPos2, 'title': 'xA',
     'value': f"{totalxA:.2f}",
     'color': 'black'},
]

for stat in stats:
    ax2.text(stat['x'], stat['y1'], stat['title'], fontsize=12,
             fontproperties=font_props, color='black', ha='left')
    ax2.text(stat['x'], stat['y2'], stat['value'], fontsize=12,
             fontproperties=font_props, color=stat['color'], ha='left')
    

    



# plt.show()
st.pyplot(fig)

recipient_stats.rename(columns={
    "recipient_name": "Recipient",
    "pass_recipient_jersey_number": "Jersey #",
    "shots_created": "Shots Assisted",
    "goals_created": "Goals Assisted",
    "total_xA": "Total xA"
}, inplace=True)
recipient_stats.reset_index(drop=True, inplace=True)
recipient_stats.index = recipient_stats.index + 1

styled_table = recipient_stats.style\
    .set_table_styles([  # Styling only the header
        {'selector': 'thead th',
         'props': [('background-color', '#f3f3f3'),  # Dark gray header
                   ('color', 'black'),  # White text
                   ('font-weight', 'bold'),
                   ('text-align', 'center')]}
    ])\
    .format({"Jersey #": "#{:.0f}", "Total xA": "{:.2f}"})

st.subheader("Recipient Metrics")
st.table(styled_table)

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