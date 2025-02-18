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
from classes.chat import PassChat
from classes.description import PassDescription

from utils.page_components import add_common_page_elements

from utils.utils import (
    create_chat
)

def summarize_passing_insights(player_metrics, selected_player):
    """
    Summarizes key insights from the passing metrics.

    Args:
        player_metrics (DataFrame): DataFrame containing passing metrics for all players.
        selected_player (str): Name of the selected player.

    Returns:
        dict: Dictionary of insights related to passing.
    """
    player_data = player_metrics[player_metrics['player_name'] == selected_player].iloc[0]

    total_passes = player_data['total_passes']
    passes_completed = player_data['passes_completed']
    pass_completion = player_data['passes_complete_perc']
    chances_created = player_data['chances_created']
    goal_assists = player_data['goal_assists']
    xA = player_data['xA']
    avg_pass_angle = player_data['avg_pass_angle']

    passing_insights = (
        f"{selected_player} attempted **{total_passes} passes**, "
        f"completing **{passes_completed}** with a pass completion rate of **{pass_completion:.1f}%**. "
        f"They created **{chances_created} chances**, providing **{goal_assists} goal assists**, "
        f"with an expected assists (xA) value of **{xA:.2f}**. "
        f"Their average pass angle was **{avg_pass_angle:.2f}°**, indicating their typical passing direction."
    )

    return {"passing_insights": passing_insights}

def summarize_passing_distribution(player_metrics, selected_player):
    """
    Summarizes distribution plot insights for the selected player.

    Args:
        player_metrics (DataFrame): DataFrame containing metrics for all players.
        selected_player (str): Name of the selected player.

    Returns:
        str: Summary of the player's ranking in key passing metrics.
    """
    player_data = player_metrics[player_metrics['player_name'] == selected_player].iloc[0]
    
    pass_completion_rank = player_metrics['passes_complete_perc'].rank(ascending=False)[player_metrics['player_name'] == selected_player].values[0]
    xA_rank = player_metrics['xA'].rank(ascending=False)[player_metrics['player_name'] == selected_player].values[0]
    goal_assist_rank = player_metrics['goal_assists'].rank(ascending=False)[player_metrics['player_name'] == selected_player].values[0]

    distribution_insights = (
        f"Compared to other players, {selected_player} ranks **{int(pass_completion_rank)}** in pass completion percentage, "
        f"**{int(xA_rank)}** in expected assists (xA), and **{int(goal_assist_rank)}** in goal assists."
    )
    return distribution_insights

def summarize_pitch_insights(player_pass_data, selected_player):
    """
    Summarizes key insights from the player's passing locations and spread.

    Args:
        player_pass_data (DataFrame): DataFrame containing passing event data.
        selected_player (str): Name of the selected player.

    Returns:
        str: Summary of the player's passing tendencies on the pitch.
    """
    total_passes = len(player_pass_data)
    
    # Calculate zone-based statistics
    bins_x = 12  # Divide the pitch into 6 vertical zones
    bins_y = 8  # Divide the pitch into 4 horizontal zones
    player_pass_data['x_bin'] = pd.cut(player_pass_data['location'].str[0], bins=bins_x, labels=False)
    player_pass_data['y_bin'] = pd.cut(player_pass_data['location'].str[1], bins=bins_y, labels=False)
    
    unique_zones = player_pass_data.groupby(['x_bin', 'y_bin']).size().reset_index()
    unique_zone_count = len(unique_zones)
    
    # Identify pass clustering: Are they spread out or concentrated?
    passes_per_zone = total_passes / unique_zone_count
    if passes_per_zone > 6:
        pass_pattern = "highly concentrated in a few zones"
    elif passes_per_zone > 3:
        pass_pattern = "moderately distributed across multiple areas"
    else:
        pass_pattern = "widely spread across the pitch"

    # Identify the most used passing area
    most_frequent_zone = player_pass_data.groupby(['x_bin', 'y_bin']).size().idxmax()
    most_frequent_zone_passes = player_pass_data.groupby(['x_bin', 'y_bin']).size().max()

    pitch_insights = (
        f"{selected_player} attempted **{total_passes} passes**. "
        f"Their passes were **{pass_pattern}**, covering **{unique_zone_count} different pitch zones**. "
        f"The most common passing area was **zone {most_frequent_zone}**, where they made **{most_frequent_zone_passes} passes**."
    )
    
    return pitch_insights




# UI Setup
sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()




st.title("Euro Pass Analysis")
st.subheader(f"Player pass and chance creation visualisation and analysis at Euro 2024")


st.text("Data Sourced from StatsBomb, competition_id=55, season_id=282")

euro_passes = EuroPasses()

with st.expander("Raw Euro Passes Dataframe"):
    st.write(euro_passes.df)

st.divider()

st.sidebar.markdown("""<span style='color: white;'>Only showing selection of players with 1 or more goal assists.</span>""", unsafe_allow_html=True)
# Filter players who have at least 1 goal assist
players_with_assists = euro_passes.df[euro_passes.df['pass_goal_assist'] > 0]

# # Extract unique teams
# unique_teams = players_with_assists['team'].unique()

# # Print the list of unique teams
# print(unique_teams)

# # If using Streamlit, display it
# st.text(unique_teams)


# Aggregate goal assists per player and ensure integers
player_goal_assists = players_with_assists.groupby(['player_name', 'team'])['pass_goal_assist'].sum().reset_index()
player_goal_assists['pass_goal_assist'] = player_goal_assists['pass_goal_assist'].astype(int)  # Convert to int

# Sort players by goal assists in descending order (most assists first)
player_goal_assists = player_goal_assists.sort_values(by=['pass_goal_assist', 'player_name'], ascending=[False,True])

# Format player names with team and correct singular/plural wording for "goal assisted"
player_options = player_goal_assists.apply(
    lambda row: f"{row['player_name']} ({row['team']}) - {row['pass_goal_assist']} Goal Assisted"
    if row['pass_goal_assist'] == 1 
    else f"{row['player_name']} ({row['team']}) - {row['pass_goal_assist']} Goals Assisted",
    axis=1
)

# Create a dictionary to map the formatted name back to the actual player name
player_mapping = dict(zip(player_options, player_goal_assists['player_name']))

# Player selection dropdown with formatted options (most assists at the top)
selected_player_display = st.sidebar.selectbox("Select a player", player_options)

# Get the actual player name from the mapping
selected_player = player_mapping[selected_player_display]

# st.text(type(euro_passes.df)) 

# Filter relevant data
player_pass_data = euro_passes.get_player_pass_data(selected_player)
player_pass_chance_data = player_pass_data[(player_pass_data['pass_shot_assist'] == True) | (player_pass_data['pass_goal_assist'] == True)]

st.header(f"{selected_player}'s Data")

gamesPlayed, jerseyNumber, totalPasses, totalPassesComplete, totalShotAssists, totalGoalAssists, totalxA = euro_passes.get_player_metrics(player_pass_data)


with st.expander(f"{selected_player}'s Euro Passes Dataframe ({totalPasses})"):
    st.write(player_pass_data)

with st.expander(f"{selected_player}'s Euro Completed Passes Dataframe ({totalPassesComplete})"):
    st.write(player_pass_data[pd.isna(player_pass_data['pass_outcome'])])

if (totalShotAssists + totalGoalAssists) > 0:
    with st.expander(f"{selected_player}'s Euro Shot and Goal Assisted Passes Dataframe ({totalShotAssists + totalGoalAssists}, xA: {totalxA:.2f})"):
        st.write(player_pass_chance_data)


st.text(f"Matches featured in: {gamesPlayed}, Total Passes: {totalPasses}, Completed Passes: {totalPassesComplete}, Shot Assists: {totalShotAssists}, Goal Assists: {totalGoalAssists}, xA: {totalxA:.2f}")

recipient_stats = euro_passes.get_recipient_metrics(player_pass_data)

if len(recipient_stats) > 0:
    with st.expander(f"{selected_player}'s Recipient Data ({len(recipient_stats)} players)"):
        st.write(recipient_stats)

st.text(f"Creating chances for {len(recipient_stats)} players:\n{', '.join(recipient_stats['recipient_name'].unique())}")


st.divider()
st.header(f"{selected_player}'s Pitch Plot")
st.subheader(f"Chances Created in {gamesPlayed} matches at Euro 2024")

# Initialize the pitch plot
plotter = PitchPlot()

# Generate the figure
fig = plotter.create_pitch_plot(
    selected_player=selected_player,
    jersey_number=jerseyNumber,
    player_pass_data=player_pass_data,
    player_pass_chance_data=player_pass_chance_data,
    totalPassesComplete=totalPassesComplete,
    totalPasses=totalPasses,
    totalShotAssists=totalShotAssists,
    totalGoalAssists=totalGoalAssists,
    totalxA=totalxA
)    

st.pyplot(fig)

# Generate and display recipient stats table
styled_table = plotter.create_recipient_stats_table(recipient_stats)

st.subheader("Recipient Metrics")
st.table(styled_table)

st.write(selected_player)

from classes.visual import EuroPassVisualizer

visualizer = EuroPassVisualizer(euro_passes.df)
visualizer.plot_xa_stacked_barchart(selected_player=selected_player)



st.divider()

from classes.visual import DistributionPlotPasses



st.header(f"{selected_player}'s Pass Analysis and comparison")
st.subheader("Only comparing to players with at least 1 goal assist.")
player_metrics = plotter.calculate_player_metrics(euro_passes.df)

with st.expander(f"Euro Passes Distribution Plot Dataframe "):
    st.write(player_metrics)


# Define the metrics to normalize
metrics = ["games_played", "passes_completed", "chances_created", "goal_assists", "passes_complete_perc", "avg_pass_angle", "xA"]
metrics = metrics[::-1]  # Reverse order if needed

# Normalize each metric between -5 and 5
for metric in metrics:
    if metric in player_metrics.columns:
        min_val = player_metrics[metric].min()
        max_val = player_metrics[metric].max()
        
        # Avoid division by zero
        if max_val - min_val == 0:
            player_metrics[f"{metric}_Z"] = 0
        else:
            player_metrics[f"{metric}_Z"] = -7 + (player_metrics[metric] - min_val) * (14 / (max_val - min_val))

# with st.expander(f"Euro Passes Distribution Plot Dataframe "):
#     st.write(player_metrics)

distribution_plot = DistributionPlotPasses(metrics=metrics)

# Add all players' data
distribution_plot.add_group_data(player_metrics)

# Highlight selected player
selected_player_data = player_metrics[player_metrics['player_name'] == selected_player].iloc[0]
distribution_plot.add_player(selected_player_data, selected_player)

# Add title
distribution_plot.add_title_from_player(selected_player)

# Show the plot in Streamlit
distribution_plot.show()



# Get selected player data
selected_player_data = player_metrics[player_metrics['player_name'] == selected_player].iloc[0]

description_text = (
    f"{selected_player} attempted **{selected_player_data['total_passes']} passes**, "
    f"completing **{selected_player_data['passes_completed']}** "
    f"with a pass completion rate of **{selected_player_data['passes_complete_perc']:.1f}%**. "
    f"They created **{selected_player_data['chances_created']} chances** and provided **{selected_player_data['goal_assists']} goal assists**. "
    f"Their expected assists (xA) value was **{selected_player_data['xA']:.2f}**, "
    f"with an average pass angle of **{selected_player_data['avg_pass_angle']:.2f} radians**."
)


st.markdown(description_text)

# Generate insights
passing_summaries = summarize_passing_insights(player_metrics, selected_player)
distribution_summary = summarize_passing_distribution(player_metrics, selected_player)
pitch_summary = summarize_pitch_insights(player_pass_data, selected_player) 

# Combine all insights
all_insights = {
    "description_text": description_text,
    "passing_insights": passing_summaries["passing_insights"],
    "distribution_insights": distribution_summary,
    "pitch_insights": pitch_summary
}

# Create chat with insights
to_hash = (selected_player,)
chat = create_chat(to_hash, PassChat, selected_player, player_metrics)

if chat.state == "empty":
    # Add content to chat
    description = PassDescription(selected_player, selected_player_data, all_insights)
    summary = description.stream_gpt()

    chat.add_message(
        f"Please summarize {selected_player}'s passing performance.",
        role="user",
        user_only=False,
        visible=False,
    )
    chat.add_message(summary)
    chat.state = "default"

# Display chat and save its state
chat.get_input()
chat.display_messages()
chat.save_state()