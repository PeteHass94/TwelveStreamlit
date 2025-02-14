# Library imports
import streamlit as st
import numpy as np

from utils.utils import normalize_text

from classes.visual import DistributionPlotRuns

from classes.data_source import RunStats 

from classes.data_source import PlayerStats
from classes.data_point import Player

from classes.description import (
    RunDescription,
)

from classes.chat import RunChat


from utils.page_components import (add_common_page_elements, select_runs)
from utils.utils import (
    select_player,
    create_chat,
    plot_player_runs,
    plot_radar
)

# def show():
sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.title("Champion League 2018 Run Analysis")
st.divider()

# Function to summarize radar and pitch plot insights
def summarize_visuals(player_runs, selected_player):
    """
    Summarizes key insights from the radar and pitch plots.

    Args:
        player_runs (DataFrame): DataFrame containing runs for the selected player.
        selected_player (str): Name of the selected player.

    Returns:
        dict: Dictionary of insights from radar and pitch plots.
    """
    # Pitch plot insights
    forward_runs = len(player_runs)
    avg_angle = player_runs['run_angle'].mean()
    runs_to_box = len(player_runs[player_runs['end_x'] >= 90])  # Runs ending near the box
    high_speed_runs = len(player_runs[player_runs['avg_speed'] > 5])  # Runs with avg speed > 5 m/s

    pitch_insights = (
        f"{selected_player} made {forward_runs} forward runs. "
        f"The average run angle was {avg_angle:.2f} degrees, "
        f"with {runs_to_box} runs ending near the opposition box. "
        f"{high_speed_runs} runs were at a speed exceeding 5 m/s."
    )

    # Radar plot insights
    angle_distribution = player_runs['run_angle'].value_counts(bins=6).idxmax()
    avg_speed = player_runs['avg_speed'].mean()

    radar_insights = (
        f"The radar plot shows that {selected_player}'s runs are "
        f"most concentrated around angles between {angle_distribution.left:.1f}° and {angle_distribution.right:.1f}°. "
        f"The player's average speed during runs was {avg_speed:.2f} m/s."
    )

    return {
        "pitch_insights": pitch_insights,
        "radar_insights": radar_insights,
    }

# Add distribution plot insights
def summarize_distribution_plot(player_metrics, selected_player):
    """
    Summarizes distribution plot insights for the selected player.

    Args:
        player_metrics (DataFrame): DataFrame containing metrics for all players.
        selected_player (str): Name of the selected player.

    Returns:
        str: Summary of the player's position in the distribution.
    """
    player_data = player_metrics[player_metrics['player'] == selected_player].iloc[0]
    forward_run_rank = player_metrics['forward_runs_Z'].rank(ascending=False)[player_metrics['player'] == selected_player].values[0]

    distribution_insights = (
        f"In comparison to other players, {selected_player} ranks "
        f"{int(forward_run_rank)} for forward runs and has above-average performance in max speed, "
        f"average speed, and total distance."
    )
    return distribution_insights





# Specify the match ID
match_id = 18768058

# Load and process run data
runs = RunStats(match_id)

runs.df['run_time'] = (runs.df['time_end'] - runs.df['time_start']).dt.total_seconds()

runs.df['player_label'] = runs.df.apply(
    lambda row: f"{row['player']} ({row['team_name']})", axis=1
)

# Add distance to goal calculation
goal_x, goal_y = 100, 50  # Coordinates of the goal center
runs.df['distance_to_goal'] = np.sqrt((runs.df['end_x'] - goal_x)**2 + (runs.df['end_y'] - goal_y)**2)

runs.df['forward_runs'] = runs.df['Forward runs']



# Define the opposition box coordinates
box_x = 83
box_y_min = 21
box_y_max = 79

# Create a derived column to check for runs into the opposition box
runs.df['runs_in_opposition_box'] = (
    (runs.df['start_x'] >= box_x) &  # Start inside the box
    (runs.df['start_y'] >= box_y_min) & (runs.df['start_y'] <= box_y_max) 
)

# Calculate deltas and angles for the selected player's runs
runs.df['delta_x'] = runs.df['end_x'] - runs.df['start_x']
runs.df['delta_y'] = runs.df['end_y'] - runs.df['start_y']
runs.df['direction_angle'] = np.where(runs.df['delta_y'] >= 0, 1, -1)
runs.df['run_angle'] = np.degrees(
    np.arctan2(abs(runs.df['delta_y']), runs.df['delta_x'])
)
runs.df['run_angle'] = runs.df['run_angle'] * runs.df['direction_angle']
runs.df['absolute_run_angle'] = runs.df['run_angle'] * runs.df['direction_angle']

# Calculate player metrics
player_metrics = runs.df.groupby(['player', 'team_name', 'player_label']).agg(
    avg_run_angle=('run_angle', 'mean'),
    avg_abs_run_angle=('absolute_run_angle', 'mean'),
    runs_in_opposition_box=('runs_in_opposition_box', 'sum'),  
    forward_runs=('forward_runs', 'sum'),  
    max_speed=('max_speed', 'max'),
    avg_speed=('avg_speed', 'mean'),
    avg_distance=('Distance', 'mean'),
    total_distance=('Distance', 'sum')
).reset_index()


# Add Z-scores for selected metrics
metrics = ['forward_runs', 'max_speed', 'avg_speed', 'avg_distance', 'total_distance', 'avg_run_angle', 'avg_abs_run_angle']
metrics = metrics[::-1]
for metric in metrics:
    if metric in player_metrics.columns:
        player_metrics[f"{metric}_Z"] = (player_metrics[metric] - player_metrics[metric].mean()) / player_metrics[metric].std()


# Sidebar to select a player and view detailed runs
selected_player, detailed_runs_df = select_runs(
    sidebar_container, player_metrics, runs
)


# Filter the runs for the selected player
player_runs = detailed_runs_df[detailed_runs_df['player'] == selected_player]

# Optional: Show raw data
with st.expander("Raw Run Dataframe"):
    st.write(runs.df)

with st.expander("Raw Player Run Dataframe"):
    st.write(player_runs)

with st.expander("Calculated Player Run Metrics"):
    st.write(player_metrics)

# Radar and pitch plots
pitch_fig = plot_player_runs(player_runs, selected_player)
radar_fig = plot_radar(player_runs, selected_player)

st.subheader(f"Pitch Plot of {selected_player}'s Runs")
st.pyplot(pitch_fig)

st.subheader(f"Radar Plot of {selected_player}'s Run Angle Distribution")
st.pyplot(radar_fig)

# Distribution plot for runs
distribution_plot = DistributionPlotRuns(metrics=metrics)
distribution_plot.add_title(
    f"Run Metric Distributions for {selected_player}",
    f"Based on match data"
)

# Add all players' data to the distribution plot
distribution_plot.add_group_data(
    df_plot=player_metrics
)

# Highlight the selected player's data
selected_player_data = player_metrics[player_metrics['player'] == selected_player].iloc[0]
distribution_plot.add_player(
    player_metrics=selected_player_data,
    player_name=selected_player
)

# Display the distribution plot
st.subheader("Distribution Plot of Run Metrics")
distribution_plot.show()

# Description of player metrics with bold formatting for stats
description_text = (
    f"{selected_player} recorded **{selected_player_data['forward_runs']} forward runs**, "
    f"with a maximum speed of **{selected_player_data['max_speed']:.2f} m/s** and an average speed of **{selected_player_data['avg_speed']:.2f} m/s**. "
    f"The average distance covered per run was **{selected_player_data['avg_distance']:.2f} meters** and total distance was **{selected_player_data['total_distance']:.2f} meters**. "
    f"The average run angle of their runs was **{selected_player_data['avg_run_angle']:.2f}°** and average absolute value of the run angle was **{selected_player_data['avg_abs_run_angle']:.2f}°**."
)


st.markdown(description_text)

# Generate insights
visual_summaries = summarize_visuals(player_runs, selected_player)
distribution_summary = summarize_distribution_plot(player_metrics, selected_player)

# Combine all insights for the chat and description
all_insights = {
    "description_text": description_text,
    "pitch_insights": visual_summaries["pitch_insights"],
    "radar_insights": visual_summaries["radar_insights"],
    "distribution_insights": distribution_summary,
}

# Create chat with insights
to_hash = (selected_player,)
chat = create_chat(to_hash, RunChat, selected_player, player_metrics)

if chat.state == "empty":

    # Add content to chat
    description = RunDescription(selected_player, selected_player_data, all_insights)
    summary = description.stream_gpt()

    chat.add_message(
        f"Please summarize {selected_player}'s runs for me.",
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
