from io import BytesIO


def split_names(player_names):
    # Iterate over each name in the player_names list
    # and return a modified list of names
    return [
        # If the name consists of only one word
        # or the second-to-last word does not have a length of 2 characters,
        # then the last word is the last name
        (
            name.split()[-1]
            if len(name.split()) == 1 or len(name.split()[-2]) != 2
            # Otherwise, join the second-to-last and last words with a space in between
            # and consider it as the last name
            else " ".join(name.split()[-2:])
        )
        for name in player_names
    ]


def add_per_90(attributes):
    return [
        (
            c + " per 90"
            if "%" not in c
            and "per" not in c
            and "adj" not in c
            and "eff" not in c
            and " - " not in c
            else c
        )
        for c in attributes
    ]


def normalize_text(s, sep_token=" \n "):
    s = " ".join(s.split())
    s = s.replace(". ,", ",")
    s = s.replace(" ,", ",")
    s = s.replace("..", ".")
    s = s.replace(". .", ".")
    s = s.replace("\n", "")
    s = s.strip()
    return s


def insert_newline(s, n_length=15):
    if len(s) <= n_length:
        return s
    else:
        last_space_before_15 = s.rfind(" ", 0, n_length)
        if last_space_before_15 == -1:  # No space found within the first 15 characters
            return s  # Return original string
        else:
            # Split the string at the space and insert a newline
            return s[:last_space_before_15] + "\n" + s[last_space_before_15 + 1 :]


# Function to convert RGBA to HEX
def rgba_to_hex(rgba):
    r, g, b, a = rgba
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def convert_df_to_csv(df, n=1000, ignore=[]):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    # cols = df.columns
    return df.head(n).to_csv(index=None).encode("utf-8")


def get_img_bytes(fig, custom=False, format="png", dpi=200):
    tmpfile = BytesIO()

    if custom:
        fig.savefig(
            tmpfile,
            format=format,
            dpi=dpi,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            pad_inches=0.35,
        )
    else:
        fig.savefig(
            tmpfile,
            format=format,
            dpi=dpi,
            facecolor=fig.get_facecolor(),
            transparent=False,
        )  # , frameon=False)  # , transparent=False, bbox_inches='tight', pad_inches=0.35)

    tmpfile.seek(0)

    return tmpfile


import matplotlib.colors as c


def hex_color_transparency(hex, alpha):
    return c.to_hex(c.to_rgba(hex, alpha), True)


import copy


def select_player(container, players, gender, position):

    # Make a copy of Players object
    player = copy.deepcopy(players)

    # Filter players by position and select a player with sidebar selectors
    with container:

        # Filter for player name
        player.select_and_filter(
            column_name="player_name",
            label="Player",
        )

        # Return data point

        player = player.to_data_point(gender, position)

    return player


def select_country(container, countries):

    # Make a copy of Players object
    country = copy.deepcopy(countries)

    # rnd = int(country.select_random()) # does not work because of page refresh!
    # Filter country by position and select a player with sidebar selectors
    with container:

        # Filter for player name
        country.select_and_filter(
            column_name="country",
            label="Country",
            # default_index=rnd,  # randomly select a country for default
        )

        # Return data point

        country = country.to_data_point()

    return country


def create_chat(to_hash, chat_class, *args, **kwargs):
    chat_hash_state = hash(to_hash)
    chat = chat_class(chat_hash_state, *args, **kwargs)
    return chat

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mplsoccer import Pitch

def plot_player_runs(player_runs, player_name):
    """
    Creates a football pitch plot with subplots showing player runs
    based on their angles (-90° to 90° and outside these bounds).

    Args:
        player_runs (DataFrame): DataFrame containing the player's runs.
        player_name (str): Name of the player.

    Returns:
        matplotlib.figure.Figure: A Matplotlib figure containing the pitch plot.
    """
    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list("mycmap", ['blue', 'red', 'blue'])

    # Create the pitch
    pitch = Pitch(
        pitch_type="opta",
        goal_type='box',
        pitch_color="w",
        linewidth=1,
        spot_scale=0,
        line_color="k",
        line_zorder=1,
        positional=True
    )

    # Create a figure with two subplots (stacked vertically)
    fig, axs = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'hspace': 0.1}, dpi=300)

    # Normalize the run_angle for colormap scaling across both subplots
    norm = Normalize(vmin=-180, vmax=180)

    # 1st Subplot: Angles between -90° and 90°
    ax1 = axs[0]
    pitch.draw(ax=ax1)

    # Filter runs with angles between -90° and 90°
    runs_between_90 = player_runs[(player_runs['run_angle'] >= -90) & (player_runs['run_angle'] <= 90)]

    for _, row in runs_between_90.iterrows():
        # Map run_angle to the colormap
        color = cmap(norm(row['run_angle']))
        
        # Draw an arrow for each run
        pitch.arrows(
            row['start_x'], row['start_y'],
            row['end_x'], row['end_y'],
            ax=ax1,
            color=color,
            width=2,
            headwidth=3,
            headlength=3,
            alpha=0.8
        )

    # Add a title to the subplot
    ax1.set_title(f"Runs with Angles between -90° and 90° - {player_name}", fontsize=14, color="black")

    # 2nd Subplot: Angles < -90° or > 90°
    ax2 = axs[1]
    pitch.draw(ax=ax2)

    # Filter runs with angles less than -90° or greater than 90°
    runs_outside_90 = player_runs[(player_runs['run_angle'] < -90) | (player_runs['run_angle'] > 90)]

    for _, row in runs_outside_90.iterrows():
        # Map run_angle to the colormap
        color = cmap(norm(row['run_angle']))
        
        # Draw an arrow for each run
        pitch.arrows(
            row['start_x'], row['start_y'],
            row['end_x'], row['end_y'],
            ax=ax2,
            color=color,
            width=2,
            headwidth=3,
            headlength=3,
            alpha=0.8
        )

    # Add a title to the subplot
    ax2.set_title(f"Runs with Angles <-90° or >90° - {player_name}", fontsize=14, color="black")

    # Add a single colorbar for both subplots
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.05, pad=0.01)
    cbar.set_label('Run Angle (Degrees)', fontsize=12)

    return fig

def plot_radar(player_runs, player_name):
    """
    Creates a radar plot showing the distribution of run angles for a player.

    Args:
        player_runs (DataFrame): DataFrame containing the player's runs.
        player_name (str): Name of the player.

    Returns:
        matplotlib.figure.Figure: A Matplotlib figure containing the radar plot.
    """
    # Define 30-degree bins for angles from -180 to 180
    angle_bins = np.arange(-180, 181, 30)  # 30° segments
    binned_angles = np.digitize(player_runs['run_angle'], bins=angle_bins, right=False)

    # Count the number of runs in each segment
    angle_counts = [np.sum(binned_angles == i) for i in range(1, len(angle_bins))]

    # Define the angles (central angle of each segment) for the radar plot
    angles = np.deg2rad(angle_bins[:-1] + 15)  # Align angle of each bin with the grid line
    width = np.deg2rad(30)  # Width of each segment (30 degrees)

    # Define a custom colormap
    cmap = LinearSegmentedColormap.from_list("mycmap", ['blue', 'red', 'blue'])
    norm = Normalize(vmin=-180, vmax=180)

    # Create radar plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True}, dpi=300)

    # Plot each segment as a curved bar
    for i, count in enumerate(angle_counts):
        # Calculate the color based on the bin's central angle
        bin_center_angle = (angle_bins[i] + angle_bins[i + 1]) / 2
        color = cmap(norm(bin_center_angle))
        
        # Draw each segment as a bar
        ax.bar(
            angles[i], count, width=width, color=color, alpha=0.7, edgecolor="black", linewidth=0.5
        )

    # Add the average angle as a dashed line
    average_angle_degrees = player_runs['run_angle'].mean()
    average_angle = np.deg2rad(average_angle_degrees)
    ax.plot(
        [average_angle, average_angle],  # Start and end at the same angle
        [0, max(angle_counts)],         # From the center to the maximum radius
        linestyle='--', color='black', linewidth=2, label=f'Average Angle {int(average_angle_degrees)}°'
    )

    # Add the modulus average angle as a dotted line
    average_angle_degrees_abs = abs(player_runs['run_angle']).mean()
    average_angle_abs = np.deg2rad(average_angle_degrees_abs)
    ax.plot(
        [average_angle_abs, average_angle_abs],  # Start and end at the same angle
        [0, max(angle_counts)],         # From the center to the maximum radius
        linestyle='dotted', color='black', linewidth=2, label=f'Average |Angle| {int(average_angle_degrees_abs)}°'
    )

    # Ensure full-circle labels
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)

    ax.set_thetamin(-180)
    ax.set_thetamax(180)

    # Add gridlines and labels
    ax.set_xticks(np.deg2rad(angle_bins[:-1]))  # Label at every 30°
    xtick_labels = ["±180°"] + [f"{angle}°" for angle in angle_bins[1:-1]]
    ax.set_xticklabels(xtick_labels)

    # Add radial gridlines
    ax.set_yticks(range(1, max(angle_counts) + 1))
    ax.set_yticklabels(range(1, max(angle_counts) + 1), fontsize=10)

    # Add a title
    ax.set_title(f"Run Angle Distribution - {player_name}", fontsize=14, pad=20)

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.1)
    cbar.set_label('Angle (Degrees)', fontsize=12)

    # Add legend for the average angle
    ax.legend(loc='lower left', bbox_to_anchor=(-0.08, -0.09), fontsize=10)

    return fig