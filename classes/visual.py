import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


from utils.sentences import format_metric
from classes.data_point import Player, Country, Person
from classes.data_source import PlayerStats, CountryStats, PersonStat
from typing import Union


def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


def rgb_to_color(rgb_color: tuple, opacity=1):
    return f"rgba{(*rgb_color, opacity)}"


def tick_text_color(color, text, alpha=1.0):
    # color: hexadecimal
    # alpha: transparency value between 0 and 1 (default is 1.0, fully opaque)
    s = (
        "<span style='color:rgba("
        + str(int(color[1:3], 16))
        + ","
        + str(int(color[3:5], 16))
        + ","
        + str(int(color[5:], 16))
        + ","
        + str(alpha)
        + ")'>"
        + str(text)
        + "</span>"
    )
    return s


class Visual:
    # Can't use streamlit options due to report generation
    bg_gray = hex_to_rgb('#f3f3f3')
    dark_green = hex_to_rgb(
        "#002c1c"
    )  # hex_to_rgb(st.get_option("theme.secondaryBackgroundColor"))
    medium_green = hex_to_rgb("#003821")
    bright_green = hex_to_rgb(
        "#00A938"
    )  # hex_to_rgb(st.get_option("theme.primaryColor"))
    bright_orange = hex_to_rgb("#ff4b00")
    bright_yellow = hex_to_rgb("#ffcc00")
    bright_blue = hex_to_rgb("#0095FF")
    white = hex_to_rgb("#ffffff")  # hex_to_rgb(st.get_option("theme.backgroundColor"))
    gray = hex_to_rgb("#808080")
    black = hex_to_rgb("#000000")
    light_gray = hex_to_rgb("#d3d3d3")
    table_green = hex_to_rgb("#009940")
    table_red = hex_to_rgb("#FF4B00")

    def __init__(self, pdf=False, plot_type="scout"):
        self.pdf = pdf
        if pdf:
            self.font_size_multiplier = 1.4
        else:
            self.font_size_multiplier = 1.0
        self.fig = go.Figure()
        self._setup_styles()
        self.plot_type = plot_type

        if plot_type == "scout":
            self.annotation_text = (
                "<span style=''>{metric_name}: {data:.2f} per 90</span>"
            )
        else:
            # self.annotation_text = "<span style=''>{metric_name}: {data:.0f}/66</span>"  # TODO: this text will not automatically update!
            self.annotation_text = "<span style=''>{metric_name}: {data:.2f}</span>"

    def show(self):
        st.plotly_chart(
            self.fig,
            config={"displayModeBar": False},
            height=500,
            use_container_width=True,
        )

    def _setup_styles(self):
        side_margin = 60
        top_margin = 75
        pad = 16
        self.fig.update_layout(
            autosize=True,
            height=500,
            margin=dict(l=side_margin, r=side_margin, b=70, t=top_margin, pad=pad),
            paper_bgcolor=rgb_to_color(self.dark_green),
            plot_bgcolor=rgb_to_color(self.dark_green),
            legend=dict(
                orientation="h",
                font={
                    "color": rgb_to_color(self.white),
                    "family": "Gilroy-Light",
                    "size": 11 * self.font_size_multiplier,
                },
                itemclick=False,
                itemdoubleclick=False,
                x=0.5,
                xanchor="center",
                y=-0.2,
                yanchor="bottom",
                valign="middle",  # Align the text to the middle of the legend
            ),
            xaxis=dict(
                tickfont={
                    "color": rgb_to_color(self.white, 0.5),
                    "family": "Gilroy-Light",
                    "size": 12 * self.font_size_multiplier,
                },
            ),
        )

    def add_title(self, title, subtitle):
        self.title = title
        self.subtitle = subtitle
        self.fig.update_layout(
            title={
                "text": f"<span style='font-size: {15*self.font_size_multiplier}px'>{title}</span><br>{subtitle}",
                "font": {
                    "family": "Gilroy-Medium",
                    "color": rgb_to_color(self.white),
                    "size": 12 * self.font_size_multiplier,
                },
                "x": 0.05,
                "xanchor": "left",
                "y": 0.93,
                "yanchor": "top",
            },
        )

    def add_low_center_annotation(self, text):
        self.fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.07,
            text=text,
            showarrow=False,
            font={
                "color": rgb_to_color(self.white, 0.5),
                "family": "Gilroy-Light",
                "size": 12 * self.font_size_multiplier,
            },
        )

    def show(self):
        st.plotly_chart(
            self.fig,
            config={"displayModeBar": False},
            height=500,
            use_container_width=True,
        )

    def close(self):
        pass


class DistributionPlot(Visual):
    def __init__(self, columns, labels=None, *args, **kwargs):
        self.empty = True
        self.columns = columns
        self.marker_color = (
            c for c in [Visual.white, Visual.bright_yellow, Visual.bright_blue]
        )
        self.marker_shape = (s for s in ["square", "hexagon", "diamond"])
        super().__init__(*args, **kwargs)
        if labels is not None:
            self._setup_axes(labels)
        else:
            self._setup_axes()

    def _setup_axes(self, labels=["Worse", "Average", "Better"]):
        self.fig.update_xaxes(
            range=[-4, 4],
            fixedrange=True,
            tickmode="array",
            tickvals=[-3, 0, 3],
            ticktext=labels,
        )
        self.fig.update_yaxes(
            showticklabels=False,
            fixedrange=True,
            gridcolor=rgb_to_color(self.medium_green),
            zerolinecolor=rgb_to_color(self.medium_green),
        )

    def add_group_data(self, df_plot, plots, names, legend, hover="", hover_string=""):
        showlegend = True

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string

            metric_name = format_metric(col)

            temp_df = pd.DataFrame(df_plot[col + hover])
            temp_df["name"] = metric_name

            self.fig.add_trace(
                go.Scatter(
                    x=df_plot[col + plots],
                    y=np.ones(len(df_plot)) * i,
                    mode="markers",
                    marker={
                        "color": rgb_to_color(self.bright_green, opacity=0.2),
                        "size": 10,
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=names,
                    customdata=df_plot[col + hover],
                    name=legend,
                    showlegend=showlegend,
                )
            )
            showlegend = False

    def add_data_point(
        self, ser_plot, plots, name, hover="", hover_string="", text=None
    ):
        if text is None:
            text = [name]
        elif isinstance(text, str):
            text = [text]
        legend = True
        color = next(self.marker_color)
        marker = next(self.marker_shape)

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string

            metric_name = format_metric(col)

            self.fig.add_trace(
                go.Scatter(
                    x=[ser_plot[col + plots]],
                    y=[i],
                    mode="markers",
                    marker={
                        "color": rgb_to_color(color, opacity=0.5),
                        "size": 10,
                        "symbol": marker,
                        "line_width": 1.5,
                        "line_color": rgb_to_color(color),
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=text,
                    customdata=[ser_plot[col + hover]],
                    name=name,
                    showlegend=legend,
                )
            )
            legend = False

            self.fig.add_annotation(
                x=0,
                y=i + 0.4,
                text=self.annotation_text.format(
                    metric_name=metric_name,
                    data=(
                        ser_plot[col]
                        # if self.plot_type == "scout"
                        # else ser_plot[col + hover]
                    ),
                ),
                showarrow=False,
                font={
                    "color": rgb_to_color(self.white),
                    "family": "Gilroy-Light",
                    "size": 12 * self.font_size_multiplier,
                },
            )

    # def add_player(self, player: Player, n_group,metrics):

    #     # Make list of all metrics with _Z and _Rank added at end
    #     metrics_Z = [metric + "_Z" for metric in metrics]
    #     metrics_Ranks = [metric + "_Ranks" for metric in metrics]

    #     self.add_data_point(
    #         ser_plot=player.ser_metrics,
    #         plots = '_Z',
    #         name=player.name,
    #         hover='_Ranks',
    #         hover_string="Rank: %{customdata}/" + str(n_group)
    #     )

    def add_player(self, player: Union[Player, Country], n_group, metrics):

        # # Make list of all metrics with _Z and _Rank added at end
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]

        # Determine the appropriate attributes for player or country
        if isinstance(player, Player):
            ser_plot = player.ser_metrics
            name = player.name
        elif isinstance(player, Country):  # Adjust this based on your class structure
            ser_plot = (
                player.ser_metrics
            )  # Assuming countries have a similar metric structure
            name = player.name
        else:
            raise TypeError("Invalid player type: expected Player or Country")

        self.add_data_point(
            ser_plot=ser_plot,
            plots="_Z",
            name=name,
            hover="_Ranks",
            hover_string="Rank: %{customdata}/" + str(n_group),
        )

    # def add_players(self, players: PlayerStats, metrics):

    #     # Make list of all metrics with _Z and _Rank added at end
    #     metrics_Z = [metric + "_Z" for metric in metrics]
    #     metrics_Ranks = [metric + "_Ranks" for metric in metrics]

    #     self.add_group_data(
    #         df_plot=players.df,
    #         plots="_Z",
    #         names=players.df["player_name"],
    #         hover="_Ranks",
    #         hover_string="Rank: %{customdata}/" + str(len(players.df)),
    #         legend=f"Other players  ",  # space at end is important
    #     )

    def add_players(self, players: Union[PlayerStats, CountryStats], metrics):

        # Make list of all metrics with _Z and _Rank added at end
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]

        if isinstance(players, PlayerStats):
            self.add_group_data(
                df_plot=players.df,
                plots="_Z",
                names=players.df["player_name"],
                hover="_Ranks",
                hover_string="Rank: %{customdata}/" + str(len(players.df)),
                legend=f"Other players  ",  # space at end is important
            )
        elif isinstance(players, CountryStats):
            self.add_group_data(
                df_plot=players.df,
                plots="_Z",
                names=players.df["country"],
                hover="_Ranks",
                hover_string="Rank: %{customdata}/" + str(len(players.df)),
                legend=f"Other countries  ",  # space at end is important
            )
        else:
            raise TypeError("Invalid player type: expected Player or Country")

    # def add_title_from_player(self, player: Player):
    #     self.player = player

    #     title = f"Evaluation of {player.name}?"
    #     subtitle = f"Based on {player.minutes_played} minutes played"

    #     self.add_title(title, subtitle)

    def add_title_from_player(self, player: Union[Player, Country]):
        self.player = player

        title = f"Evaluation of {player.name}?"
        if isinstance(player, Player):
            subtitle = f"Based on {player.minutes_played} minutes played"
        elif isinstance(player, Country):
            subtitle = f"Based on questions answered in the World Values Survey"
        else:
            raise TypeError("Invalid player type: expected Player or Country")

        self.add_title(title, subtitle)


# ---------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------


class DistributionPlotPersonality(Visual):
    def __init__(self, columns, *args, **kwargs):
        self.empty = True
        self.columns = columns
        self.marker_color = (
            c for c in [Visual.white, Visual.bright_yellow, Visual.bright_blue]
        )
        self.marker_shape = (s for s in ["square", "hexagon", "diamond"])
        super().__init__(*args, **kwargs)
        self._setup_axes()

    def _setup_axes(self):
        self.fig.update_xaxes(
            range=[-4, 4],
            fixedrange=True,
            tickmode="array",
            tickvals=[-3, 0, 3],
            ticktext=["Worse", "Average", "Better"],
        )
        self.fig.update_yaxes(
            showticklabels=False,
            fixedrange=True,
            gridcolor=rgb_to_color(self.medium_green),
            zerolinecolor=rgb_to_color(self.medium_green),
        )

    def add_group_data(self, df_plot, plots, names, legend, hover="", hover_string=""):
        showlegend = True

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string

            metric_name = format_metric(col)

            temp_df = pd.DataFrame(df_plot[col + hover])
            temp_df["name"] = metric_name

            self.fig.add_trace(
                go.Scatter(
                    x=df_plot[col + plots],
                    y=np.ones(len(df_plot)) * i,
                    mode="markers",
                    marker={
                        "color": rgb_to_color(self.bright_green, opacity=0.2),
                        "size": 10,
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=names,
                    customdata=round(df_plot[col + hover]),
                    name=legend,
                    showlegend=showlegend,
                )
            )
            showlegend = False

    def add_data_point(
        self, ser_plot, plots, name, hover="", hover_string="", text=None
    ):
        if text is None:
            text = [name]
        elif isinstance(text, str):
            text = [text]
        legend = True
        color = next(self.marker_color)
        marker = next(self.marker_shape)

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string

            metric_name = format_metric(col)

            self.fig.add_trace(
                go.Scatter(
                    x=[ser_plot[col + plots]],
                    y=[i],
                    mode="markers",
                    marker={
                        "color": rgb_to_color(color, opacity=0.5),
                        "size": 10,
                        "symbol": marker,
                        "line_width": 1.5,
                        "line_color": rgb_to_color(color),
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=text,
                    customdata=[round(ser_plot[col + hover])],
                    name=name,
                    showlegend=legend,
                )
            )
            legend = False

            self.fig.add_annotation(
                x=0,
                y=i + 0.4,
                text=f"<span style=''>{metric_name}: {int(ser_plot[col]):.0f}</span>",
                showarrow=False,
                font={
                    "color": rgb_to_color(self.white),
                    "family": "Gilroy-Light",
                    "size": 12 * self.font_size_multiplier,
                },
            )

    def add_person(self, person: Person, n_group, metrics):
        # Make list of all metrics with _Z and _Rank added at end
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]

        self.add_data_point(
            ser_plot=person.ser_metrics,
            plots="_Z",
            name=person.name,
            hover="_Ranks",
            hover_string="Rank: %{customdata}/" + str(n_group),
        )

    def add_persons(self, persons: PersonStat, metrics):

        # Make list of all metrics with _Z and _Rank added at end
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]

        self.add_group_data(
            df_plot=persons.df,
            plots="_Z",
            names=persons.df["name"],
            hover="_Ranks",
            hover_string="Rank: %{customdata}/" + str(len(persons.df)),
            legend=f"Other persons  ",
        )

    def add_title_from_person(self, person: Person):
        self.person = person
        title = f"Evaluation of {person.name}"
        subtitle = f"Based on Big Five scores"
        self.add_title(title, subtitle)


"""class ViolinPlot(Visual):
    def violin(data, point_data):
        # Create a figure object
        fig = go.Figure()

        # Labels for the columnshover
        labels = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']

        # Loop through each label to add a violin plot trace
        for label in labels:
            fig.add_trace(go.Violin(
                x=df_plot[label],  # Use x for the data
                name=label,      # Label each violin plot correctly
                box_visible=True,
                meanline_visible=True,
                line_color='black',  # Color of the violin outline
                fillcolor='rgba(0,100,200,0.3)',  # Color of the violin fill
                opacity=0.6,
                orientation='h'  # Set orientation to horizontal
            )
        )
        for label, value in point_data.items():
            fig.add_trace(
                go.Scatter(x=[value], y=[label], mode='markers', marker=dict(color='red', size=8, symbol='cross'), name=f'{label} Candidate Point'))

        # Update layout for better visualization
        fig.update_layout(
            title='Distribution of Personality Traits',
            xaxis_title='Score',  
            yaxis_title='Trait',
            xaxis=dict(range=[0, 40]),
            violinmode='overlay', 
            showlegend=True)

        # Display the plot in Streamlit
        st.plotly_chart(fig)


    def radarPlot(Visual):
        # Data import
        data_r = data_p.to_list()  
        labels = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
        df = pd.DataFrame({'data': data_r,'label': labels})
    
        # Create the radar plot
        fig = px.line_polar(df, r='data', theta='label', line_close=True, markers=True)
        fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0, 40])),showlegend=True, title= 'Candidate profile')
        fig.update_traces(fill='toself', marker=dict(size=5))
        # Display the plot in Streamlit
        st.plotly_chart(fig)"""


class DistributionPlotRuns(Visual):
    """
    Creates a distribution plot for player run metrics.
    """
    def __init__(self, metrics, *args, **kwargs):
        """
        Initialize the distribution plot for player runs.

        Args:
            metrics (list): List of metrics to visualize.
        """
        self.metrics = metrics
        self.marker_color = (
            c for c in [Visual.white, Visual.bright_yellow, Visual.bright_blue]
        )
        self.marker_shape = (s for s in ["square", "hexagon", "diamond"])
        super().__init__(*args, **kwargs)
        self._setup_axes()

    def _setup_axes(self, labels=["Worse", "Average", "Better"]):
        """
        Set up the x and y axes for the plot.
        """
        self.fig.update_xaxes(
            range=[-4, 4],
            fixedrange=True,
            tickmode="array",
            tickvals=[-3, 0, 3],
            ticktext=labels,
        )
        self.fig.update_yaxes(
            showticklabels=False,
            fixedrange=True,
            gridcolor=rgb_to_color(self.medium_green),
            zerolinecolor=rgb_to_color(self.medium_green),
        )


    def add_group_data(self, df_plot):
        """
        Add all players' data points to the plot.

        Args:
            df_plot (pd.DataFrame): DataFrame with all player metrics.
        """
        for i, metric in enumerate(self.metrics):
            # Generate hover text with player name and metric value
            hover_text = df_plot.apply(
                lambda row: f"Player: {row['player']}<br>{metric}: {row[metric]:.2f}" if pd.notnull(row[metric]) else f"Player: {row['player']}<br>{metric}: N/A",
                axis=1
            ).tolist()
            
            # Add scatter trace for this metric
            self.fig.add_trace(
                go.Scatter(
                    x=df_plot[f"{metric}_Z"],
                    y=np.ones(len(df_plot)) * i,
                    mode="markers",
                    marker=dict(
                        color=rgb_to_color(self.bright_green, opacity=0.2),
                        size=10
                                                
                    ),
                    hovertext=hover_text,  # Use hover text here
                    name="Other players",
                    showlegend=(i == 0),
                )
            )

            # Add an annotation for the metric title on the left side of each row
            self.fig.add_annotation(
                x=-3,  # Place the annotation outside the plot area on the left
                y=i,
                text=f"<b>{metric.replace('_', ' ').title()}</b>",
                showarrow=False,
                font=dict(
                    color=rgb_to_color(self.white, 0.8),
                    size=12 * self.font_size_multiplier,
                    family="Arial",
                    
                ),
                xref="x",
                yref="y",
                align="right",
                xanchor="right"
            )
        
    def add_player(self, player_metrics, player_name):
        """
        Add a specific player's metrics to the plot.

        Args:
            player_metrics (pd.Series): Player metrics for visualization.
            player_name (str): Name of the player.
        """
        color = next(self.marker_color)
        marker = next(self.marker_shape)

        for i, metric in enumerate(self.metrics):
            self.fig.add_trace(
                go.Scatter(
                    x=[player_metrics[f"{metric}_Z"]],
                    y=[i],
                    mode="markers",
                    marker=dict(
                        color=rgb_to_color(color, opacity=0.7),
                        size=12,
                        symbol=marker,
                        line_width=1.5,
                        line_color=rgb_to_color(color)
                    ),
                    hovertemplate=f"{metric}: {player_metrics[metric]:.2f}",
                    name=player_name,
                    showlegend=(i == 0),
                )
            )

    def add_title_from_player(self, player_name):
        """
        Add a title to the plot based on the player.

        Args:
            player_name (str): Name of the player.
        """
        self.fig.update_layout(
            title={
                "text": f"Run Metrics Distribution for {player_name}",
                "x": 0.5,
                "xanchor": "center",
            }
        )

import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
import matplotlib.font_manager as font_manager
import matplotlib.patheffects as path_effects

class PitchPlot:
    def __init__(self, font_path='data/fonts/futura/futura.ttf', font_path_light='data/fonts/futura/Futura Light font.ttf'):
        self.font_props = font_manager.FontProperties(fname=font_path)
        self.font_props_light = font_manager.FontProperties(fname=font_path_light)
        self.path_eff = [path_effects.Stroke(linewidth=0.5, foreground='black'), path_effects.Normal()]

    def create_pitch_plot(self, selected_player, jersey_number, player_pass_data, player_pass_chance_data, totalPassesComplete, totalPasses, totalShotAssists, totalGoalAssists, totalxA):
        fig = plt.figure(figsize=(8, 12), dpi=300)
        fig.patch.set_facecolor('#f3f3f3')
        
        ax1 = fig.add_axes([0.1, 0.82, 0.9, 0.22])
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')

        ax1.text(0.5, 0.85, f'{selected_player} #{jersey_number}', fontsize=20, fontproperties=self.font_props, fontweight='bold', color='black', ha='center')
        ax1.text(0.5, 0.72, 'Chances Created, at Euro 2024', fontsize=14, fontproperties=self.font_props, color='black', ha='center')
        
        yPos1 = 0.55
        yPos2 = yPos1 + 0.02
        
        key_items = [
            {'x_text': 0.15, 'y_text': yPos1, 'text': 'Pass Key:', 'fontprops': self.font_props},
            {'x_text': 0.3, 'y_text': yPos1, 'text': 'Shot Assists', 'fontprops': self.font_props_light, 'x_scatter': 0.4, 'y_scatter': yPos2, 'color': 'seagreen'},
            {'x_text': 0.55, 'y_text': yPos1, 'text': 'Goal Assists', 'fontprops': self.font_props_light, 'x_scatter': 0.65, 'y_scatter': yPos2, 'color': 'goldenrod'}
        ]
        
        for item in key_items:
            self.add_key_item(ax1, item)
        
        ax2 = fig.add_axes([0.05, 0.4, 0.95, 0.6])
        pitch = VerticalPitch(pitch_type='statsbomb', line_zorder=2, half=True, pitch_color='#0C0D0E', line_color='black', linewidth=.75, axis=True, label=True)
        pitch.draw(ax=ax2)
        ax2.axis('off')
        
        bin_statistic = pitch.bin_statistic(
            player_pass_data['location'].str[0],
            player_pass_data['location'].str[1],
            statistic='count', bins=(12, 8), normalize=False)
        pcm = pitch.heatmap(bin_statistic, cmap='Reds', edgecolor='dimgrey', ax=ax2, alpha=0.7)
        
        self.plot_events(ax2, pitch, player_pass_chance_data, jersey_number)
        
        ax_cbar = fig.add_axes([0.1, 0.405, 0.39, 0.02])
        cbar = plt.colorbar(pcm, cax=ax_cbar, orientation='horizontal')
        cbar.outline.set_edgecolor('black')
        cbar.ax.xaxis.set_tick_params(color='black')
        
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='black')
        
        max_value = np.max(bin_statistic['statistic'])
        ax_cbar.text(x=max_value/2, y=1.4, s='Completed Passes', fontsize=12,
            fontproperties=self.font_props, color='black', ha='center')
        
        yPos1 = 52.7
        yPos2 = yPos1 - 3.7
        
        stats = [
            {'x': 38, 'y1': yPos1, 'y2': yPos2, 'title': 'Passes', 'value': f"{totalPassesComplete}/{totalPasses}", 'color': 'red'},    
            {'x': 50, 'y1': yPos1, 'y2': yPos2, 'title': 'Shot Asts.', 'value': f"{totalShotAssists}", 'color': 'seagreen'},
            {'x': 61, 'y1': yPos1, 'y2': yPos2, 'title': 'Goal Asts.', 'value': f"{totalGoalAssists}", 'color': 'goldenrod'},
            {'x': 73.5, 'y1': yPos1, 'y2': yPos2, 'title': 'xA', 'value': f"{totalxA:.2f}", 'color': 'black'},
        ]
        
        for stat in stats:
            ax2.text(stat['x'], stat['y1'], stat['title'], fontsize=12, fontproperties=self.font_props, color='black', ha='left')
            ax2.text(stat['x'], stat['y2'], stat['value'], fontsize=12, fontproperties=self.font_props, color=stat['color'], ha='left')
        
        return fig
    
    def add_key_item(self, ax, item):
        ax.text(item['x_text'], item['y_text'], item['text'], fontsize=12, fontproperties=item['fontprops'], color='black', ha='left')
        if 'x_scatter' in item:
            ax.scatter(item['x_scatter'], item['y_scatter'], s=150, color=item['color'], edgecolor=item['color'], linewidth=0.8)
            ax.arrow(item['x_scatter'], item['y_scatter'], dx=0.075, dy=0, width=0.02, head_length=0.02, color=item['color'], linewidth=1)
            ax.text(item['x_scatter'], item['y_scatter']-0.009, s="19", fontsize=10, fontproperties=item['fontprops'], path_effects=self.path_eff, color='black', ha='center', va='center')
    
    def plot_events(self, ax, pitch, events, jersey_number):
        for x in events.to_dict(orient='records'):
            color = 'goldenrod' if x['pass_goal_assist'] == True else 'seagreen'
            pitch.scatter(x['location'][0], x['location'][1], s=125, color=color, ax=ax, linewidth=0.8, edgecolor=color)
            pitch.arrows(x['location'][0], x['location'][1], x['pass_end_location'][0], x['pass_end_location'][1], color=color, ax=ax, width=2, headwidth=5, headlength=5, pivot='tail')
            pitch.annotate(jersey_number, xy=(x['location'][0]-0.1, x['location'][1]), ax=ax, fontsize=8, fontproperties=self.font_props_light, path_effects=self.path_eff, color='black', ha='center', va='center')
   
    def create_recipient_stats_table(self, recipient_stats):
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
            .set_table_styles([
                {'selector': 'thead th',
                 'props': [('background-color', 'red'),
                           ('opacity', '0.7'),
                           ('color', 'black'),
                           ('font-weight', 'bold'),
                           ('text-align', 'center')]}
            ])\
            .format({"Jersey #": "#{:.0f}", "Total xA": "{:.2f}"})
        
        return styled_table
    
    def calculate_player_metrics(self, players_with_assists):
        player_metrics = players_with_assists.groupby(['player_name', 'team']).agg(
            games_played=('match_id', 'nunique'),
            passes_completed=('pass_outcome', lambda x: x.isna().sum()),
            total_passes=('pass_outcome', lambda x: len(x)),
            passes_complete_perc=('pass_outcome', lambda x: (x.isna().sum() / len(x)) * 100),  
            chances_created=('pass_shot_assist', 'sum'),
            goal_assists=('pass_goal_assist', 'sum'),
            xA=('xA', 'sum'),
            avg_pass_angle=('pass_angle', lambda x: abs(x).mean())
        ).reset_index()
        
        player_metrics['games_played'] = player_metrics['games_played'].astype(int)        
        player_metrics['goal_assists'] = player_metrics['goal_assists'].astype(int)
        player_metrics['chances_created'] += player_metrics['goal_assists']
        
        player_metrics['avg_pass_angle'] = 3.14 - player_metrics['avg_pass_angle']
        
        player_metrics = player_metrics[player_metrics['goal_assists'] > 0].reset_index()
        return player_metrics
    
    def create_distribution_plot(self, player_metrics, selected_player):
        fig = go.Figure()
        metrics = ["games_played", "passes_completed", "passes_complete_perc", "chances_created", "goal_assists", "xA", "avg_pass_angle"]
        
        for metric in metrics:
            fig.add_trace(go.Box(y=player_metrics[metric], name=metric, marker_color='lightgray'))
        
        selected_player_metrics = player_metrics[player_metrics['player_name'] == selected_player]
        
        for metric in metrics:
            fig.add_trace(go.Scatter(y=[selected_player_metrics[metric].values[0]], x=[metric],
                                     mode='markers', marker=dict(color='red', size=10),
                                     name=f"{selected_player}"))
        
        fig.update_layout(
            title_text=f"{selected_player}'s Performance Distribution",
            yaxis_title="Metric Values",
            xaxis_title="Metrics",
            template="plotly_white"
        )
        
        return fig

import plotly.express as px
from collections import defaultdict
import textwrap

class EuroPassVisualizer:
    def __init__(self, df):
        self.df = df
        self.team_colors = self._generate_team_colors()
    
    def _generate_team_colors(self):
        """Assigns unique colors to each team based on national team colors."""
        team_colors = {
            'England': '#00247D', 'Spain': '#FFCC00', 'France': '#0055A4', 'Turkey': '#E30A17',
            'Netherlands': '#FF6600', 'Austria': '#ED2939', 'Denmark': '#C60C30', 'Germany': '#ffffff',
            'Slovakia': '#005BAC', 'Switzerland': '#D52B1E', 'Hungary': '#436F4D', 'Albania': '#E41B17',
            'Croatia': '#FF0000', 'Italy': '#008C45', 'Poland': '#DC143C', 'Ukraine': '#FFD700',
            'Georgia': '#FF0000', 'Romania': '#FFD700', 'Belgium': '#FAAB18', 'Portugal': '#006600',
            'Slovenia': '#0093DD', 'Serbia': '#C63633', 'Scotland': '#002147', 'Czech Republic': '#D7141A'
        }
        return team_colors
    
    def _custom_wrap(self, s, width=30):
        """Wrap text with line breaks for better display in charts."""
        return "<br>".join(textwrap.wrap(s, width=width))
    
    def plot_xa_stacked_barchart(self, selected_player=None):
        """Generates a stacked bar chart showing xA sum for each team and player using Plotly."""
        print(selected_player)
        
        players_with_xA = self.df[self.df['xA'] > 0].copy()
        
        # Add opponent column
        players_with_xA['opponent'] = players_with_xA.apply(
            lambda row: row['home_team'] if row['team'] != row['home_team'] else row['away_team'], axis=1
        )
        
        # Add chances created column, handling NaN values
        players_with_xA['Chances Created'] = (
            players_with_xA['pass_shot_assist'].fillna(0).astype(int) + 
            players_with_xA['pass_goal_assist'].fillna(0).astype(int)
        )
        
        # Add shot description column
        players_with_xA['shot_description'] = players_with_xA.apply(
            lambda row: f"{row['recipient_name']} Vs {row['opponent']} in {row['competition_stage']}, xA: {row['xA']:.2f}", axis=1
        )
        
        players_xa = players_with_xA.groupby(['team', 'player_name'])[['xA', 'Chances Created']].sum().reset_index()
        team_xa = players_with_xA.groupby('team')['xA'].sum().reset_index().sort_values(by='xA', ascending=False)
        
        players_xa = players_xa.rename(columns={'player_name': 'Player', 'team': 'Team'})
        players_xa['Player'] = players_xa['Player'].apply(self._custom_wrap)
        players_xa['color'] = players_xa['Team'].map(self.team_colors)
        
        # Rank players by xA
        players_xa['Overall Rank'] = players_xa['xA'].rank(method='min', ascending=False).astype(int)
        players_xa['Team Rank'] = players_xa.groupby('Team')['xA'].rank(method='min', ascending=False).astype(int)
        
        # Retrieve ranks for selected player
        team_rank, overall_rank = None, None
        if selected_player:
            selected_data = players_xa[players_xa['Player'] == selected_player]
            if not selected_data.empty:
                team_rank = selected_data['Team Rank'].values[0]
                overall_rank = selected_data['Overall Rank'].values[0]
        
        # Add shot description column
        players_xa['player_text'] = players_xa.apply(
            lambda row: f"{row['Player']}<br>xA: {row['xA']:.2f}", axis=1
        )
        
        # Sort players by xA so highest is at the bottom of the stack
        players_xa = players_xa.sort_values(by='xA', ascending=False)
        
        with st.expander(f"Euro Passes xA Bar chart Dataframe "):
            st.write(players_xa)
        
        # Default selected team is the team of the selected player
        default_team = None
        if selected_player:
            default_team = players_xa.loc[players_xa['Player'] == selected_player, 'Team'].values[0]
        
        selected_teams = st.multiselect(
            "Select teams:", options=team_xa['team'].tolist(), default=[default_team] if default_team else []
        )
        
        filtered_data = players_xa[players_xa['Team'].isin(selected_teams)]
        
        pattern_shape_sequence = []
        pattern_shape_sequence = ["" if player != selected_player else "/" for player in filtered_data['Player']]
        
        
        
        fig = px.bar(
            filtered_data, 
            x='Team', 
            y='xA', 
            color='Team', 
            text='player_text', 
            color_discrete_map=self.team_colors,
            orientation='v',
            height=500,
            hover_name='Player',
            hover_data={'Player': False, 'xA': True, 'Team': False, 'Chances Created': True, 'player_text': False, 'Team Rank': True, 'Overall Rank': True},
            pattern_shape='Player',
            pattern_shape_sequence=pattern_shape_sequence
        )
        
        fig.update_traces(marker=dict(line_width=1.5, line_color="grey"), textposition="inside")
        
        fig.update_layout(barmode='stack', xaxis_title='Teams', yaxis_title='Expected Assists (xA)', uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False)
        
        
        st.plotly_chart(fig)
        
        return team_rank, overall_rank
    
    def summarize_xa_insights(self, selected_player):
        """Summarizes key insights from xA rankings."""
        players_with_xA = self.df[self.df['xA'] > 0].copy()
        
        players_xa = players_with_xA.groupby(['team', 'player_name'])[['xA']].sum().reset_index()
        
        # Rank players by xA
        players_xa['Overall Rank'] = players_xa['xA'].rank(method='min', ascending=False).astype(int)
        players_xa['Team Rank'] = players_xa.groupby('team')['xA'].rank(method='min', ascending=False).astype(int)
        
        # Retrieve ranks for selected player
        selected_data = players_xa[players_xa['player_name'] == selected_player]
        if not selected_data.empty:
            team_rank = selected_data['Team Rank'].values[0]
            overall_rank = selected_data['Overall Rank'].values[0]
        else:
            return {"xA_insights": f"No xA data available for {selected_player}."}
        
        xA_insights = (
            f"{selected_player} ranks **{team_rank}** in their team and **{overall_rank}** overall for expected assists (xA)."
        )
        
        return xA_insights


from plotly.subplots import make_subplots

class DistributionPlotPasses(Visual):
    """
    Creates a distribution plot for player pass metrics with two subplots.
    """
    def __init__(self, metrics, *args, **kwargs):
        """
        Initialize the distribution plot with two sections.
        
        Args:
            metrics (list): List of metrics to visualize.
        """
        self.metrics = metrics
        self.counted_stats = ["games_played", "passes_completed", "chances_created", "goal_assists"]
        self.calculated_stats = ["passes_complete_perc", "avg_pass_angle", "xA"]
        
        self.marker_color = (c for c in [Visual.white, Visual.bright_yellow, Visual.bright_blue])
        self.marker_shape = (s for s in ["square", "hexagon", "diamond"])
        
        super().__init__(*args, **kwargs)

        # Correctly create subplots to allow row and column referencing
        self.fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Counted Stats", "Calculated Stats"),
            shared_xaxes=True,
            vertical_spacing=0.15
        )
        
        self._setup_axes()
        
        self.fig.update_layout(
            paper_bgcolor=rgb_to_color(self.bg_gray),
            plot_bgcolor=rgb_to_color(self.bg_gray),
            legend=dict(
                orientation="h",
                font={"color": rgb_to_color(self.black)},
                x=0.5,
                xanchor="center"                
            ),
            xaxis=dict(
                tickfont={"color": rgb_to_color(self.black, 0.5)}
            )
        )
    
    def _setup_axes(self, labels=["Worse", "Average", "Better"]):
        """
        Set up the x and y axes for the plot.
        """
        for i in range(1, 3):  # 1 = counted stats, 2 = calculated stats
            self.fig.update_xaxes(
                range=[-10.5, 10.5],
                fixedrange=True,
                tickmode="array",
                tickvals=[-8, 0, 8],
                ticktext=labels,
                tickfont=dict(color=rgb_to_color(self.black)),
                row=i, col=1
            )
            self.fig.update_yaxes(
                showticklabels=False,
                fixedrange=True,
                gridcolor=rgb_to_color(self.light_gray),
                zerolinecolor=rgb_to_color(self.light_gray),
                row=i, col=1
            )


    def add_group_data(self, df_plot):
        """
        Add all players' data points to the plot.

        Args:
            df_plot (pd.DataFrame): DataFrame with all player metrics.
        """
        for i, metric in enumerate(self.metrics):
            formatted_metric = metric.replace('_', ' ').title()
            if metric == 'xA':
                formatted_metric = 'xA'
            elif metric == 'passes_complete_perc':
                formatted_metric = 'Passes Completed %'
            elif metric == 'avg_pass_angle':
                formatted_metric = 'π - |Avg. Pass Angle|'

            # Assign metric to correct subplot row
            row = 1 if metric in self.counted_stats else 2

            hover_text = df_plot.apply(
                lambda row: f"Player: {row['player_name']}<br>{formatted_metric}: {row[metric]:.2f}" 
                if pd.notnull(row[metric]) else f"Player: {row['player_name']}<br>{formatted_metric}: N/A",
                axis=1
            ).tolist()

            # Add scatter trace to the correct row
            self.fig.add_trace(
                go.Scatter(
                    x=df_plot[f"{metric}_Z"],
                    y=np.ones(len(df_plot)) * i,
                    mode="markers",
                    marker=dict(color=rgb_to_color(self.table_red, opacity=0.4), size=10),
                    hovertext=hover_text,
                    name="Other players",
                    showlegend=(i == 0),
                ),
                row=row, col=1  # Ensuring traces are assigned to the correct subplot
            )

            # Add metric annotation
            self.fig.add_annotation(
                x=0,  # Shift annotation outside the plot
                y=i+0.5,
                text=f"<b>{formatted_metric}</b>",
                showarrow=False,
                font=dict(color=rgb_to_color(self.black, 0.8), size=12 * self.font_size_multiplier, family="Arial"),
                xref="x",
                yref="y",
                align="center",
                xanchor="center",
                row=row, col=1
            )
        
    def add_player(self, player_metrics, player_name):
        """
        Add a specific player's metrics to the plot.

        Args:
            player_metrics (pd.Series): Player metrics for visualization.
            player_name (str): Name of the player.
        """
        color = next(self.marker_color)
        marker = next(self.marker_shape)

        for i, metric in enumerate(self.metrics):
            row = 1 if metric in self.counted_stats else 2

            self.fig.add_trace(
                go.Scatter(
                    x=[player_metrics[f"{metric}_Z"]],
                    y=[i],
                    mode="markers",
                    marker=dict(
                        color=rgb_to_color(self.gray, opacity=0.7),
                        size=12,
                        symbol=marker,
                        line_width=1.5,
                        line_color=rgb_to_color(self.black)
                    ),
                    hovertemplate=f"{metric}: {player_metrics[metric]:.2f}",
                    name=player_name,
                    showlegend=(i == 0),
                ),
                row=row, col=1  # Assign trace to correct subplot
            )

    def add_title_from_player(self, player_name):
        """
        Add a title to the plot based on the player.

        Args:
            player_name (str): Name of the player.
        """
        self.fig.update_layout(
            title={
                "text": f"Pass Metrics Distribution for {player_name}",
                "x": 0.5,
                "xanchor": "center",
            }
        )

from mplsoccer import Pitch
from matplotlib.lines import Line2D
import math

class ShotMap(Visual):    
    """
    Class to visualize shot data using Streamlit and mplsoccer.
    """
    def __init__(self, shot_data):
        """
        Initialize with shot data.
        
        Parameters:
            shot_data (DataFrame): Data containing shot locations and player/team info.
        """
        self.shot_data = shot_data
        self.shot_data['color'] = np.where(self.shot_data['Goal'], 'green', 
                                 np.where(self.shot_data['shot_on_target'], 'yellow', 'red'))
        self.model_variables = ['shot_Angle_value', "b_intercept", 'shot_Distance_value', 'shot_X_value', 'shot_C_value', 'shot_D2_value', 'shot_AX_value', 'shot_preferred_foot_value', 'shot_in_box_value']
        self.op_columns = ["b_intercept", "shot_Angle_value", "shot_b_Angle", "shot_Distance_value", "shot_b_Distance", "shot_X_value", "shot_b_X", "shot_C_value", "shot_b_C", "shot_D2_value", "shot_b_D2", "shot_AX_value", "shot_b_AX", "shot_preferred_foot_value", "shot_b_preferred_foot", "shot_in_box_value", "shot_b_in_box"]
        self.fk_columns = ["b_intercept", "shot_Angle_value", "shot_b_Angle", "shot_Distance_value", "shot_b_Distance", "shot_X_value", "shot_b_X", "shot_C_value", "shot_b_C", "shot_D2_value", "shot_b_D2", "shot_AX_value", "shot_b_AX"]
        self.pk_columns = ["b_intercept","shot_Angle_value", "shot_b_Angle", "shot_Distance_value", "shot_b_Distance"]
        

    def filter_data(self, df, team: str, player: str):
        """
        Filters shot data based on selected team and player.
        
        Parameters:
            team (str): Selected team.
            player (str): Selected player.
        
        Returns:
            DataFrame: Filtered shot data.
        """
        
        if team:
            df = df[df['teamName'] == team]
        if player:
            df = df[df['playerName'] == player]
        return df

    # Plotting function
    def plot_shots(self, df, df2, ax, pitch):
        
        # Plot the scatter plot of shots
        pitch.scatter(df['X'], df['Y'], ax=ax, s=400*df['xG'].apply(math.sqrt), c=df['color'], edgecolors='black', alpha=1, zorder=2)
        
        # Plot the scatter plot of shots
        pitch.scatter(df2['X'], df2['Y'], ax=ax, s=400*df2['xG'].apply(math.sqrt), c='dimgrey', alpha=0.2, zorder=1.5)
    
        # Ensure the columns are all the same length and drop NaNs
        # df = df.dropna(subset=['oldX', 'Y'])  # Drop rows where any of the key columns are missing
        
        # Create arrays for xend and yend with the same length as df
        xend = np.full_like(df['X'], 105)  # Assuming pitch length is 105
        yend_top = np.full_like(df['Y'], 34 + 7.32/2)  # Top corner of the goal
        yend_bottom = np.full_like(df['Y'], 34 - 7.32/2)  # Bottom corner of the goal

        # Plot arrows to the top corner of the goal
        pitch.arrows(
            df['X'], 
            df['Y'],
            xend,
            yend_top, 
            color='black', 
            alpha=.95,
            ax=ax,
            width=2,
            linewidth=2,
            headlength=0,
            headwidth=1
        )

        # Plot arrows to the bottom corner of the goal
        pitch.arrows(
            df['X'], 
            df['Y'],
            xend,
            yend_bottom, 
            color='black', 
            alpha=.95,
            ax=ax,
            width=2,
            linewidth=2,
            headlength=0,
            headwidth=1
        )  
    
    def display_shot_map(self):
        """
        Displays the shot map in Streamlit.
        """
        st.subheader("Wyscout 2017/18 England Shot Map")
        st.subheader("Filter to any team/player to see all their shots taken!")
        
        #df = self.shot_data[['teamName', 'playerName', 'X', 'Y', 'xG', 'shot_on_target', 'Goal', 'color', 'shot_label']].reset_index(drop=True)
        df = self.shot_data
        
        DEFAULT = '< PICK A VALUE >'
        
        team = st.selectbox("Select a team", df.sort_values(by='xG', ascending=False)['teamName'].unique(), index=2)
        player = st.selectbox("Select a player", 
                      df[df['teamName'] == team].sort_values(by='xG', ascending=False)['playerName'].unique(), 
                      placeholder=DEFAULT)

        # team = st.selectbox("Select a team", df['teamName'].sort_values(by='xG', ascending=False).unique(), placeholder=DEFAULT)
        # player = st.selectbox("Select a player", df[df['teamName'] == team]['playerName'].sort_values(by='xG', ascending=False).unique(), placeholder=DEFAULT)
        
        filtered_df = self.filter_data(df, team, player)

        filtered_df = filtered_df.sort_values(by='xG', ascending=False)
        
        
        # Multiselect for individual shots
        shotLabels = st.multiselect(
            "Select shot(s)", 
            filtered_df['shot_label'].tolist(), 
            placeholder="Choose one or more shots",
            default=filtered_df['shot_label'].iloc[0]
        )
        
        # If shots are selected, filter the dataframe to only those shots
        if shotLabels:
            selected_shots_df = filtered_df[filtered_df['shot_label'].isin(shotLabels)]
        else:
            selected_shots_df = filtered_df

        # Calculate total shots, shots on target, and goals
        total_shots = len(filtered_df)
        total_on_target = filtered_df['shot_on_target'].sum()
        total_goals = filtered_df['Goal'].sum()
        
        # Draw the pitch and plot the filtered shots
        pitch = VerticalPitch(pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder=1, pitch_color='#aabb97', stripe_color='#c2d59d', stripe=True, 
                            line_color='white', half=True, axis=True, label=True)
        fig, ax = pitch.draw(figsize=(10, 10))
        self.plot_shots(selected_shots_df, filtered_df, ax, pitch)

        # Add custom legend for the colors
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Goal (Green)', markerfacecolor='green', markersize=10, markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', label='On Target (Yellow)', markerfacecolor='yellow', markersize=10, markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', label='Off Target (Red)', markerfacecolor='red', markersize=10, markeredgecolor='black')
        ]

        # Add the legend to the plot
        ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.125), ncol=3, fontsize=12)
        
        # Add text annotations for total shots, on target shots, and goals
        fig.text(0.25, 0.9, f'Total Shots: {total_shots}', ha='center', fontsize=14)
        fig.text(0.5, 0.9, f'Total Shots on Target: {total_on_target:.0f}', ha='center', fontsize=14)
        fig.text(0.75, 0.9, f'Total Goals: {total_goals}', ha='center', fontsize=14)
        
        st.pyplot(fig)
        
        # Function to show how xG is calculated

               
        # Show how xG is calculated for each selected shot
        if shotLabels:
            st.subheader("xG Calculation Breakdown for Selected Shots:")
            st.latex(r'''xG =  \frac{1}{1 + e^{(\beta_0 + \beta_1 x_1 + \cdots + \beta_i x_i )}}''')
            for idx, shot_label in enumerate(shotLabels):
                shot_row = filtered_df[filtered_df['shot_label'] == shot_label].iloc[0]
                calculation = self.show_xg_calculation(shot_row)
                st.write(f"**Shot {idx + 1}:** {shot_label} - {shot_row['shot_type']}")
                st.markdown(calculation)
            st.text("Note: Angle is in Radians, X is horizontal distance from goal,\nC is distance from centre of the pitch, AX is Angle multipled by X,\nDA is distance time angle, preferred foot is found using players dataframe")
        else:
            st.write("No shots selected.")
        
    def show_xg_calculation(self, row):
        """
        Generates a readable breakdown of xG calculation for a selected shot.
        """
        shot_type = row['shot_type']
        if shot_type in ["Open Play - Header", "Open Play - Non-header"]:
            columns = self.op_columns
        elif shot_type == "Free Kick":
            columns = self.fk_columns
        elif shot_type == "Penalty":
            columns = self.pk_columns
        else:
            return "Unknown shot type."
    
        def format_variable_name(variable):
            parts = variable.split("_")
            if len(parts) > 2:
                return " ".join(parts[1:-1])
            return variable
        
        terms = [f"{row.get('b_intercept', 0):.4f} (Intercept)"]
        for i in range(1, len(columns), 2):
            if i + 1 < len(columns) and columns[i] in row and columns[i + 1] in row:
                variable_name = format_variable_name(columns[i])
                terms.append(f"+ {row[columns[i + 1]]:.2f} * {variable_name} ({row[columns[i]]:.2f})")
        # terms.append(f" = {row['xG']:.4f}")       
        terms = " ".join(terms)
        
        calculation = f"xG = 1 / 1 + $exp(${terms}$)$"
        calculation = calculation + f" = {row['xG']:.2f}"
        return calculation
        
        
class ThreatMap(Visual):
    """
    Class to visualize possession chains that end in shots and calculate expected threat (xT).
    """
    def __init__(self, possession_data):
        """
        Initialize with possession chain data.
        
        Parameters:
            possession_data (DataFrame): Data containing possession chains and xT metrics.
        """
        self.possession_data = possession_data

    def display_threat_map(self):
        """
        Displays the possession chain visualization and expected threat metrics in Streamlit.
        """
        # st.title("Expected Threat Calculation xT")
        st.subheader("Visualising possession chains to show how expected threat is in use")
        
        
        
        
        df = self.possession_data[['playerName', 'team', 'eventName', 'subEventName', 'positions', 'xG', 'xT', 'possession_chain', 'has_shot', 'has_goal', 'shot_assist', 
                                   'xT_pre_assist', 'xT_shot_assist', 'xT_shot_assist_diff', 'xT_post_assist', 'xG_pre_shot', 'xG_diff', 'xCarry_pre_shot', 'possession_chain_label',
                                   'xG_pred', 'shot_prob' ]]
        
        typesOfPassesLst = df[df['eventName'] == 'Pass']['subEventName'].unique()
        typesOfPasses = ", ".join(typesOfPassesLst)
        typesOfDribblesLst = df[df['eventName'].isin(['Duel', 'Others on the ball'])]['subEventName'].unique()
        typesOfDribbles = ", ".join(typesOfDribblesLst)
        st.text("xT is assigned to Passes or Dribbles in a possession chain, where the probability that action leads to a shot multiplied by the xG predicted from that shot. \n"
                "Linear regression is used to find the probability of shot and the predicted xG value. These are muliplied together to get xT.\n\n"
                "Features affecting Passes and Dribbles include position and type.\n"
                f"Types of Passes which include: {typesOfPasses}. And Types of Dribbles which include: {typesOfDribbles}.")
        
        
        sorted_chains = df[df['eventName'] == 'Shot'].sort_values(by='xT_pre_assist', ascending=False)
        possession_chain_options = sorted_chains['possession_chain'].unique()
        
        chain_xG = df[df['eventName'] == 'Shot'].groupby('possession_chain')['xG'].sum().to_dict()
        chain_xT = df.groupby('possession_chain')['xT'].sum().to_dict()
        chain_team = df[df['eventName'] == 'Shot'].groupby('possession_chain')['team'].first().to_dict()
        chain_label = df[df['eventName'] == 'Shot'].groupby('possession_chain')['possession_chain_label'].first().to_dict()
        
        
        possession_chain = st.selectbox(
            "Select possession chain",
            possession_chain_options,
            format_func=lambda x: f"Chain {x}, xG: {chain_xG.get(x, 0):.3f}, xT: {chain_xT.get(x, 0):.2f}, {chain_team.get(x, '')} {chain_label.get(x, '')}",
        )

        selected_pc_df = df[df['possession_chain'] == possession_chain]

        unique_players = selected_pc_df['playerName'].dropna().unique()
        total_xG = selected_pc_df[selected_pc_df['eventName'] == 'Shot']['xG'].sum()
        total_xT_passes = selected_pc_df[selected_pc_df['eventName'] == 'Pass']['xT'].sum()
        total_xT_dribbles = selected_pc_df[selected_pc_df['eventName'].isin(['Duel', 'Others on the ball'])]['xT'].sum()
        
        st.subheader("Possession Chain Pitch Visualisation")
        self.plot_possession_chain(selected_pc_df)
    
        self.display_player_data(selected_pc_df, unique_players)        
        st.write(f"**xG from Shot**: {total_xG:.2f}, **Total xT from Passes**: {total_xT_passes:.2f}, **Total xT from Dribbles**: {total_xT_dribbles:.2f}")
       
    def display_player_data(self, chain_df, unique_players_list):
        """
        Displays player involvement in the possession chain.
        """
        # chain_df = chain_df[chain_df['eventName'] != 'Shot']
        player_count = chain_df['playerName'].nunique()
        title = "Player Involvement" if player_count == 1 else "Players Involvement"
        st.subheader(title)
        st.write(f"**Players Involved**: {', '.join(unique_players_list)}")
        
        display_df = chain_df[['playerName', 'eventName', 'subEventName', 'xG_pred', 'shot_prob', 'xT', 'xG']].reset_index(drop=True)
        display_df.columns = ['Player Name', 'Event', 'Sub Event Name','xG Prediction', 'Shot Probability', 'xT', 'xG']
        display_df.index = display_df.index + 1
        
        st.dataframe(display_df)
    
    def plot_possession_chain(self, chain_df):
        """
        Plots the possession chain that leads to a shot.
        """
        pitch = Pitch(pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder=1, pitch_color='#aabb97',
                      stripe_color='#c2d59d', stripe=True, line_color='white', half=False, axis=True, label=True)
        fig, ax = pitch.draw(figsize=(10, 10))
        
        for _, row in chain_df.iterrows():
            
            if _ == 0:
                pitch.scatter(
                    row.positions[0]['x'] * 105 / 100, 
                    row.positions[0]['y'] * 68 / 100, 
                    ax=ax, 
                    s=500,
                    c='yellow',
                    edgecolors='black'
                )
        
            
            
            start_x = row.positions[0]['x'] * 105 / 100
            start_y = row.positions[0]['y'] * 68 / 100
            end_x = row.positions[1]['x'] * 105 / 100 if len(row.positions) > 1 else start_x
            end_y = row.positions[1]['y'] * 68 / 100 if len(row.positions) > 1 else start_y

            if row.eventName == 'Pass':
                color = 'blue'
                label = f"{row.playerName}\n{row.eventName} - {row.subEventName} ({row.xT:.3f})"
            elif row.eventName in ['Duel', 'Others on the ball']:
                color = 'red'
                label = f"{row.playerName}\n{row.subEventName} ({row.xT:.2f})"
            elif row.eventName == 'Shot':
                color = 'green' if row.has_goal == 1 else 'black'
                end_x = 105
                end_y = 34
                label = f"{row.playerName} ({row.xG:.2f})"
            
            pitch.arrows(start_x, start_y, end_x, end_y, color=color, ax=ax, width=2, headlength=5, headwidth=5)
            ax.text((start_x + end_x) / 2, (start_y + end_y) / 2, label, fontsize=8, color='white', ha='center', va='center',
                    bbox=dict(facecolor=color, edgecolor='none', alpha=0.6, boxstyle='round,pad=0.2'))

        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='Pass'),
            Line2D([0], [0], color='red', lw=2, label='Duel/Touch'),
            Line2D([0], [0], color='green', lw=2, label='Goal'),
            Line2D([0], [0], color='black', lw=2, label='Shot')
        ]
        ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.125), ncol=4, fontsize=12)

        st.pyplot(fig)

from sklearn.preprocessing import MinMaxScaler


class AttackerRadarAndDistPlot(Visual): 
    
    def __init__(self, player_data):
        self.player_data = player_data[player_data['total_goals'] > 0]
        self.categories = ['total_goals', 'total_xG', 'total_xT_received', 'total_xT_passes', 'total_xT_dribbles']
        self.category_labels = {
            'total_goals': 'Goals',
            'total_xG': 'xG',
            'total_xT_received': 'xT received',
            'total_xT_passes': 'xT passes',
            'total_xT_dribbles': 'xT dribbles'
        }

        self.metrics = list(reversed(self.categories))
        self.display_names = self.category_labels

        # Normalize for radar
        self.normalised_data = player_data.copy()
        scaler = MinMaxScaler()
        self.normalised_data[self.categories] = scaler.fit_transform(self.normalised_data[self.categories])
        self.normalised_data['average_rank'] = self.normalised_data[self.categories].rank(ascending=False).mean(axis=1)

        # Z-score for distribution
        self.zscore_data = player_data.copy()
        for m in self.metrics:
            self.zscore_data[f"{m}_norm"] = (self.zscore_data[m] - self.zscore_data[m].mean()) / self.zscore_data[m].std()

        # Setup polar subplot for distribution
        self.marker_color = (c for c in [Visual.white, Visual.bright_yellow, Visual.bright_blue])
        self.marker_shape = (s for s in ["square", "hexagon", "diamond"])

    def _setup_axes(self):
        self.fig.update_xaxes(
            #title_text="Z-Score",
            range=[-1.5, 11.5],
            fixedrange=True,
            tickmode="array",
            tickvals=[-1, 5, 11],
            ticktext=["Worse", "Average", "Better"],
            tickfont=dict(color=rgb_to_color(self.black)),
            # showline=False, 
            row=1, col=1)
        
        self.fig.update_yaxes(
            showticklabels=False, 
            showgrid=False, 
            zeroline=False,
            fixedrange=True, 
            gridcolor=rgb_to_color(self.light_gray),
            zerolinecolor=rgb_to_color(self.light_gray),
            row=1, col=1)

    def display_player_radar(self):
        df = self.normalised_data
        z_df = self.zscore_data

        team = st.selectbox("Select a team:", df.sort_values(by='total_goals', ascending=False)['team'].unique(), index=5)
        player_name = st.selectbox("Select a player:", 
                      df[df['team'] == team].sort_values(by='total_xG', ascending=False)['playerName'].unique() 
                      )

        player_raw = self.player_data[self.player_data['playerName'] == player_name].iloc[0]
        player_normalized = df[df['playerName'] == player_name].iloc[0]

        # Radar Data
        display_labels = [self.category_labels[cat] for cat in self.categories]
        raw_values = [player_raw[cat] for cat in self.categories]
        norm_values = [player_normalized[cat] for cat in self.categories]

        fig = go.Figure()

        fig.add_trace(go.Barpolar(
            r=norm_values,
            theta=display_labels,
            text=[f"Raw: {rv:.2f}<br>Norm: {nv:.2f}" for rv, nv in zip(raw_values, norm_values)],
            hoverinfo='text',
            marker=dict(
                color=norm_values,
                colorscale='Reds',
                cmin=0,
                cmax=1,
                line=dict(color='black', width=1),
                colorbar=dict(
                    # title="Performance",
                    tickvals=[0.05, 0.5, 0.95],
                    ticktext=["Worse", "Average", "Better"],
                    len=1.1,
                    thickness=15,
                    x=0.8,
                    xanchor='center',
                    y=0.5,
                    yanchor='middle'
                )
            ),
            opacity=0.8
            
        ))

        fig.update_layout(
            title=f"Performance Polar Bar Chart - {player_name} (Normalised data)",
            polar=dict(
                radialaxis=dict(range=[0, 1], showticklabels=False, ticks='', showline=False),
                angularaxis=dict(direction="clockwise")
            ),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show stats table
        st.markdown(f"###### Stats Table - {player_name}")
        stats_df = pd.DataFrame({
            'Metric': display_labels,
            'Raw Value': raw_values,
            'Normalized Value': [round(val, 3) for val in norm_values]
        })

        st.dataframe(stats_df, hide_index=True)

        # Distribution Plot
        st.subheader("Comparative Player Distribution Plot")

        self.fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=["Attacker Metric Distribution"],
            shared_xaxes=False
        )
        self._setup_axes()

        # Add group points
        for i, metric in enumerate(self.metrics):
            display_label = self.display_names[metric]
            hover_text = z_df.apply(
                lambda row: (
                    f"Player: {row['playerName']}<br>"
                    f"{display_label}<br>Raw: {row[metric]:.2f}<br>Norm: {row[f'{metric}_norm']:.2f}"
                ),
                axis=1
            )

            self.fig.add_trace(
                go.Scatter(
                    x=z_df[f"{metric}_norm"],
                    y=[i] * len(z_df),
                    mode="markers",
                    marker=dict(color="rgba(200, 30, 30, 0.4)", size=10),
                    hovertext=hover_text,
                    name="Other players",
                    showlegend=(i == 0)
                ),
                row=1, col=1
            )

            self.fig.add_annotation(
                x=5,
                y=i+0.5,
                text=f"<b>{display_label}</b>",
                showarrow=False,
                font=dict(color=rgb_to_color(self.black, 0.8), size=12, family="Arial"),
                xref="x",
                yref="y",
                align="center",
                xanchor="center",
            )

        # Add selected player
        color = next(self.marker_color)
        shape = next(self.marker_shape)
        selected_row = z_df[z_df['playerName'] == player_name].iloc[0]

        for i, metric in enumerate(self.metrics):
            self.fig.add_trace(
                go.Scatter(
                    x=[selected_row[f"{metric}_norm"]],
                    y=[i],
                    mode="markers",
                    marker=dict(
                        color=color,
                        size=14,
                        symbol=shape,
                        line=dict(width=2, color="black")
                    ),
                    hovertemplate=f"{self.display_names[metric]}<br>Raw: {selected_row[metric]:.2f}<br>Norm: {selected_row[f'{metric}_norm']:.2f}",
                    name=player_name,
                    showlegend=(i == 0)
                ),
                row=1, col=1
            )

        self.fig.update_layout(
            title=f"{player_name} vs Others (Z-Score)",
            # height=450,
            # margin=dict(t=60, b=40, l=80, r=30),
            paper_bgcolor=rgb_to_color(self.bg_gray),
            plot_bgcolor=rgb_to_color(self.bg_gray),
            legend=dict(
                orientation="h",
                font={"color": rgb_to_color(self.black)},
                x=0.5,
                xanchor="center"                
            ),
            xaxis=dict(
                tickfont={"color": rgb_to_color(self.black, 0.5)}
            )
        )

        st.plotly_chart(self.fig, use_container_width=True)