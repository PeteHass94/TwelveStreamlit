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
            'Netherlands': '#FF6600', 'Austria': '#ED2939', 'Denmark': '#C60C30', 'Germany': '#000000',
            'Slovakia': '#005BAC', 'Switzerland': '#D52B1E', 'Hungary': '#436F4D', 'Albania': '#E41B17',
            'Croatia': '#FF0000', 'Italy': '#008C45', 'Poland': '#DC143C', 'Ukraine': '#FFD700',
            'Georgia': '#FF0000', 'Romania': '#FFD700', 'Belgium': '#FAAB18', 'Portugal': '#006600',
            'Slovenia': '#0093DD', 'Serbia': '#C63633', 'Scotland': '#002147', 'Czech Republic': '#D7141A'
        }
        return team_colors
    
    def _custom_wrap(self, s, width=10):
        """Wrap text with line breaks for better display in treemap."""
        return "<br>".join(textwrap.wrap(s, width=width))
    
    def plot_xa_treemap(self, selected_player=None):
        """Generates a treemap showing xA sum for each team and player using Plotly."""
        players_with_assists = self.df #[self.df['pass_goal_assist'] > 0]
        players_xa = players_with_assists.groupby(['team', 'player_name'])['xA'].sum().reset_index()
        team_xa = players_with_assists.groupby('team')['xA'].sum().reset_index()
        
        team_xa['color'] = team_xa['team'].map(self.team_colors)
        
        players_xa['color'] = players_xa['team'].map(self.team_colors)
        if selected_player:
            players_xa.loc[players_xa['player_name'] == selected_player, 'color'] = '#808080'  # Highlight selected player in grey
            
        players_xa['player_name'] = players_xa['player_name'].apply(self._custom_wrap)
        
        fig1 = px.treemap(
            team_xa, 
            path=[px.Constant("all"), 'team'], 
            values='xA', 
            color='team', 
            color_discrete_map=self.team_colors,
            height=500
        )
        
        players_xa = players_xa[players_xa['xA'] >= 1]
        
        fig2 = px.treemap(
            players_xa, 
            path=[px.Constant("all"), 'player_name'], 
            values='xA', 
            color='team', 
            color_discrete_map=self.team_colors,
            height=900
        )
        
        fig1.update_traces(root_color="lightgrey", textfont=dict(size=12, family="Arial"))
        fig1.update_layout(margin=dict(t=20, l=5, r=5, b=20), uniformtext=dict(minsize=8, mode='show'))
        
        fig2.update_traces(root_color="lightgrey", textfont=dict(size=12, family="Arial"))
        fig2.update_layout(margin=dict(t=20, l=5, r=5, b=20), uniformtext=dict(minsize=8, mode='show'))
        
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)


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