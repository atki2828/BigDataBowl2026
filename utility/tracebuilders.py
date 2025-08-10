from typing import Callable

import pandas as pd
import plotly.graph_objects as go

from .colors import nfl_colors


def ball_carrier_speed_trace_func(df: pd.DataFrame) -> go.Scatter:
    frame_ids = sorted(df["frameId"].unique())
    x_vals = []
    y_vals = []

    for frame_id in frame_ids:
        current_frame = df[df["frameId"] == frame_id]
        if not current_frame.empty:
            x_vals.append(frame_id)
            y_vals.append(current_frame.iloc[0]["bcs"])

        trace = go.Scatter(
            x=x_vals.copy(),
            y=y_vals.copy(),
            mode="lines+markers",
            line=dict(color="red"),
            marker=dict(size=6),
            name="Ball Carrier Speed",
            showlegend=(True),
        )

        # IMPORTANT: must use same trace index and name across all frames
        trace.uid = "ball_carrier_speed"

    return trace


def gameplay_trace_func(
    frame_df: pd.DataFrame, nfl_colors: dict = nfl_colors
) -> go.Scatter:
    """Generates a Plotly trace for visualizing players and football positions for a single frame.

    Args:
        df (pd.DataFrame): DataFrame containing a single frame of player and football position data.
            Must include 'x', 'y', 'displayName' columns, and for players:
            'club', 'jerseyNumber'.
        nfl_colors (dict, optional): Dictionary mapping NFL team codes to their
            color hex codes. Defaults to predefined nfl_colors.

    Returns:
        go.Scatter: A single Plotly scatter trace containing all players and football positions
    """
    # Split into players and football - should be only one football entry
    is_football = frame_df["displayName"].str.lower() == "football"
    players_df = frame_df[~is_football].reset_index(drop=True)
    football_df = frame_df[is_football].reset_index(drop=True)

    # Player positions and properties
    player_x = players_df["x"].tolist()
    player_y = players_df["y"].tolist()
    player_colors = [nfl_colors.get(club, "#888888") for club in players_df["club"]]
    player_text = [
        str(int(num)) if pd.notna(num) else "" for num in players_df["jerseyNumber"]
    ]

    # Football position - should be single entry
    if not football_df.empty:
        x_positions = player_x + [football_df.iloc[0]["x"]]
        y_positions = player_y + [football_df.iloc[0]["y"]]
        marker_colors = player_colors + ["saddlebrown"]
        marker_sizes = [24] * len(player_x) + [10]
        marker_line_widths = [1] * len(player_x) + [0]
        marker_line_colors = ["white"] * len(player_x) + ["saddlebrown"]
        text = player_text + [""]
    else:
        x_positions = player_x
        y_positions = player_y
        marker_colors = player_colors
        marker_sizes = [24] * len(player_x)
        marker_line_widths = [1] * len(player_x)
        marker_line_colors = ["white"] * len(player_x)
        text = player_text

    trace = go.Scatter(
        x=x_positions,
        y=y_positions,
        mode="markers+text",
        marker=dict(
            color=marker_colors,
            size=marker_sizes,
            line=dict(width=marker_line_widths, color=marker_line_colors),
        ),
        text=text,
        textposition="middle center",
        textfont=dict(color="white", size=10),
        hoverinfo="text",
        showlegend=False,
    )
    trace.name = "gameplay_trace"
    return trace


def ball_carrier_circle_trace_func(frame_df: pd.DataFrame) -> go.Scatter:
    """
    Adds a translucent circle around the ball carrier for a given frame.

    Args:
        frame_df (pd.DataFrame): DataFrame containing a single frame of player data.
            Must include 'x', 'y', 'nflId', 'ballCarrierId' columns.

    Returns:
        go.Scatter: A trace containing a translucent circle around the ball carrier,
                   or None if no ball carrier is found in the frame.
    """
    # Find the ball carrier
    ball_carrier = frame_df[frame_df["nflId"] == frame_df["ballCarrierId"]]

    if ball_carrier.empty:
        return None

    trace = go.Scatter(
        x=[ball_carrier.iloc[0]["x"]],
        y=[ball_carrier.iloc[0]["y"]],
        mode="markers",
        marker=dict(
            size=40,
            color="rgba(255, 0, 0, 0.25)",
            line=dict(width=2, color="red"),
            symbol="circle",
        ),
        name="Ball Carrier",
        hoverinfo="skip",
        showlegend=False,
    )
    trace.name = "ball_carrier_circle_trace"

    return trace


# Closure pattern works nice with any line plot
def build_metric_trace_func(play_df: pd.DataFrame, x_col: str, y_col, name) -> Callable:
    play_df = play_df.sort_values(x_col)

    def trace_func(frame_df: pd.DataFrame) -> go.Scatter:
        fid = int(frame_df["frameId"].iloc[0])
        sub = play_df[play_df["frameId"] <= fid]
        tr = go.Scatter(
            x=sub[x_col].to_numpy(),
            y=sub[y_col].to_numpy(),
            mode="lines",
            name=name,
            showlegend=True,
        )
        tr.uid = "bcs_line"
        return tr

    return trace_func
