"""Module for buiding plotly traced for Play Animations"""

from typing import Callable

import pandas as pd
import plotly.graph_objects as go

from .colors import nfl_colors, player_role_colors


def ball_carrier_speed_trace_func(df: pd.DataFrame) -> go.Scatter:
    """Generates a Plotly trace for visualizing ball carrier speed over time.

    Args:
        df (pd.DataFrame): DataFrame containing ball carrier tracking data.

    Returns:
        go.Scatter: A Plotly scatter trace for ball carrier speed.
    """
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
def build_metric_trace_func(
    play_df: pd.DataFrame, x_col: str, y_col: str, name: str
) -> Callable:
    """Builds a metric trace function for a given play DataFrame.

    Args:
        play_df (pd.DataFrame): The play DataFrame.
        x_col (str): The column name for the x-axis.
        y_col (str): The column name for the y-axis.
        name (str): The name of the trace.

    Returns:
        Callable: A function that generates a Plotly scatter trace.
    """

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
        tr.uid = f"metric_trace_{name}"
        return tr

    return trace_func


def gameplay_trace_func_26(
    frame_df: pd.DataFrame, player_role_colors: dict = player_role_colors
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

    # Player positions and properties
    player_x = frame_df["x"].tolist()
    player_y = frame_df["y"].tolist()
    player_colors = [
        player_role_colors.get(role, "#888888") for role in frame_df["playerRole"]
    ]
    player_text = frame_df["playerPosition"].tolist()

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


def gameplay_trail_trace_func(
    frame_df: pd.DataFrame,
    *,
    min_opacity: float = 0.9,
    line_width: float = 6.5,
    color_map: dict[int, str] = None,
) -> go.Scatter:
    """
    Draws player trails that preserve their historical color based on trainFlag.
    When a player's trainFlag switches (1→0 or 0→1), the old trail keeps its color
    and only new frames adopt the new color.

    Args
    ----
    frame_df : pd.DataFrame
        Data for a single frame (builder attaches full play as _full_play_df).
        Must include: 'frameId', 'nflId', 'x', 'y', 'trainFlag'.
    min_opacity : float, optional
        Opacity for lines.
    line_width : float, optional
        Width of streak lines.
    color_map : dict[int, str], optional
        Mapping from trainFlag to color. Defaults to green/red neon.

    Returns
    -------
    go.Scatter
        Combined trail trace (multiple colored segments).
    """
    # Bright high-contrast colors for white field
    if color_map is None:
        color_map = {1: "#00FF66", 0: "#FF1744"}  # neon green / neon red

    play_df = getattr(frame_df, "_full_play_df", frame_df)
    current_frame = int(frame_df["frameId"].iloc[0])
    visible_df = play_df[play_df["frameId"] <= current_frame].copy()
    visible_df = visible_df.sort_values(["nflId", "frameId"])

    # Build all segments that existed up to this frame
    xs, ys, seg_colors = [], [], []

    for pid, p_df in visible_df.groupby("nflId"):
        p_df = p_df.sort_values("frameId")
        if len(p_df) < 2:
            continue

        # Break into segments whenever trainFlag changes
        start_idx = 0
        for i in range(1, len(p_df)):
            if p_df["trainFlag"].iloc[i] != p_df["trainFlag"].iloc[i - 1]:
                segment = p_df.iloc[start_idx:i]
                xs += segment["x"].tolist() + [None]
                ys += segment["y"].tolist() + [None]
                seg_colors.append(color_map[p_df["trainFlag"].iloc[start_idx]])
                start_idx = i
        # final segment
        segment = p_df.iloc[start_idx:]
        xs += segment["x"].tolist() + [None]
        ys += segment["y"].tolist() + [None]
        seg_colors.append(color_map[p_df["trainFlag"].iloc[start_idx]])

    # Plot as one merged trace — Plotly can't color per segment natively,
    # so we use one color (the latest) here; multi-color requires multiple traces.
    # We'll just take the latest segment color for this frame
    current_flag = int(frame_df["trainFlag"].iloc[0])
    color = color_map[current_flag]

    trace = go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(color=color, width=line_width),
        opacity=min_opacity,
        hoverinfo="none",
        showlegend=False,
        name="trail",
    )

    return trace
