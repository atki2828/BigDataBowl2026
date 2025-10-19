"""Module for buiding plotly traced for Play Animations"""

from typing import Callable

import numpy as np
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
    trail_mark_opacity: float = 0.5,
    trail_mark_size: float = 9.5,
    color_map: dict[int, str] = None,
) -> go.Scatter:
    """
    Draws player trails that preserve their historical color based on trainFlag.
    Old segments remain their original color even after flag changes.
    Compatible with PlayAnimator (returns ONE go.Scatter).
    """
    if color_map is None:
        color_map = {1: "#00FF66", 0: "#FF1744"}  # neon green / neon red

    play_df = getattr(frame_df, "_full_play_df", frame_df)
    current_frame = int(frame_df["frameId"].iloc[0])
    visible_df = play_df[play_df["frameId"] <= current_frame - 3].copy()
    visible_df = visible_df.sort_values(["nflId", "frameId"])

    xs_all, ys_all, color_all = [], [], []

    for pid, p_df in visible_df.groupby("nflId"):
        p_df = p_df.sort_values("frameId")
        if len(p_df) < 2:
            continue

        start_idx = 0
        for i in range(1, len(p_df)):
            if p_df["trainFlag"].iloc[i] != p_df["trainFlag"].iloc[i - 1]:
                segment = p_df.iloc[start_idx:i]
                flag = int(p_df["trainFlag"].iloc[start_idx])
                xs_all += segment["x"].tolist() + [None]
                ys_all += segment["y"].tolist() + [None]
                color_all += [color_map[flag]] * (len(segment) + 1)
                start_idx = i

        # Final segment
        segment = p_df.iloc[start_idx:]
        flag = int(p_df["trainFlag"].iloc[start_idx])
        xs_all += segment["x"].tolist() + [None]
        ys_all += segment["y"].tolist() + [None]
        color_all += [color_map[flag]] * (len(segment) + 1)

    # Plotly trick: use markers to carry per-point color but hide them (size=0)
    trace = go.Scatter(
        x=xs_all,
        y=ys_all,
        mode="markers",
        # line=dict(width=line_width),
        marker=dict(
            color=color_all,
            size=trail_mark_size,  # invisible, just for coloring the line
            symbol=0,
        ),
        opacity=trail_mark_opacity,
        hoverinfo="none",
        showlegend=False,
        name="trail",
    )

    return trace


def player_to_predict_trace_func_with_sa(frame_df: pd.DataFrame) -> go.Scatter:
    """Draws black circle outlines around players to predict, with speed and acceleration text.

    Args:
        frame_df (pd.DataFrame): Single-frame DataFrame.
            Must include columns: 'x', 'y', 'playerToPredict', 's', 'a'.

    Returns:
        go.Scatter: Scatter trace with black outlines and (s,a) text labels.
    """
    if not {"playerToPredict", "x", "y", "s", "a"}.issubset(frame_df.columns):
        raise ValueError(
            "frame_df must contain 'x', 'y', 'playerToPredict', 's', and 'a' columns"
        )

    predict_df = frame_df[frame_df["playerToPredict"] == True]
    if predict_df.empty:
        return go.Scatter(
            x=[], y=[], mode="markers", showlegend=False, name="player_to_predict_trace"
        )

    train_mask = predict_df["trainFlag"] == 1
    x_positions = predict_df.loc[train_mask, "x"].tolist()
    y_positions = predict_df.loc[train_mask, "y"].tolist()
    text_labels = [
        f"s:{s:.1f}<br>a:{a:.1f}"
        for s, a in zip(
            predict_df.loc[train_mask, "s"], predict_df.loc[train_mask, "a"]
        )
    ]

    trace = go.Scatter(
        x=x_positions,
        y=y_positions,
        mode="markers+text",
        marker=dict(
            symbol="circle",
            size=34,  # outline slightly larger than normal markers
            color="rgba(0,0,0,0)",  # transparent fill
            line=dict(width=3, color="black"),
        ),
        text=text_labels,
        textposition="top center",
        textfont=dict(size=8, color="black"),
        hoverinfo="skip",
        showlegend=False,
        name="player_to_predict_trace",
    )
    return trace


def arrow_trace_func(
    frame_df: pd.DataFrame,
    angle_col: str,
    color: str = "blue",
    arrow_length: float = 3.0,
) -> go.Scatter:
    """
    Builds a Plotly line trace showing short arrow-like segments for players to predict.

    Args:
        frame_df (pd.DataFrame): DataFrame for a single frame.
            Must include 'x', 'y', 'playerToPredict', and `angle_col`.
        angle_col (str): Column containing angles in degrees (e.g., 'dir' or 'o').
        color (str): Line color for the arrows.
        arrow_length (float): Length of each arrow segment in yards.

    Returns:
        go.Scatter: A Plotly line trace representing arrows for players to predict.
    """
    required_cols = {"x", "y", "playerToPredict", angle_col}
    if not required_cols.issubset(frame_df.columns):
        raise ValueError(f"frame_df must contain {required_cols}")

    play_to_predict_mask = frame_df["playerToPredict"] == True
    train_mask = frame_df["trainFlag"] == 1
    total_mask = play_to_predict_mask & train_mask
    predict_df = frame_df[total_mask]
    if predict_df.empty:
        return go.Scatter(
            x=[], y=[], mode="lines", showlegend=False, name=f"{angle_col}_arrows"
        )

    # Compute arrow segments
    x_all, y_all = [], []
    for _, row in predict_df.iterrows():
        x, y = row["x"], row["y"]
        rad = np.deg2rad(row[angle_col])
        x_end = x + arrow_length * np.sin(rad)
        y_end = y + arrow_length * np.cos(rad)
        x_all += [x, x_end, None]
        y_all += [y, y_end, None]

    trace = go.Scatter(
        x=x_all,
        y=y_all,
        mode="lines",
        line=dict(width=2, color=color),
        hoverinfo="skip",
        showlegend=False,
        name=f"{angle_col}_arrows",
    )
    return trace


def player_to_predict_trace_func(frame_df: pd.DataFrame) -> go.Scatter:
    """Generates a Plotly trace that outlines players to predict with a black circle.

    Args:
        frame_df (pd.DataFrame): DataFrame for a single frame.
            Must include columns 'x', 'y', and boolean 'playerToPredict'.

    Returns:
        go.Scatter: A Plotly scatter trace showing black circle outlines for players to predict.
    """
    if "playerToPredict" not in frame_df.columns:
        raise ValueError("frame_df must contain a 'playerToPredict' column")

    # Filter only players to predict
    predict_df = frame_df[frame_df["playerToPredict"] == True]
    if predict_df.empty:
        # Return an invisible trace so layout doesn’t shift
        return go.Scatter(
            x=[], y=[], mode="markers", showlegend=False, name="player_to_predict_trace"
        )

    x_positions = predict_df["x"].tolist()
    y_positions = predict_df["y"].tolist()

    trace = go.Scatter(
        x=x_positions,
        y=y_positions,
        mode="markers",
        marker=dict(
            symbol="circle",
            size=34,  # slightly larger than normal player markers (24)
            color="rgba(0,0,0,0)",  # transparent fill
            line=dict(width=3, color="black"),  # black circular outline
        ),
        hoverinfo="skip",
        showlegend=False,
        name="player_to_predict_trace",
    )
    return trace


def ball_landing_trace_func(frame_df: pd.DataFrame) -> go.Scatter:
    """Generates a Plotly trace marking the ball landing location as a large red X.

    Args:
        frame_df (pd.DataFrame): DataFrame containing at least 'ballLandX' and 'ballLandY' columns.

    Returns:
        go.Scatter: A single Plotly scatter trace marking the ball landing point.
    """
    if "ballLandX" not in frame_df.columns or "ballLandY" not in frame_df.columns:
        raise ValueError("frame_df must contain 'ballLandX' and 'ballLandY' columns")

    # Extract single point coordinates
    x = frame_df["ballLandX"].clip(0, 120).iloc[0]
    y = frame_df["ballLandY"].clip(0, 53.3).iloc[0]

    trace = go.Scatter(
        x=[x],
        y=[y],
        mode="markers",
        marker=dict(
            symbol="x",
            color="red",
            size=20,  # large X
            line=dict(width=3, color="darkred"),  # bold stroke
        ),
        hoverinfo="skip",
        showlegend=False,
        name="ball_landing_trace",
    )
    return trace
