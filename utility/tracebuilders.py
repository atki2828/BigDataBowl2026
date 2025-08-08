from typing import Dict, List, Tuple

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


def gameplay_trace_func(row: pd.Series, nfl_colors: dict = nfl_colors) -> go.Scatter:
    """Generates a Plotly trace for visualizing a player or football position on the field.
    This function creates a scatter plot trace with specific styling based on whether
    the input represents a football or a player. For players, it shows their jersey
    numbers and uses team-specific colors.
        row (pd.Series): A pandas Series containing position and player data.
            Must include 'x', 'y', 'displayName' columns, and for players:
            'club', 'jerseyNumber'.
        nfl_colors (dict, optional): Dictionary mapping NFL team codes to their
            color hex codes. Defaults to predefined nfl_colors.
        row_idx (int, optional): Row index for subplot positioning. Defaults to 1.
        col_idx (int, optional): Column index for subplot positioning. Defaults to 1.
        dict: A dictionary containing:
            - trace (go.Scatter): The generated Plotly scatter trace
            - row (int): Row index for subplot positioning
            - col (int): Column index for subplot positioning
    Example:
        >>> player_data = pd.Series({
        ...     'x': 25.5,
        ...     'y': 53.3,
        ...     'displayName': 'John Doe',
        ...     'club': 'SEA',
        ...     'jerseyNumber': 12
        ... })
        >>> trace_dict = gameplay_trace_func(player_data)
    """
    if row["displayName"].lower() == "football":
        trace = go.Scatter(
            x=[row["x"]],
            y=[row["y"]],
            mode="markers",
            marker=dict(size=10, color="saddlebrown", symbol="circle"),
            name="Football",
            hoverinfo="skip",
            showlegend=False,
        )
    else:
        club = row.get("club", "")
        color = nfl_colors.get(club, "#888888")
        trace = go.Scatter(
            x=[row["x"]],
            y=[row["y"]],
            mode="markers+text",
            marker=dict(size=24, color=color, line=dict(width=1, color="white")),
            text=[str(int(row["jerseyNumber"]))],
            textposition="middle center",
            textfont=dict(color="white", size=10),
            name=club,
            hoverinfo="text",
            showlegend=False,
        )

    return trace


def ball_carrier_circle_trace_func(row) -> go.Scatter:
    """
    Adds a translucent circle around the ball carrier for a given frame row.
    Returns a subplot-aware trace dict only if the row is the ball carrier.
    """
    if row["nflId"] != row["ballCarrierId"]:
        return None  # Skip non-carriers

    trace = go.Scatter(
        x=[row["x"]],
        y=[row["y"]],
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

    return trace
