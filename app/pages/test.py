import os
from typing import Callable, List, Optional, Union

import pandas as pd
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from utility.animations import Field, PlayAnimator, TraceConfig, build_trace_configs
from utility.colors import player_role_colors
from utility.dbx import DatabricksSQLClient
from utility.tracebuilders import gameplay_trace_func_26

databricks_client = DatabricksSQLClient()

st.sidebar.title("Big Data Bowl Explorer")

animation_config = {
    "duration": 300,
    "redraw": False,
    "slider_prefix": "Frame: ",
    "play_label": "▶",
    "pause_label": "⏸",
}


def build_trace_configs(
    play_df: pd.DataFrame,
    trace_func: Callable,
    *,
    row: int = 1,
    col: int = 1,
    trailing: bool = False,
) -> List["TraceConfig"]:
    """
    Build TraceConfig objects for each frame.

    Parameters
    ----------
    play_df : pd.DataFrame
        Full play data containing frameId.
    trace_func : callable
        Function that takes a frame_df and returns a Plotly trace.
    row, col : int
        Subplot coordinates.
    trailing : bool, optional
        If False (default): pass only that frame's rows to trace_func.
        If True: pass all rows up to the current frame_id (for trails).

    Returns
    -------
    list[TraceConfig]
    """
    configs: List["TraceConfig"] = []

    frame_ids = sorted(play_df["frameId"].unique())

    if not trailing:
        # Normal frame-by-frame traces
        for _, df in play_df.groupby("frameId", sort=True):
            configs.append(
                TraceConfig(frame_df=df, trace_func=trace_func, row=row, col=col)
            )
    else:
        # Cumulative trails — build progressively longer DataFrames
        for fid in frame_ids:
            df = play_df[play_df["frameId"] <= fid]
            configs.append(
                TraceConfig(frame_df=df, trace_func=trace_func, row=row, col=col)
            )

    return configs


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


def create_play_fig(
    animate_play_df: pd.DataFrame, animation_config: Optional[dict] = None
) -> go.Figure:
    """
    Creates a 1x2 subplot figure: field on (1,1) and a metric plot on (1,2),
    animated over frameId using the PlayAnimator/TraceConfig pattern.
    """
    # 1) Field on left subplot (1,1); grid is 1 row x 2 columns
    field = Field(
        play_df=animate_play_df,
        row=1,
        col=1,
        subplot_rows=1,
        subplot_cols=1,
    )

    # 2) Build traces for each frame + target subplot cell
    gameplay_trace_configs = build_trace_configs(
        play_df=animate_play_df,
        trace_func=gameplay_trace_func_26,  # returns go.Scatter of positions
        row=1,
        col=1,
    )

    trail_trace_configs = build_trace_configs(
        play_df=animate_play_df,
        trace_func=gameplay_trail_trace_func,  # returns go.Scatter of trails
        row=1,
        col=1,
        trailing=True,
    )

    # 3) Concatenate all trace configs
    trace_configs = gameplay_trace_configs + trail_trace_configs
    # 4) Animate
    play_fig = PlayAnimator(
        field=field,
        animation_config=animation_config,
        trace_configs=trace_configs,
    ).create_animation()

    return play_fig


def build_animation_query(game_id: int, play_id: int) -> str:
    """Construct SQL query to fetch animation data for a specific game and play."""
    return f"""
  SELECT * 
        FROM workspace.bigdatabowl2026.play_animation_data
        WHERE gameId = {game_id}
        AND playId = {play_id}
        ORDER BY frameId DESC

    """


write_dir = "."


# --- Main App ---
def main(databricks_client):
    animation_query = build_animation_query(2023091001, 284)
    animation_df = databricks_client.query_to_pl(animation_query)
    fig = create_play_fig(animation_df.to_pandas(), animation_config)
    fig.write_html(
        os.path.join(write_dir, f"game_test_play_test_animation.html"),
        include_plotlyjs="cdn",
        auto_play=False,
    )


if __name__ == "__main__":
    # Pass in your Databricks client instance here
    main(databricks_client)
