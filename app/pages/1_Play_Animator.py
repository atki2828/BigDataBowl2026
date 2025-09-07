"""Just Testing Animation Look"""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from utility.animations import Field, PlayAnimator, build_trace_configs
from utility.dbx import DatabricksSQLClient
from utility.tracebuilders import (
    ball_carrier_circle_trace_func,
    build_metric_trace_func,
    gameplay_trace_func,
)
from utility.transformations import join_track_play_df

databricks_client = DatabricksSQLClient()

animation_config = {
    "duration": 30,
    "redraw": False,
    "slider_prefix": "Frame: ",
    "play_label": "▶",
    "pause_label": "⏸",
}

PLAY_ID = 2238
GAME_ID = 2022091811

play_query = f"""SELECT *
    FROM workspace.bigdatabowl2024.plays    
    WHERE playId = {PLAY_ID} AND gameId = {GAME_ID}
"""
track_query = f"""SELECT *
    FROM workspace.bigdatabowl2024.tracking
    WHERE playId = {PLAY_ID} AND gameId = {GAME_ID}
    ORDER BY frameId
"""

play_df = databricks_client.query_to_pl(play_query).with_columns(
    pl.col("ballCarrierId").cast(pl.Utf8).alias("ballCarrierId")
)
track_df = databricks_client.query_to_pl(track_query)


def create_animation_df(
    track_df: pl.DataFrame, play_df: pl.DataFrame, game_id: int, play_id: int
) -> pl.DataFrame:
    """
    Joins track and play DataFrames to create a DataFrame suitable for animation.

    Parameters:
        track_df (pl.DataFrame): DataFrame containing tracking data.
        play_df (pl.DataFrame): DataFrame containing play data.

    Returns:
        pl.DataFrame: Combined DataFrame ready for animation.
    """

    animate_play_df = join_track_play_df(track_df, play_df)
    print(animate_play_df.schema)
    ball_carrier_df = animate_play_df.filter(
        pl.col("nflId") == pl.col("ballCarrierId")
    ).select(["gameId", "playId", "frameId", pl.col("s").alias("bcs")])
    animate_play_df = animate_play_df.join(
        ball_carrier_df, on=["gameId", "playId", "frameId"]
    )
    return animate_play_df


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
        trace_func=gameplay_trace_func,  # returns go.Scatter of positions
        row=1,
        col=1,
    )

    ball_carrier_circle_trace_configs = build_trace_configs(
        play_df=animate_play_df,
        trace_func=ball_carrier_circle_trace_func,  # returns go.Scatter (marker circle)
        row=1,
        col=1,
    )

    # 3) Concatenate all trace configs
    trace_configs = gameplay_trace_configs + ball_carrier_circle_trace_configs

    # 4) Animate
    play_fig = PlayAnimator(
        field=field,
        animation_config=animation_config,
        trace_configs=trace_configs,
    ).create_animation()

    return play_fig


def main():
    animate_play_df = create_animation_df(
        track_df, play_df, GAME_ID, PLAY_ID
    ).to_pandas()
    fig = create_play_fig(animate_play_df, animation_config)
    st.title("Play Animation Demo")
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
