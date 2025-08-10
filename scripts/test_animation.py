import os
from functools import partial
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
import polars as pl

from utility.animations import Field, PlayAnimator, TraceConfig, build_trace_configs
from utility.tracebuilders import (
    ball_carrier_circle_trace_func,
    build_metric_trace_func,
    gameplay_trace_func,
)
from utility.transformations import join_track_play_df

tracking_file_path = "./data/bigdatabowl2024/tracking_week_1.csv"
play_df_file_path = "./data/bigdatabowl2024/plays.csv"
write_dir = "./animations/"
animation_config = {
    "duration": 33,  # ~30 fps (super smooth)
    "redraw": False,  # reuse traces
    "slider_prefix": "Frame: ",
    "play_label": "▶",
    "pause_label": "⏸",
}

PLAY_ID = 2329
GAME_ID = 2022091106


def make_cumulative_metric_trace_func(all_df, x_col, y_col, name):
    all_df = all_df.sort_values(x_col)

    def trace_func(frame_df: pd.DataFrame) -> go.Scatter:
        fid = int(frame_df["frameId"].iloc[0])
        sub = all_df[all_df["frameId"] <= fid]
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
    all_plays_df = join_track_play_df(track_df, play_df)
    animate_play_df = all_plays_df.filter(
        (pl.col("gameId") == game_id) & (pl.col("playId") == play_id)
    )
    ball_carrier_df = animate_play_df.filter(
        pl.col("nflId") == pl.col("ballCarrierId")
    ).select(["gameId", "playId", "frameId", pl.col("s").alias("bcs")])
    animate_play_df = animate_play_df.join(
        ball_carrier_df, on=["gameId", "playId", "frameId"]
    )
    return animate_play_df


def create_play_fig(
    animate_play_df: pd.DataFrame, animation_config=animation_config
) -> go.Figure:
    """Creates a figure for the animated play using the provided DataFrame.

    Args:
        animate_play_df (pl.DataFrame): The DataFrame containing the animated play data.

    Returns:
        go.Figure: The created play figure.
    """
    field = Field(play_df=animate_play_df)
    gameplay_trace_configs = (
        animate_play_df.groupby("frameId")
        .apply(
            lambda df: TraceConfig(
                frame_df=df,
                trace_func=gameplay_trace_func,
            )
        )
        .to_list()
    )

    ball_carrier_highlight_trace_configs = (
        animate_play_df.groupby("frameId")
        .apply(
            lambda df: TraceConfig(
                frame_df=df,
                trace_func=ball_carrier_circle_trace_func,
            )
        )
        .to_list()
    )

    trace_configs = gameplay_trace_configs + ball_carrier_highlight_trace_configs
    play_fig = PlayAnimator(
        field=field,
        animation_config=animation_config,
        trace_configs=trace_configs,
    ).create_animation()
    return play_fig


def create_play_metric_fig(
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
        subplot_cols=2,
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

    speed_metric_trace_func = make_cumulative_metric_trace_func(
        all_df=animate_play_df, x_col="frameId", y_col="bcs", name="Ball Carrier Speed"
    )

    ball_carrier_circle_trace_configs = build_trace_configs(
        play_df=animate_play_df,
        trace_func=speed_metric_trace_func,
        row=1,
        col=2,
    )
    # 3) Concatenate all trace configs
    trace_configs = (
        gameplay_trace_configs
        + ball_carrier_circle_trace_configs
        + ball_carrier_circle_trace_configs
    )

    # 4) Animate
    play_metric_fig = PlayAnimator(
        field=field,
        animation_config=animation_config,
        trace_configs=trace_configs,
    ).create_animation()

    min_f, max_f = int(animate_play_df["frameId"].min()), int(
        animate_play_df["frameId"].max()
    )
    ymin, ymax = animate_play_df["bcs"].min(), animate_play_df["bcs"].max()
    pad = 0.05 * max(1e-9, ymax - ymin)  # avoid zero-width

    play_metric_fig.update_xaxes(range=[min_f, max_f], row=1, col=2)
    play_metric_fig.update_yaxes(range=[ymin - pad, ymax + pad], row=1, col=2)

    return play_metric_fig


def main():
    # Read In Data
    track_df = pl.read_csv(tracking_file_path, null_values="NA")
    plays_df = pl.read_csv(play_df_file_path, null_values="NA")

    # Transform Data to Create animate_play_df
    animate_play_df = create_animation_df(track_df, plays_df, GAME_ID, PLAY_ID)

    play_metric_fig = create_play_metric_fig(
        animate_play_df.to_pandas(), animation_config=animation_config
    )

    play_metric_fig.write_html(
        os.path.join(write_dir, f"game_{GAME_ID}_play_{PLAY_ID}_metric.html"),
        include_plotlyjs="cdn",
        auto_play=False,
    )


if __name__ == "__main__":
    main()
