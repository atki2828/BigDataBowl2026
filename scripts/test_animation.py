import os
from functools import partial

import pandas as pd
import plotly.graph_objects as go
import polars as pl

from utility.animations import (
    Field,
    PlayAnimator,
    TraceConfig,
    build_trace_configs,
    create_bdb_field_figure,
)
from utility.tracebuilders import (
    ball_carrier_circle_trace_func,
    ball_carrier_speed_trace_func,
    gameplay_trace_func,
)
from utility.transformations import join_track_play_df, merge_trace_dicts

tracking_file_path = "./data/bigdatabowl2024/tracking_week_1.csv"
play_df_file_path = "./data/bigdatabowl2024/plays.csv"
write_dir = "./animations/"
animation_config = {
    "duration": 300,
    "slider_prefix": "Frame: ",
    "play_label": "▶",
    "pause_label": "⏸",
}

PLAY_ID = 767
GAME_ID = 2022091109


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
    animate_play_df: pd.DataFrame, animation_config=animation_config
) -> go.Figure:
    """Creates a figure for the play metrics using the provided DataFrame.

    Args:
        animate_play_df (pl.DataFrame): The DataFrame containing the animated play data.

    Returns:
        go.Figure: The created play metrics figure.
    """
    field = Field(play_df=animate_play_df, use_subplots=True, row=1, col=1)
    gameplay_trace_configs = build_trace_configs(
        play_df=animate_play_df, trace_func=gameplay_trace_func, row=1, col=1
    )
    ball_carrier_circle_trace_configs = build_trace_configs(
        play_df=animate_play_df, trace_func=ball_carrier_circle_trace_func, row=1, col=1
    )

    ball_carrier_speed_trace_configs = build_trace_configs(
        play_df=animate_play_df, trace_func=ball_carrier_speed_trace_func, row=1, col=2
    )

    # This will concatenate a list of lists
    trace_configs = sum(
        [
            gameplay_trace_configs,
            ball_carrier_speed_trace_configs,
            ball_carrier_circle_trace_configs,
        ],
        [],
    )
    play_metric_fig = PlayAnimator(
        field=field,
        animation_config=animation_config,
        trace_configs=trace_configs,
    ).create_animation()
    return play_metric_fig


def main():
    # Read In Data
    track_df = pl.read_csv(tracking_file_path, null_values="NA")
    plays_df = pl.read_csv(play_df_file_path, null_values="NA")

    # Transform Data to Create animate_play_df
    animate_play_df = create_animation_df(track_df, plays_df, GAME_ID, PLAY_ID)
    play_fig = create_play_fig(animate_play_df.to_pandas())
    play_metric_fig = create_play_metric_fig(animate_play_df.to_pandas())

    # Write out the figures to HTML files
    play_fig.write_html(
        os.path.join(write_dir, f"game_{GAME_ID}_play_{PLAY_ID}.html"),
        include_plotlyjs="cdn",
    )
    play_metric_fig.write_html(
        os.path.join(write_dir, f"game_{GAME_ID}_play_{PLAY_ID}_metric.html"),
        include_plotlyjs="cdn",
    )


if __name__ == "__main__":
    main()
