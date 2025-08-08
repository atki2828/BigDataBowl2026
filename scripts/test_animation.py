import os
from functools import partial

import plotly.graph_objects as go
import polars as pl

from utility.animations import Field, PlayAnimator, TraceConfig
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


def create_play_fig(animate_play_df: pl.DataFrame) -> go.Figure:
    field = Field(play_df=animate_play_df.to_pandas())
    trace_configs = []


def main():
    # Read In Data
    track_df = pl.read_csv(tracking_file_path, null_values="NA")
    plays_df = pl.read_csv(play_df_file_path, null_values="NA")

    # Transform Data to Create animate_play_df
    animate_play_df = create_animation_df(track_df, plays_df, 2022091109, 767)

    # Write out the figures to HTML files
    play_fig.write_html(
        os.path.join(write_dir, "game_2022091109_play_767.html"),
        include_plotlyjs="cdn",
    )
    play_metric_fig.write_html(
        os.path.join(write_dir, "game_2022091109_play_767_metric.html"),
        include_plotlyjs="cdn",
    )


if __name__ == "__main__":
    main()
