import os
from functools import partial

import plotly.graph_objects as go
import polars as pl

from utility.animations import (
    add_initial_traces,
    add_los_and_first_down,
    animate_play,
    ball_carrier_circle_trace_func,
    ball_carrier_speed_trace_func,
    create_bdb_field_figure,
    create_frames,
    create_trace_hash,
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


def create_field_animation(df: pl.DataFrame, animation_config: dict) -> go.Figure:
    """
    Creates an animated field figure from the provided DataFrame.

    Parameters:
        df (pl.DataFrame): DataFrame containing tracking data.

    Returns:
        go.Figure: Animated field figure.
    """
    fig = create_bdb_field_figure()
    fig = add_los_and_first_down(fig, df.to_pandas())

    play_trace_hash = create_trace_hash(
        df.to_pandas(), gameplay_trace_func, metric=False
    )
    ball_carrier_circle_hash = create_trace_hash(
        df.to_pandas(), ball_carrier_circle_trace_func, metric=False
    )

    final_play_trace_hash = merge_trace_dicts(play_trace_hash, ball_carrier_circle_hash)
    frames = create_frames(final_play_trace_hash)

    return animate_play(fig, frames, config=animation_config)


def play_metric_animation(df: pl.DataFrame, animation_config: dict) -> go.Figure:
    """Creates a play metric animation figure.

    Args:
        df (pl.DataFrame): DataFrame containing tracking data.
        animation_config (dict): Configuration dictionary for animation.

    Returns:
        go.Figure: Animated field figure.
    """
    max_frame_id = df.select(pl.col("frameId").max()).item()
    speed_trace_func = partial(ball_carrier_speed_trace_func, row_idx=1, col_idx=2)
    play_trace_hash = create_trace_hash(df.to_pandas(), gameplay_trace_func)
    ball_carrier_circle_hash = create_trace_hash(
        df.to_pandas(), ball_carrier_circle_trace_func
    )
    ball_carrier_speed_hash = create_trace_hash(
        df.to_pandas(), speed_trace_func, metric=True
    )
    # Add static field with subplot support
    fig = create_bdb_field_figure(
        use_subplots=True, rows=1, cols=2, subplot_row=1, subplot_col=1
    )
    ball_carrier_circle_hash = create_trace_hash(
        df.to_pandas(), ball_carrier_circle_trace_func
    )
    ball_carrier_speed_hash = create_trace_hash(
        df.to_pandas(), speed_trace_func, metric=True
    )
    # Add static field with subplot support
    fig = create_bdb_field_figure(
        use_subplots=True, rows=1, cols=2, subplot_row=1, subplot_col=1
    )
    fig = add_los_and_first_down(fig, df.to_pandas(), row=1, col=1)

    play_trace_hash = create_trace_hash(df.to_pandas(), gameplay_trace_func)
    ball_carrier_circle_hash = create_trace_hash(
        df.to_pandas(), ball_carrier_circle_trace_func
    )
    ball_carrier_speed_hash = create_trace_hash(
        df.to_pandas(), speed_trace_func, metric=True
    )

    final_trace_hash = merge_trace_dicts(
        play_trace_hash, ball_carrier_circle_hash, ball_carrier_speed_hash
    )
    add_initial_traces(fig, final_trace_hash)
    # TODO: These are hardcoded values, consider making them dynamic
    frames = create_frames(final_trace_hash)
    fig = animate_play(fig, frames, config=animation_config)
    fig.update_layout(
        xaxis2=dict(
            range=[0, max_frame_id], fixedrange=True
        ),  # lock x-axis for subplot
        yaxis2=dict(range=[0, 15], fixedrange=True),  # lock y-axis for subplot
    )
    return fig


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


def main():
    # Read In Data
    track_df = pl.read_csv(tracking_file_path, null_values="NA")
    plays_df = pl.read_csv(play_df_file_path, null_values="NA")

    # Transform Data to Create animate_play_df
    animate_play_df = create_animation_df(track_df, plays_df, 2022091109, 767)

    # Create Animations
    play_fig = create_field_animation(animate_play_df, animation_config)
    play_metric_fig = play_metric_animation(animate_play_df, animation_config)

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
