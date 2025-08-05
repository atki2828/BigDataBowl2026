import os

import pandas as pd
import polars as pl

from .animations import (
    add_los_and_first_down,
    animate_play,
    ball_carrier_circle_trace_func,
    create_bdb_field_figure,
    create_frames,
    create_trace_hash,
    gameplay_trace_func,
)
from .transformations import join_track_play_df, merge_trace_dicts

tracking_file_path = "../data/bigdatabowl2024/tracking_week_1.csv"
play_df_file_path = "../data/bigdatabowl2024/plays.csv"
write_dir = "../animations/"


def main():
    # Read In Data
    track_df = pl.read_csv(tracking_file_path, null_values="NA")
    plays_df = pl.read_csv(play_df_file_path, null_values="NA")

    # Transform Data to Create animate_play_df
    all_plays_df = join_track_play_df(track_df, plays_df)
    animate_play_df = all_plays_df.filter(
        (pl.col("gameId") == 2022091103) & (pl.col("playId") == 3126)
    )
    ball_carrier_df = animate_play_df.filter(
        pl.col("nflId") == pl.col("ballCarrierId")
    ).select(["gameId", "playId", "frameId", pl.col("s").alias("bcs")])
    animate_play_df = animate_play_df.join(
        ball_carrier_df, on=["gameId", "playId", "frameId"]
    )

    fig = create_bdb_field_figure()
    fig = add_los_and_first_down(fig, animate_play_df.to_pandas())

    play_trace_hash = create_trace_hash(
        animate_play_df.to_pandas(), gameplay_trace_func, metric=False
    )
    ball_carrier_circle_hash = create_trace_hash(
        animate_play_df.to_pandas(), ball_carrier_circle_trace_func, metric=False
    )
    final_play_trace_hash = merge_trace_dicts(play_trace_hash, ball_carrier_circle_hash)
    frames = create_frames(final_play_trace_hash)
    play_fig = animate_play(fig, frames)
    play_fig.write_html(
        os.path.join(write_dir, "game_2022091103_play_3126.html"),
        include_plotlyjs="cdn",
    )


if __name__ == "__main__":
    main()
