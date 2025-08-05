from collections import defaultdict
from functools import reduce

import polars as pl


def join_track_play_df(
    track_df: pl.DataFrame,
    play_df: pl.DataFrame,
) -> pl.DataFrame:
    return track_df.join(play_df, on=["gameId", "playId"], how="inner")


def merge_trace_dicts(*dicts: dict) -> dict:
    """
    Merges multiple trace_dicts of the form {frame_id: [(trace, row, col), ...]}.
    If the same frame_id exists in multiple dicts, their trace lists are concatenated.

    Parameters:
        *dicts: One or more trace_dicts to merge.

    Returns:
        A single merged dict with combined traces per frame_id.
    """

    def merge_two(d1, d2):
        merged = defaultdict(list, {k: v.copy() for k, v in d1.items()})
        for key, val in d2.items():
            merged[key].extend(val)
        return merged

    return dict(reduce(merge_two, dicts))
