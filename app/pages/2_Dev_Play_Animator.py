from typing import Optional

import plotly.graph_objects as go
import polars as pl
import pandas as pd
import streamlit as st

from utility.animations import Field, PlayAnimator, build_trace_configs
from utility.dbx import DatabricksSQLClient
from utility.tracebuilders import (

    gameplay_trace_func
)

databricks_client = DatabricksSQLClient()

st.sidebar.title("Big Data Bowl Explorer")




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

    # 3) Concatenate all trace configs
    trace_configs = gameplay_trace_configs + 

    # 4) Animate
    play_fig = PlayAnimator(
        field=field,
        animation_config=animation_config,
        trace_configs=trace_configs,
    ).create_animation()

    return play_fig


# --- Query Helpers ---
def get_game_ids(databricks_client):
    """Fetch distinct game IDs and allow user to select one."""
    game_query = """
        SELECT DISTINCT game_id
        FROM workspace.bigdatabowl2026.input_data
        ORDER BY game_id
    """
    games_ids = (
        databricks_client.query_to_pl(game_query)
        .select(pl.col("game_id").cast(pl.Int64))
        .to_series()
        .to_list()
    )

    game_id = st.sidebar.selectbox("Select Game ID", games_ids, index=None)
    return game_id


def get_play_id(databricks_client, game_id: int | None):
    """Fetch distinct play IDs for a given game and allow user to select one."""
    if not game_id:
        return None

    play_query = f"""
        SELECT DISTINCT play_id
        FROM workspace.bigdatabowl2026.input_data
        WHERE game_id = {game_id}
        ORDER BY play_id
    """
    plays_df = databricks_client.query_to_pl(play_query)
    play_id = st.sidebar.selectbox(
        "Select Play ID", plays_df["play_id"].to_list(), index=None
    )
    return play_id


def build_animation_query(game_id: int, play_id: int) -> str:
    """Construct SQL query to fetch animation data for a specific game and play."""
    return f"""
        SELECT *
        FROM workspace.bigdatabowl2026.input_data
        WHERE game_id = {game_id} AND play_id = {play_id}
    """


def animate_play(play_df: pl.DataFrame):
    pass


# --- Main App ---
def main(databricks_client):
    st.title("Big Data Bowl Play Explorer")

    game_id = get_game_ids(databricks_client)
    play_id = get_play_id(databricks_client, game_id)

    if not game_id or not play_id:
        st.info("Select a Game ID and Play ID to view the animation.")
    else:
        st.success(f"Selected Game ID: {game_id}")
        st.success(f"Selected Play ID: {play_id}")
        animation_query = build_animation_query(game_id, play_id)
        animation_df = databricks_client.query_to_pl(animation_query)

        # TODO: Add your animation/visualization logic here

        # Example: Display the animation data as a table
        st.dataframe(animation_df.sample(10))


if __name__ == "__main__":
    # Pass in your Databricks client instance here
    # Example: from my_package import databricks_client
    main(databricks_client)
