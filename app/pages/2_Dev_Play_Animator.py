import polars as pl
import streamlit as st

from utility.dbx import DatabricksSQLClient

databricks_client = DatabricksSQLClient()

st.sidebar.title("Big Data Bowl Explorer")


import polars as pl
import streamlit as st


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
        ORDER BY frame_id, nfl_id
    """


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
        # TODO: Add your animation/visualization logic here

        animation_query = build_animation_query(game_id, play_id)
        animation_df = databricks_client.query_to_pl(animation_query)

        # Example: Display the animation data as a table
        st.dataframe(animation_df.sample(10))


if __name__ == "__main__":
    # Pass in your Databricks client instance here
    # Example: from my_package import databricks_client
    main(databricks_client)
