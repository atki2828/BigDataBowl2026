from functools import partial
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from utility.animations import Field, PlayAnimator, build_trace_configs
from utility.colors import player_role_colors
from utility.dbx import DatabricksSQLClient
from utility.tracebuilders import (
    arrow_trace_func,
    ball_landing_trace_func,
    gameplay_trace_func_26,
    gameplay_trail_trace_func,
    player_to_predict_trace_func_with_sa,
)

databricks_client = DatabricksSQLClient()

st.sidebar.title("Big Data Bowl Explorer")

animation_config = {
    "duration": 125,
    "redraw": False,
    "slider_prefix": "Frame: ",
    "play_label": "▶",
    "pause_label": "⏸",
}

import streamlit as st
import streamlit.components.v1 as components


def show_play_legend(player_role_colors: dict):
    """Render a custom HTML legend for player roles and arrows."""
    legend_html = """
    <style>
        .legend-container {
            display: flex;
            flex-direction: column;
            gap: 6px;
            padding: 10px;
            border-radius: 10px;
            background-color: #f8f9fa;
            font-family: 'Helvetica', sans-serif;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9rem;
        }
        .legend-color {
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }
        .legend-header {
            font-weight: 600;
            font-size: 1rem;
            margin-bottom: 5px;
            color: #333;
        }
    </style>
    <div class="legend-container">
        <div class="legend-header">🎨 Player Role Colors</div>
    """

    # Player roles
    for role, color in player_role_colors.items():
        legend_html += f"""
        <div class="legend-item">
            <div class="legend-color" style="background-color:{color};"></div>
            <span>{role}</span>
        </div>
        """

    # Add directional arrow colors
    legend_html += """
        <div class="legend-header" style="margin-top:8px;">🧭 Arrows</div>
        <div class="legend-item">
            <div class="legend-color" style="background-color:blue;"></div>
            <span>Direction (dir)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color:green;"></div>
            <span>Orientation (o)</span>
        </div>
    </div>
    """

    components.html(legend_html, height=260)


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
    # Trail trace config
    trail_trace_configs = build_trace_configs(
        play_df=animate_play_df,
        trace_func=gameplay_trail_trace_func,  # returns go.Scatter of trails
        row=1,
        col=1,
        trailing=True,
    )
    # Ball landing trace config
    ball_landing_trace_configs = build_trace_configs(
        play_df=animate_play_df,
        trace_func=ball_landing_trace_func,  # returns go.Scatter of ball landing point
        row=1,
        col=1,
    )
    # Player to predict trace config
    player_to_predict_trace_configs = build_trace_configs(
        play_df=animate_play_df,
        trace_func=player_to_predict_trace_func_with_sa,  # returns go.Scatter of players to predict
        row=1,
        col=1,
    )
    # Direction arrow trace func
    direction_arrow_trace_func = partial(
        arrow_trace_func, angle_col="dir", color="blue", arrow_length=3.0
    )

    # Direction arrow trace configs
    direction_arrow_trace_configs = build_trace_configs(
        play_df=animate_play_df,
        trace_func=direction_arrow_trace_func,  # returns go.Scatter of orientation and direction arrows
        row=1,
        col=1,
    )

    # Orientation arrow trace func
    orientation_arrow_trace_func = partial(
        arrow_trace_func, angle_col="o", color="green", arrow_length=3.0
    )
    # Orientation arrow trace configs
    orientation_arrow_trace_configs = build_trace_configs(
        play_df=animate_play_df,
        trace_func=orientation_arrow_trace_func,  # returns go.Scatter of orientation arrows
        row=1,
        col=1,
    )
    # Combine arrow trace configs
    arrow_trace_configs = (
        direction_arrow_trace_configs + orientation_arrow_trace_configs
    )

    # 3) Concatenate all trace configs
    trace_configs = (
        gameplay_trace_configs
        + trail_trace_configs
        + ball_landing_trace_configs
        + player_to_predict_trace_configs
        + arrow_trace_configs
    )
    # 4) Animate
    play_fig = PlayAnimator(
        field=field,
        animation_config=animation_config,
        trace_configs=trace_configs,
    ).create_animation()

    return play_fig


def get_game_ids(databricks_client):
    """Fetch distinct games from play_animation_data and let user select one game_id."""
    game_query = """
        SELECT DISTINCT 
            gameId AS game_id,
            homeTeamAbbr,
            visitorTeamAbbr,
            CAST(gameDate AS DATE) AS game_date
        FROM workspace.bigdatabowl2026.play_animation_data
        ORDER BY game_date DESC
    """

    games_df = databricks_client.query_to_pl(game_query)

    if games_df.is_empty():
        st.warning("No games found in play_animation_data.")
        return None

    # Build readable display labels
    games_df = games_df.with_columns(
        (
            pl.col("game_date").dt.strftime("%Y-%m-%d")
            + " | "
            + pl.col("visitorTeamAbbr")
            + " @ "
            + pl.col("homeTeamAbbr")
        ).alias("game_label")
    )

    # Convert to Python lists for Streamlit selectbox
    labels = games_df["game_label"].to_list()
    ids = games_df["game_id"].to_list()

    # Let user select one game
    selected_label = st.sidebar.selectbox("Select Game", options=labels, index=None)

    if selected_label is None:
        return None

    # Return only the game_id corresponding to the selected label
    return ids[labels.index(selected_label)]


def get_play_id(databricks_client, game_id: int | None):
    """Fetch distinct plays for a selected game from play_animation_data and return one play_id."""
    if not game_id:
        return None

    play_query = f"""
        SELECT DISTINCT 
            playId AS play_id,
            playDescription,
            down,
            yardsToGo,
            quarter
        FROM workspace.bigdatabowl2026.play_animation_data
        WHERE gameId = {game_id}
        ORDER BY play_id
    """

    plays_df = databricks_client.query_to_pl(play_query)

    if plays_df.is_empty():
        st.warning("No plays found for this game in play_animation_data.")
        return None

    # Build readable labels for dropdown
    plays_df = plays_df.with_columns(
        (
            "Q"
            + pl.col("quarter").cast(pl.Utf8)
            + " | "
            + pl.col("down").cast(pl.Utf8)
            + " & "
            + pl.col("yardsToGo").cast(pl.Utf8)
            + " | "
            + pl.col("playDescription")
        ).alias("play_label")
    )

    labels = plays_df["play_label"].to_list()
    ids = plays_df["play_id"].to_list()

    selected_label = st.sidebar.selectbox("Select Play", options=labels, index=None)

    if selected_label is None:
        return None

    return ids[labels.index(selected_label)]


def build_animation_query(game_id: int, play_id: int) -> str:
    """Construct SQL query to fetch animation data for a specific game and play."""
    return f"""
  SELECT * 
        FROM workspace.bigdatabowl2026.play_animation_data
        WHERE gameId = {game_id}
        AND playId = {play_id}
        ORDER BY frameId DESC

    """


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

        if animation_df.is_empty():
            st.warning("No data found for this play.")
            return

        # --- Extract one row of play-level metadata ---
        play_info_cols = [
            "playDescription",
            "quarter",
            "down",
            "yardsToGo",
            "yardlineNumber",
            "possessionTeam",
            "gameClock",
        ]
        play_info_cols = [c for c in play_info_cols if c in animation_df.columns]

        play_info = animation_df.select(play_info_cols).unique().to_pandas().iloc[0]

        # --- Display Play Summary Card ---
        with st.container():
            st.markdown("## 🏈 Play Summary")
            st.markdown(
                f"**{play_info.get('playDescription', 'No description available.')}**"
            )
            st.caption(
                f"{play_info.get('possessionTeam', 'N/A')} | "
                f"Q{play_info.get('quarter', 'N/A')} | "
                f"{play_info.get('down', 'N/A')} & {play_info.get('yardsToGo', 'N/A')} | "
                f"Yardline {play_info.get('yardlineNumber', 'N/A')} | "
                f"Clock {play_info.get('gameClock', 'N/A')}"
            )

        # --- Create and display animation ---
        fig = create_play_fig(animation_df.to_pandas(), animation_config)
        st.markdown("### Play Animation Demo")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Legend")
        show_play_legend(player_role_colors)


if __name__ == "__main__":
    # Pass in your Databricks client instance here
    # Example: from my_package import databricks_client
    main(databricks_client)
