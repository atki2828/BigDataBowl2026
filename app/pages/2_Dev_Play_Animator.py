import polars as pl
import streamlit as st

from utility.dbx import DatabricksSQLClient

databricks_client = DatabricksSQLClient()

st.sidebar.title("Big Data Bowl Explorer")

# 1) Year (always available)
year = st.sidebar.selectbox("Select Year", list(range(2019, 2026)), index=None)

# --- Game dropdown ---
games_df, game_options, selected_game_id = pl.DataFrame(), [], None
if year:
    games_query = f"""
        SELECT gameId, homeTeamAbbr, visitorTeamAbbr, gameDate
        FROM workspace.bigdatabowl{year}.games
        ORDER BY gameDate
    """
    games_df = databricks_client.query_to_pl(games_query)
    game_options = games_df.select(
        (
            pl.col("gameId").cast(str)
            + " | "
            + pl.col("visitorTeamAbbr")
            + " @ "
            + pl.col("homeTeamAbbr")
            + " ("
            + pl.col("gameDate").cast(str)
            + ")"
        ).alias("label"),
        pl.col("gameId"),
    ).to_dicts()

game_choice = st.sidebar.selectbox(
    "Select Game",
    game_options,
    format_func=lambda x: x["label"] if isinstance(x, dict) else x,
    index=None,
    disabled=not year,
)
if game_choice:
    selected_game_id = game_choice["gameId"]

# --- Play dropdown ---
plays_df, play_options, selected_play_id = pl.DataFrame(), [], None
if selected_game_id:
    plays_query = f"""
        SELECT playId, quarter, down, yardsToGo, playDescription
        FROM workspace.bigdatabowl{year}.plays
        WHERE gameId = {selected_game_id}
        ORDER BY playId
    """
    plays_df = databricks_client.query_to_pl(plays_query)
    play_options = plays_df.to_dicts()

play_choice = st.sidebar.selectbox(
    "Select Play",
    play_options,
    format_func=lambda x: (
        (
            f"Play {x['playId']} (Q{x['quarter']} {x['down']} & {x['yardsToGo']}): "
            f"{x['playDescription'][:50]}..."
        )
        if isinstance(x, dict)
        else x
    ),
    index=None,
    disabled=not selected_game_id,
)
if play_choice:
    selected_play_id = play_choice["playId"]

# --- Player dropdown ---
players_df, player_options, selected_player_id = pl.DataFrame(), [], None
if selected_play_id:
    players_query = f"""
        SELECT DISTINCT p.nflId, p.displayName, p.position
        FROM workspace.bigdatabowl{year}.players p
        JOIN workspace.bigdatabowl{year}.tracking t
          ON p.nflId_str = t.nflId
        WHERE t.gameId = {selected_game_id}
          AND t.playId = {selected_play_id}
        ORDER BY p.displayName
    """
    players_df = databricks_client.query_to_pl(players_query)
    player_options = [
        {"nflId": None, "displayName": "None", "position": ""}
    ] + players_df.to_dicts()

player_choice = st.sidebar.selectbox(
    "Highlight Player",
    player_options,
    format_func=lambda x: (
        ("None" if x["nflId"] is None else f"{x['displayName']} ({x['position']})")
        if isinstance(x, dict)
        else x
    ),
    index=0,
    disabled=not selected_play_id,
)
if player_choice:
    selected_player_id = player_choice["nflId"]

# --- Metric dropdown ---
metrics = {
    "Ball Carrier Speed": ("bcs", "Ball Carrier Speed"),
    "Total Distance": ("dist", "Total Distance"),
}
metric_label = st.sidebar.selectbox(
    "Metric",
    list(metrics.keys()),
    index=None,
    disabled=not selected_play_id,
)
metric_key, metric_name = metrics[metric_label] if metric_label else (None, None)

# --- Main Panel ---
st.title("Big Data Bowl Play Explorer")

if selected_play_id and metric_label:
    st.write(
        f"**Year:** {year}, **Game:** {selected_game_id}, **Play:** {selected_play_id}"
    )
    st.write(f"**Metric:** {metric_name}")
    st.write(f"**Highlight Player:** {selected_player_id}")

    # Example where you'd slot in your animation functions
    # animate_play_df = create_animation_df(track_df, play_df, selected_game_id, selected_play_id)
    # fig = create_play_metric_fig(animate_play_df, animation_config)
    # st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Work through the sidebar selections to view the animation.")
