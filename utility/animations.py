# animations.py
import os

import pandas as pd
import plotly.express as px


def create_dummy_football_animation(output_path: str):
    # Dummy data for 2 teams and 10 frames
    data = []
    for frame in range(10):  # 10 time steps
        for player_id in range(1, 12):  # Team A
            data.append(
                {
                    "frame": frame,
                    "x": 10 + frame + player_id,
                    "y": 20 + player_id,
                    "team": "A",
                    "player": f"A{player_id}",
                }
            )
        for player_id in range(1, 12):  # Team B
            data.append(
                {
                    "frame": frame,
                    "x": 60 - frame - player_id,
                    "y": 30 + player_id,
                    "team": "B",
                    "player": f"B{player_id}",
                }
            )

    df = pd.DataFrame(data)

    # Create Plotly animation
    fig = px.scatter(
        df,
        x="x",
        y="y",
        animation_frame="frame",
        animation_group="player",
        color="team",
        text="player",
        range_x=[0, 120],
        range_y=[0, 53.3],
        title="Dummy Football Play Animation",
    )

    fig.update_traces(marker=dict(size=12), textposition="middle center")
    fig.update_layout(
        width=800,
        height=450,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
    )

    # Save as HTML
    fig.write_html(output_path)
    print(f"Animation saved at: {output_path}")


if __name__ == "__main__":

    output_dir = os.path.join(".", "animations")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "dummy_football_animation.html")

    create_dummy_football_animation(output_file)
