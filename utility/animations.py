import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from colors import nfl_colors
from plotly.subplots import make_subplots


def create_bdb_field_figure(
    use_subplots: bool = False,
    rows: int = 1,
    cols: int = 1,
    subplot_row: int = 1,
    subplot_col: int = 1,
):
    field_length = 120
    field_width = 53.3

    if use_subplots:
        fig = make_subplots(rows=rows, cols=cols)
    else:
        fig = go.Figure()

    # Field boundary
    field_shape = go.layout.Shape(
        type="rect",
        x0=0,
        x1=field_length,
        y0=0,
        y1=field_width,
        line=dict(color="black", width=2),
        fillcolor="white",
        layer="below",
    )

    if use_subplots:
        fig.add_shape(field_shape, row=subplot_row, col=subplot_col)
    else:
        fig.add_shape(field_shape)

    # Yard lines every 5 yards
    for x in range(10, 111, 5):
        line = go.layout.Shape(
            type="line",
            x0=x,
            x1=x,
            y0=0,
            y1=field_width,
            line=dict(color="lightgray", width=1),
            layer="below",
        )
        if use_subplots:
            fig.add_shape(line, row=subplot_row, col=subplot_col)
        else:
            fig.add_shape(line)

    # Hash marks (two rows)
    for x in range(11, 110):
        for y in [23.366, 29.934]:
            line = go.layout.Shape(
                type="line",
                x0=x,
                x1=x,
                y0=y,
                y1=(y + 0.4 if y < 26 else y - 0.4),
                line=dict(color="black", width=1),
            )
            if use_subplots:
                fig.add_shape(line, row=subplot_row, col=subplot_col)
            else:
                fig.add_shape(line)

    # End zones
    for x0, x1 in [(0, 10), (110, 120)]:
        endzone = go.layout.Shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=0,
            y1=field_width,
            fillcolor="lightgray",
            opacity=0.1,
            line=dict(width=0),
        )
        if use_subplots:
            fig.add_shape(endzone, row=subplot_row, col=subplot_col)
        else:
            fig.add_shape(endzone)

    # Yard line numbers
    yard_numbers = list(range(10, 60, 10)) + list(range(50, 0, -10))
    yard_positions = list(range(10, 110, 10))
    for yard, x in zip(yard_numbers, yard_positions):
        annotations = [
            dict(
                x=x,
                y=2,
                text=str(yard),
                showarrow=False,
                font=dict(size=12, color="black"),
            ),
            dict(
                x=x,
                y=field_width - 2,
                text=str(yard),
                showarrow=False,
                font=dict(size=12, color="black"),
                textangle=180,
            ),
        ]
        for ann in annotations:
            fig.add_annotation(ann)

    # Layout config
    fig.update_layout(
        width=1200,
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            range=[0, field_length],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            range=[0, field_width], showgrid=False, zeroline=False, showticklabels=False
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def add_los_and_first_down(fig, play_df, row: int = None, col: int = None):
    play = play_df.iloc[0]  # Assume this is one row (single play metadata)

    los_x = play["absoluteYardlineNumber"]
    yards_to_go = play["yardsToGo"]
    direction = play["playDirection"].lower()

    if direction == "right":
        first_down_x = los_x + yards_to_go
    elif direction == "left":
        first_down_x = los_x - yards_to_go
    else:
        raise ValueError(f"Unknown play direction: {direction}")

    los_shape = dict(
        type="line",
        x0=los_x,
        x1=los_x,
        y0=0,
        y1=53.3,
        line=dict(color="darkblue", width=3),
        name="Line of Scrimmage",
    )

    fd_shape = dict(
        type="line",
        x0=first_down_x,
        x1=first_down_x,
        y0=0,
        y1=53.3,
        line=dict(color="yellow", width=3),
        name="Yard to Gain",
    )

    if row is not None and col is not None:
        fig.add_shape(los_shape, row=row, col=col)
        fig.add_shape(fd_shape, row=row, col=col)
    else:
        fig.add_shape(los_shape)
        fig.add_shape(fd_shape)

    return fig


def gameplay_trace_func(row, nfl_colors=nfl_colors, row_idx=1, col_idx=1):
    """
    Trace function that returns a dict compatible with subplot-aware create_trace_hash().
    """
    if row["displayName"].lower() == "football":
        trace = go.Scatter(
            x=[row["x"]],
            y=[row["y"]],
            mode="markers",
            marker=dict(size=10, color="saddlebrown", symbol="circle"),
            name="Football",
            hoverinfo="skip",
            showlegend=False,
        )
    else:
        club = row.get("club", "")
        color = nfl_colors.get(club, "#888888")
        trace = go.Scatter(
            x=[row["x"]],
            y=[row["y"]],
            mode="markers+text",
            marker=dict(size=24, color=color, line=dict(width=1, color="white")),
            text=[str(int(row["jerseyNumber"]))],
            textposition="middle center",
            textfont=dict(color="white", size=10),
            name=club,
            hoverinfo="text",
            showlegend=False,
        )

    return {"trace": trace, "row": row_idx, "col": col_idx}


def ball_carrier_circle_trace_func(row, row_idx=1, col_idx=1):
    """
    Adds a translucent circle around the ball carrier for a given frame row.
    Returns a subplot-aware trace dict only if the row is the ball carrier.
    """
    if row["nflId"] != row["ballCarrierId"]:
        return None  # Skip non-carriers

    trace = go.Scatter(
        x=[row["x"]],
        y=[row["y"]],
        mode="markers",
        marker=dict(
            size=40,
            color="rgba(255, 0, 0, 0.25)",
            line=dict(width=2, color="red"),
            symbol="circle",
        ),
        name="Ball Carrier",
        hoverinfo="skip",
        showlegend=False,
    )

    return {"trace": trace, "row": row_idx, "col": col_idx}


def gameplay_trace_func_df(df, nfl_colors=nfl_colors, row_idx=1, col_idx=1):
    """
    Trace function that returns a dict compatible with subplot-aware create_trace_hash().
    """
    df_football = df.query("displayName=='football'").reset_index(drop=True)
    df_players = df.query("displayName!='football'").reset_index(drop=True)
    df_players["plot_color"] = df_players["club"].map(nfl_colors)

    trace_football = go.Scatter(
        x=df_football["x"],
        y=df_football["y"],
        mode="markers",
        marker=dict(size=10, color="saddlebrown", symbol="circle"),
        name="Football",
        hoverinfo="skip",
        showlegend=False,
    )
    trace_players = go.Scatter(
        x=df_players["x"],
        y=df_players["y"],
        mode="markers+text",
        marker=dict(
            size=24, color=df_players["plot_color"], line=dict(width=1, color="white")
        ),
        text=[df_players["jerseyNumber"].astype(int).astype(str)],
        textposition="middle center",
        textfont=dict(color="white", size=10),
        name="play",
        hoverinfo="text",
        showlegend=False,
    )

    return {"trace": [trace_football, trace_players], "row": row_idx, "col": col_idx}


def create_trace_hash(df: pd.DataFrame, trace_func, metric=False):
    trace_dict = dict()
    if not metric:
        for frame_id in df["frameId"].unique():
            df_frame = df.query(f"frameId=={frame_id}").reset_index(drop=True)
            traces = []
            for _, row in df_frame.iterrows():
                out = trace_func(row)
                if out is None:
                    continue
                if isinstance(out, dict):
                    trace = out["trace"]
                    row_idx = out.get("row", 1)
                    col_idx = out.get("col", 1)
                    traces.append((trace, row_idx, col_idx))
                else:
                    traces.append((out, 1, 1))  # fallback
            trace_dict[frame_id] = traces
        return trace_dict
    else:
        print(metric)
        return trace_func(df)


def create_frames(trace_hash):
    frames = []
    for frame_id, trace_tuples in trace_hash.items():
        frame_data = []
        for trace, _, _ in trace_tuples:
            frame_data.append(trace)
        frames.append(go.Frame(name=str(frame_id), data=frame_data))
    return frames


def add_initial_traces(fig, trace_dict):
    first_frame = list(trace_dict.values())[0]
    for trace, row, col in first_frame:
        try:
            fig.add_trace(trace, row=row, col=col)
        except Exception:
            fig.add_trace(trace)


def animate_play(fig, frames, config=None):
    if not fig.data and frames:
        fig.add_traces(frames[0].data)
    config = config or {}
    fig.frames = frames

    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": config.get("play_label", "Play"),
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": config.get("duration", 100),
                                    "redraw": config.get("redraw", False),
                                },
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": config.get("pause_label", "Pause"),
                        "method": "animate",
                        "args": [[None], {"mode": "immediate"}],
                    },
                ],
                "direction": "left",
                "x": 0.1,
                "y": 0,
                "showactive": False,
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "method": "animate",
                        "args": [[f.name], {"mode": "immediate"}],
                        "label": f.name,
                    }
                    for f in frames
                ],
                "x": 0.1,
                "y": -0.05,
                "currentvalue": {"prefix": config.get("slider_prefix", "Frame: ")},
            }
        ],
    )

    return fig


def ball_carrier_speed_trace_func(df, row_idx=1, col_idx=2):
    trace_dict = {}

    frame_ids = sorted(df["frameId"].unique())
    x_vals = []
    y_vals = []

    for frame_id in frame_ids:
        current_frame = df[df["frameId"] == frame_id]
        if not current_frame.empty:
            x_vals.append(frame_id)
            y_vals.append(current_frame.iloc[0]["bcs"])

        trace = go.Scatter(
            x=x_vals.copy(),
            y=y_vals.copy(),
            mode="lines+markers",
            line=dict(color="red"),
            marker=dict(size=6),
            name="Ball Carrier Speed",
            showlegend=(True),
        )

        # IMPORTANT: must use same trace index and name across all frames
        trace.uid = "ball_carrier_speed"
        trace_dict[frame_id] = [(trace, row_idx, col_idx)]

    return trace_dict
