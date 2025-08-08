from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from .colors import nfl_colors


def create_bdb_field_figure(
    use_subplots: bool = False,
    rows: int = 1,
    cols: int = 1,
    subplot_row: int = 1,
    subplot_col: int = 1,
) -> go.Figure:
    """Creates a field figure for the Big Data Bowl.

    Args:
        use_subplots (bool, optional): Whether to use subplots. Defaults to False.
        rows (int, optional): Number of rows in the subplot. Defaults to 1.
        cols (int, optional): Number of columns in the subplot. Defaults to 1.
        subplot_row (int, optional): Row number of the subplot. Defaults to 1.
        subplot_col (int, optional): Column number of the subplot. Defaults to 1.

    Returns:
        go.Figure: The created field figure.
    """
    # Create figure
    fig = make_subplots(rows=rows, cols=cols) if use_subplots else go.Figure()

    field_length = 120
    field_width = 53.3
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


def animate_play(
    fig: go.Figure, frames: List[go.Frame], config: dict = None
) -> go.Figure:

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


class Field:
    """Represents the static portion of the playing field for the animation."""

    def __init__(
        self,
        play_df: pd.DataFrame,
        # TODO: maybe make this callabale part of the class
        field_fig_drawer: Callable = create_bdb_field_figure,
        use_subplots: bool = False,
        row: int = None,
        col: int = None,
    ):
        self.play_df = play_df
        self.field_fig = field_fig_drawer()
        self.field_fig = self.add_los_and_first_down()
        self.use_subplots = use_subplots
        self.row = row
        self.col = col

    def _add_los_and_first_down(self) -> go.Figure:
        """Adds the line of scrimmage and first down markers to the field figure.

        Returns:
            go.Figure: The updated figure with LOS and first down markers.
        """
        play = self.play_df.iloc[0]  # Assume this is one row (single play metadata)

        los_x = play["absoluteYardlineNumber"]
        yards_to_go = play["yardsToGo"]
        direction = play["playDirection"].lower()

        if direction == "right":
            first_down_x = los_x + yards_to_go
        elif direction == "left":
            first_down_x = los_x - yards_to_go
        else:
            raise ValueError(f"Unknown play direction: {direction}")

        # Hard Coded Colors And Line Shapes May Want to make this configurable
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

        if self.row is not None and self.col is not None:
            self.field_fig.add_shape(los_shape, row=self.row, col=self.col)
            self.field_fig.add_shape(fd_shape, row=self.row, col=self.col)
        else:
            self.field_fig.add_shape(los_shape)
            self.field_fig.add_shape(fd_shape)

        return self.field_fig


class TraceConfig:
    def __init__(
        self, row: pd.Series, trace_func: Callable, row_idx: int = 1, col_idx: int = 1
    ):
        """
        Initializes a TraceConfig instance for creating traces from a DataFrame.

        Args:
            trace_func (Callable): The function used to generate the trace.
            row_idx (int, optional): The row index for subplot positioning. Defaults to 1.
            col_idx (int, optional): The column index for subplot positioning. Defaults to 1.
        """
        self.frame_id = row["frameId"].iloc[0]
        self.trace = trace_func(row)
        self.row_idx = row_idx
        self.col_idx = col_idx


class PlayAnimator:
    """Handles the creation of an animated play figure with traces for players and ball carrier speed."""

    def __init__(
        self,
        play_df: pd.DataFrame,
        field: Field = None,
        animation_config: dict = None,
        trace_configs: List[TraceConfig] = None,
    ):
        self.play_df = play_df
        self.field = field
        self.animation_config = animation_config or {}
        self.frames = self._create_frames(trace_configs)

    def _create_frames(self, trace_configs: List[TraceConfig]) -> List[go.Frame]:
        return [
            go.Frame(data=[config.trace], name=str(config.frame_id))
            for config in trace_configs
        ]

    def create_animation(self) -> go.Figure:
        """Creates the animated play figure with traces for players and ball carrier speed.

        Returns:
            go.Figure: The animated play figure.
        """
        if not self.field:
            raise ValueError("Field must be provided to create the animation.")

        fig = self.field.field_fig

        # Add initial traces from trace_configs
        for config in self.frames[0].data:
            fig.add_trace(config)

        # Add frames to the figure
        fig.frames = self.frames

        return animate_play(fig, self.frames, config=self.animation_config)
