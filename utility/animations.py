from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from .colors import nfl_colors


def animate_play(
    fig: go.Figure, frames: List[go.Frame], config: dict = None
) -> go.Figure:

    if not fig.data and frames:
        fig.add_traces(frames[0].data)
    config = config or {}
    fig.frames = frames

    fig.update_layout(
        margin=dict(b=140),  # extra bottom margin for buttons + slider
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
                                    "duration": config.get("duration", 33),
                                    "redraw": config.get("redraw", False),
                                },
                                "transition": {"duration": 0},
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
                "x": 0.05,
                "y": -0.23,  # further down, under the slider
                "xanchor": "center",
                "yanchor": "top",
                "showactive": False,
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "method": "animate",
                        "args": [
                            [f.name],
                            {
                                "mode": "immediate",
                                "frame": {
                                    "duration": config.get("duration", 33),
                                    "redraw": config.get("redraw", False),
                                },
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": f.name,
                    }
                    for f in frames
                ],
                "x": 0.1,
                "y": -0.12,  # keep slider just below plot
                "currentvalue": {"prefix": config.get("slider_prefix", "Frame: ")},
            }
        ],
    )
    return fig


# --------------------------
# Field (always subplots)
# --------------------------
class Field:
    """Represents the static portion of the playing field for the animation."""

    def __init__(
        self,
        play_df: pd.DataFrame,
        row: int = 1,
        col: int = 1,
        subplot_rows: int = 1,
        subplot_cols: int = 1,
        width: int = 1200,
        height: int = 600,
    ):
        self.play_df = play_df
        self.row = row
        self.col = col
        self.subplot_rows = subplot_rows
        self.subplot_cols = subplot_cols

        self.field_fig = make_subplots(rows=subplot_rows, cols=subplot_cols)
        self._draw_field()
        self._add_los_and_first_down()

        field_length = 120
        field_width = 53.3

        self.field_fig.update_layout(
            width=width,
            height=height,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        self.field_fig.update_xaxes(
            range=[0, field_length],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            row=self.row,
            col=self.col,
        )
        self.field_fig.update_yaxes(
            range=[0, field_width],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            row=self.row,
            col=self.col,
        )

    def _draw_field(self) -> None:
        field_length = 120
        field_width = 53.3

        # Boundary
        self.field_fig.add_shape(
            type="rect",
            x0=0,
            x1=field_length,
            y0=0,
            y1=field_width,
            line=dict(color="black", width=2),
            fillcolor="white",
            layer="below",
            row=self.row,
            col=self.col,
        )

        # Yard lines every 5 yards
        for x in range(10, 111, 5):
            self.field_fig.add_shape(
                type="line",
                x0=x,
                x1=x,
                y0=0,
                y1=field_width,
                line=dict(color="lightgray", width=1),
                layer="below",
                row=self.row,
                col=self.col,
            )

        # Hash marks (two rows)
        for x in range(11, 110):
            for y in [23.366, 29.934]:
                self.field_fig.add_shape(
                    type="line",
                    x0=x,
                    x1=x,
                    y0=y,
                    y1=(y + 0.4 if y < 26 else y - 0.4),
                    line=dict(color="black", width=1),
                    row=self.row,
                    col=self.col,
                )

        # End zones
        for x0, x1 in [(0, 10), (110, 120)]:
            self.field_fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=0,
                y1=field_width,
                fillcolor="lightgray",
                opacity=0.1,
                line=dict(width=0),
                row=self.row,
                col=self.col,
            )

        yard_positions = list(range(10, 111, 10))  # 10, 20, ..., 110

        for x in yard_positions:
            if x in (10, 110):
                label, size = "G", 14
            elif x <= 60:
                label, size = str(x - 10), 12
            else:
                label, size = str(110 - x), 12

            # bottom numbers
            self.field_fig.add_annotation(
                x=x,
                y=2,
                text=label,
                showarrow=False,
                font=dict(size=size, color="black"),
                xanchor="center",
                yanchor="middle",
                row=self.row,
                col=self.col,
            )
            # top numbers (rotated)
            self.field_fig.add_annotation(
                x=x,
                y=field_width - 2,
                text=label,
                showarrow=False,
                font=dict(size=size, color="black"),
                textangle=180,
                xanchor="center",
                yanchor="middle",
                row=self.row,
                col=self.col,
            )

    def _add_los_and_first_down(self) -> None:
        play = self.play_df.iloc[0]
        los_x = float(play["absoluteYardlineNumber"])
        yards_to_go = float(play["yardsToGo"])
        direction = str(play["playDirection"]).lower()

        if direction == "right":
            first_down_x = los_x + yards_to_go
        elif direction == "left":
            first_down_x = los_x - yards_to_go
        else:
            raise ValueError(f"Unknown play direction: {direction}")

        self.field_fig.add_shape(
            type="line",
            x0=los_x,
            x1=los_x,
            y0=0,
            y1=53.3,
            line=dict(color="darkblue", width=3),
            row=self.row,
            col=self.col,
        )
        self.field_fig.add_shape(
            type="line",
            x0=first_down_x,
            x1=first_down_x,
            y0=0,
            y1=53.3,
            line=dict(color="yellow", width=3),
            row=self.row,
            col=self.col,
        )


# --------------------------
# Trace plumbing
# --------------------------
@dataclass
class TraceConfig:
    frame_df: pd.DataFrame
    trace_func: Callable[[pd.DataFrame], go.Scatter]
    row: int = 1
    col: int = 1
    frame_id: int = field(init=False)
    trace: go.Scatter = field(init=False)

    def __post_init__(self):
        self.frame_id = int(self.frame_df["frameId"].iloc[0])
        self.trace = self.trace_func(self.frame_df)


def build_trace_configs(
    play_df: pd.DataFrame,
    trace_func: Callable[[pd.DataFrame], go.Scatter],
    row: int = 1,
    col: int = 1,
) -> List[TraceConfig]:
    """
    Build TraceConfig objects for each frame.
    """
    configs: List[TraceConfig] = []
    for _, df in play_df.groupby("frameId", sort=True):
        configs.append(
            TraceConfig(frame_df=df, trace_func=trace_func, row=row, col=col)
        )
    return configs


# --------------------------
# Animator
# --------------------------
class PlayAnimator:
    """Creates an animated play figure with traces spanning multiple subplots."""

    def __init__(
        self,
        field: Field,
        animation_config: Optional[dict] = None,
        trace_configs: Optional[List[TraceConfig]] = None,
    ):
        if field is None:
            raise ValueError("Field must be provided to create the animation.")

        self.field = field
        self.animation_config = animation_config
        self.trace_configs = trace_configs

        self.frame_configs = self._group_trace_configs(self.trace_configs)
        self.frames = self._create_frames(self.frame_configs)

    def _group_trace_configs(
        self, trace_configs: List[TraceConfig]
    ) -> Dict[int, List[TraceConfig]]:
        frame_configs: Dict[int, List[TraceConfig]] = {}
        for config in trace_configs:
            frame_configs.setdefault(config.frame_id, []).append(config)
        return frame_configs

    def _axis_names_for_cell(self, row: int, col: int) -> Tuple[str, str]:
        """
        Map (row, col) -> ('x' or 'xN', 'y' or 'yN') for frames.
        make_subplots names axes left-to-right, top-to-bottom:
          index = (row-1)*subplot_cols + col
        """
        idx = (row - 1) * self.field.subplot_cols + col
        xaxis = "x" if idx == 1 else f"x{idx}"
        yaxis = "y" if idx == 1 else f"y{idx}"
        return xaxis, yaxis

    def _clone_scatter_with_axes(
        self, trace: go.Scatter, row: int, col: int
    ) -> go.Scatter:
        """Return a new Scatter with the correct subplot xaxis/yaxis assigned."""
        xaxis, yaxis = self._axis_names_for_cell(row, col)
        # Construct a new Scatter; reuse common properties
        return go.Scatter(
            x=trace.x,
            y=trace.y,
            mode=trace.mode,
            marker=trace.marker,
            line=getattr(trace, "line", None),
            text=getattr(trace, "text", None),
            textposition=getattr(trace, "textposition", None),
            textfont=getattr(trace, "textfont", None),
            hoverinfo=getattr(trace, "hoverinfo", None),
            showlegend=getattr(trace, "showlegend", False),
            name=getattr(trace, "name", None),
            xaxis=xaxis,
            yaxis=yaxis,
        )

    def _create_frames(
        self, frame_configs: Dict[int, List[TraceConfig]]
    ) -> List[go.Frame]:
        frames: List[go.Frame] = []
        for frame_id in sorted(frame_configs.keys()):
            configs = frame_configs[frame_id]
            frame_data = [
                self._clone_scatter_with_axes(cfg.trace, cfg.row, cfg.col)
                for cfg in configs
            ]
            frames.append(go.Frame(data=frame_data, name=str(frame_id)))
        return frames

    def create_animation(self) -> go.Figure:
        fig = self.field.field_fig

        # Add initial traces in the right subplots
        first_fid = min(self.frame_configs.keys())
        for cfg in self.frame_configs[first_fid]:
            fig.add_trace(cfg.trace, row=cfg.row, col=cfg.col)

        # Attach frames
        fig.frames = self.frames

        return animate_play(fig, self.frames, config=self.animation_config)
