"""This module provides utilities for creating animations of NFL plays using Plotly."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
from plotly.subplots import make_subplots

from .colors import nfl_colors

Number = Union[int, float]


def _is_num(x) -> bool:
    """Check if x is a number (int or float) and not NaN."""
    try:
        return (
            x is not None
            and not (isinstance(x, float) and math.isnan(x))
            and isinstance(float(x), (int, float))
        )
    except Exception:
        return False


def _flatten_numeric(arr: Iterable) -> List[float]:
    """Flatten an iterable and return only numeric values as floats."""
    if arr is None:
        return []
    out = []
    for v in arr:
        if _is_num(v):
            out.append(float(v))
    return out


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
    trace_func: Callable[[pd.DataFrame], BaseTraceType]
    row: int = 1
    col: int = 1
    frame_id: int = field(init=False)
    trace: BaseTraceType = field(init=False)

    def __post_init__(self):
        self.frame_id = int(self.frame_df["frameId"].iloc[0])
        self.trace = self.trace_func(self.frame_df)


def build_trace_configs(
    play_df: pd.DataFrame,
    trace_func: Callable[[pd.DataFrame], BaseTraceType],
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
    """
    Creates an animated play figure with traces spanning multiple subplots.
    - Groups provided TraceConfig by frameId
    - Builds plotly Frames
    - Sets per-subplot axis ranges automatically, with optional overrides
    """

    def __init__(
        self,
        field: "Field",
        animation_config: Optional[dict] = None,
        trace_configs: Optional[List["TraceConfig"]] = None,
        *,
        pad_ratio: float = 0.05,
        fixed_axis: Optional[
            Dict[Tuple[int, int], Dict[str, Tuple[Number, Number]]]
        ] = None,
    ):
        """
        Args:
            field: Field instance (provides .field_fig, .subplot_cols, .play_df, etc.)
            animation_config: dict with duration/redraw/labels, etc.
            trace_configs: list of TraceConfig(trace: go.Scatter, frame_id: int, row: int, col: int)
            pad_ratio: padding fraction for inferred y-range (and x-range if inferred from trace data)
            fixed_axis: optional per-cell axis overrides, keyed by (row, col) -> {"x": (min,max), "y": (min,max)}
        """
        if field is None:
            raise ValueError("Field must be provided to create the animation.")

        self.field = field
        self.animation_config = animation_config or {}
        self.trace_configs = trace_configs or []

        self.pad_ratio = pad_ratio
        self.fixed_axis = fixed_axis or {}

        self.frame_configs = self._group_trace_configs(self.trace_configs)
        self.frames = self._create_frames(self.frame_configs)

    # ---------- helpers ----------

    def _group_trace_configs(
        self, trace_configs: List["TraceConfig"]
    ) -> Dict[int, List["TraceConfig"]]:
        frame_configs: Dict[int, List["TraceConfig"]] = {}
        for cfg in trace_configs:
            frame_configs.setdefault(cfg.frame_id, []).append(cfg)
        return frame_configs

    def _axis_names_for_cell(self, row: int, col: int) -> Tuple[str, str]:
        """
        Map (row, col) -> ('x' or 'xN', 'y' or 'yN') as used by make_subplots:
          idx = (row-1)*subplot_cols + col
        """
        idx = (row - 1) * self.field.subplot_cols + col
        xaxis = "x" if idx == 1 else f"x{idx}"
        yaxis = "y" if idx == 1 else f"y{idx}"
        return xaxis, yaxis

    def _clone_scatter_with_axes(
        self, trace: go.Scatter, row: int, col: int
    ) -> go.Scatter:
        xaxis, yaxis = self._axis_names_for_cell(row, col)
        return go.Scatter(
            x=trace.x,
            y=trace.y,
            mode=trace.mode,
            marker=getattr(trace, "marker", None),
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
        self, frame_configs: Dict[int, List["TraceConfig"]]
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

    # ---------- axis inference ----------

    def _infer_axis_ranges(
        self,
    ) -> Dict[Tuple[int, int], Dict[str, Tuple[float, float]]]:
        """
        Scan ALL traces across ALL frames by subplot cell, infer numeric x/y ranges.
        If a cell has no numeric x but we have frames, fall back to [min(frameId), max(frameId)].
        Apply pad_ratio around inferred ranges.
        """
        # collect numeric x/y by cell
        by_cell = defaultdict(lambda: {"x": [], "y": []})

        # Prefer reading from all frame traces (so cumulative metric panes are covered)
        for cfg in self.trace_configs:
            if cfg.row == 1 and cfg.col == 1:
                # Do not update axis on field and always put field in (1,1)
                continue
            if isinstance(cfg.trace, BaseTraceType):
                xs = _flatten_numeric(cfg.trace.x)
                ys = _flatten_numeric(cfg.trace.y)
                if xs:
                    by_cell[(cfg.row, cfg.col)]["x"].extend(xs)
                if ys:
                    by_cell[(cfg.row, cfg.col)]["y"].extend(ys)

        inferred: Dict[Tuple[int, int], Dict[str, Tuple[float, float]]] = {}

        min_fid = min(self.frame_configs.keys()) if self.frame_configs else None
        max_fid = max(self.frame_configs.keys()) if self.frame_configs else None

        for cell, vals in by_cell.items():
            xr = None
            yr = None

            if vals["x"]:
                xmin, xmax = min(vals["x"]), max(vals["x"])
                pad = self.pad_ratio * max(1e-12, xmax - xmin)
                xr = (xmin - pad, xmax + pad)
            elif min_fid is not None and max_fid is not None:
                # fallback to frameId range
                xr = (float(min_fid), float(max_fid))

            if vals["y"]:
                ymin, ymax = min(vals["y"]), max(vals["y"])
                pad = self.pad_ratio * max(1e-12, ymax - ymin)
                yr = (ymin - pad, ymax + pad)

            rngs = {}
            if xr is not None:
                rngs["x"] = xr
            if yr is not None:
                rngs["y"] = yr

            if rngs:
                inferred[cell] = rngs

        # apply explicit overrides last
        for cell, axes in self.fixed_axis.items():
            if cell not in inferred:
                inferred[cell] = {}
            for axis_name, rng in axes.items():
                inferred[cell][axis_name] = (float(rng[0]), float(rng[1]))

        return inferred

    def _apply_axis_ranges(self, fig: go.Figure) -> None:
        rngs = self._infer_axis_ranges()
        for (row, col), axes in rngs.items():
            if "x" in axes:
                fig.update_xaxes(range=list(axes["x"]), row=row, col=col)
            if "y" in axes:
                fig.update_yaxes(range=list(axes["y"]), row=row, col=col)

    # ---------- private animate ----------

    def _animate_play(self, fig: go.Figure) -> go.Figure:
        """
        Internal method that wires buttons/slider and attaches frames.
        Mirrors your previous animate_play(), driven by self.animation_config.
        """
        cfg = self.animation_config or {}
        if not fig.data and self.frames:
            fig.add_traces(self.frames[0].data)

        fig.frames = self.frames

        fig.update_layout(
            margin=dict(b=140),
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": cfg.get("play_label", "Play"),
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {
                                        "duration": cfg.get("duration", 33),
                                        "redraw": cfg.get("redraw", False),
                                    },
                                    "transition": {"duration": 0},
                                    "fromcurrent": True,
                                },
                            ],
                        },
                        {
                            "label": cfg.get("pause_label", "Pause"),
                            "method": "animate",
                            "args": [[None], {"mode": "immediate"}],
                        },
                    ],
                    "direction": "left",
                    "x": 0.05,
                    "y": -0.23,
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
                                        "duration": cfg.get("duration", 33),
                                        "redraw": cfg.get("redraw", False),
                                    },
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": f.name,
                        }
                        for f in self.frames
                    ],
                    "x": 0.1,
                    "y": -0.12,
                    "currentvalue": {"prefix": cfg.get("slider_prefix", "Frame: ")},
                }
            ],
        )
        return fig

    # ---------- public API ----------

    def create_animation(self) -> go.Figure:
        """
        Returns a fully wired animation figure with axis ranges set per subplot.
        """
        fig = self.field.field_fig

        # Add initial traces in the correct subplots
        if self.frame_configs:
            first_fid = min(self.frame_configs.keys())
            for cfg in self.frame_configs[first_fid]:
                fig.add_trace(cfg.trace, row=cfg.row, col=cfg.col)

        # Attach frames and controls
        fig = self._animate_play(fig)

        # Auto ranges (with optional overrides)
        self._apply_axis_ranges(fig)

        return fig
