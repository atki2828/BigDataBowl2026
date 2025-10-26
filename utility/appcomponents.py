import streamlit.components.v1 as components


def set_speed_config(speed: str, max_frames: int) -> dict:
    duration = (1 / speed) * max_frames

    """Set animation speed configuration based on user selection."""
    return {
        "duration": duration,
        "redraw": False,
        "slider_prefix": "Frame: ",
        "play_label": "▶",
        "pause_label": "⏸",
    }


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
