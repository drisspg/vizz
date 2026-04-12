from dataclasses import dataclass

from manim import config


@dataclass
class Theme:
    background: str = "#ffffff"
    text: str = "#111827"
    muted_text: str = "#334155"
    accent_primary: str = "#2563eb"
    accent_secondary: str = "#d97706"
    accent_success: str = "#15803d"
    accent_danger: str = "#b91c1c"
    panel_fill: str = "#f8fafc"
    panel_stroke: str = "#cbd5e1"
    divider: str = "#cbd5e1"
    display_font: str = "Georgia"
    sans_font: str = "Helvetica Neue"
    mono_font: str = "Menlo"
    title_font_size: int = 42
    body_font_size: int = 28
    bullet_font_size: int = 27
    code_font_size: int = 16
    meta_font_size: int = 16
    code_style: str = "vim"
    code_background: str = ""
    panel_corner_radius: float = 0.18
    panel_stroke_width: float = 2.0

    @property
    def max_title_width(self) -> float:
        return config.frame_width - 0.8


LIGHT_THEME = Theme()

FRONTIER_LIGHT_THEME = Theme(
    background="#f4f0e8",
    text="#223127",
    muted_text="#5d6a60",
    accent_primary="#617a69",
    accent_secondary="#7f8f86",
    accent_success="#6f8f7b",
    accent_danger="#8a7256",
    panel_fill="#f7f3ec",
    panel_stroke="#d5cec0",
    divider="#cfc6b7",
    title_font_size=40,
    body_font_size=26,
    bullet_font_size=24,
    code_font_size=15,
    meta_font_size=15,
    code_style="tango",
    code_background="#eee9df",
    panel_corner_radius=0.06,
    panel_stroke_width=1.5,
)

PYTORCH_THEME = Theme(
    background="#ffffff",
    text="#333333",
    muted_text="#555555",
    accent_primary="#EE4C2C",
    accent_secondary="#2196F3",
    accent_success="#4CAF50",
    accent_danger="#7B1FA2",
    panel_fill="#f8f8f8",
    panel_stroke="#d0d0d0",
    divider="#d0d0d0",
)
