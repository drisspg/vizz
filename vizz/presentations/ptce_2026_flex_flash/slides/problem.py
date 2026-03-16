from __future__ import annotations

import re
from typing import TYPE_CHECKING

from manim import *

if TYPE_CHECKING:
    from vizz.presentations.components import SlideBase

from vizz.presentations.charts import (
    Series,
    cumulative_line_chart,
    grouped_bar_chart,
    load_csv,
    seq_label,
)
from vizz.presentations.ptce_2026_flex_flash import DATA_ROOT

_SHAPE_RE = re.compile(r"\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)")
_MONTH_ABBR = {
    "01": "J",
    "02": "F",
    "03": "M",
    "04": "A",
    "05": "M",
    "06": "J",
    "07": "J",
    "08": "A",
    "09": "S",
    "10": "O",
    "11": "N",
    "12": "D",
}


def _adoption_chart(scene: SlideBase) -> VGroup:
    t = scene.theme
    rows = load_csv(DATA_ROOT / "flex_adoption_monthly.csv")
    months = [_MONTH_ABBR.get(r["month"][5:7], "") for r in rows]
    cumulative = [float(r["cumulative"]) for r in rows]
    return cumulative_line_chart(
        t,
        months,
        cumulative,
        line_color=t.accent_primary,
        y_label="Repos",
        x_length=4.5,
        y_length=3.6,
        show_values=False,
        value_font_size=9,
    )


def _fa3_vs_flex_chart(scene: SlideBase) -> VGroup:
    t = scene.theme
    rows = load_csv(DATA_ROOT / "flex_vs_fav3_causal.csv")
    rows.sort(key=lambda r: int(_SHAPE_RE.search(r["shape(B,Hq,M,Hkv,N,D)"]).group(3)))

    labels = [
        seq_label(int(_SHAPE_RE.search(r["shape(B,Hq,M,Hkv,N,D)"]).group(3)))
        for r in rows
    ]
    flex_fwd = [float(r["flex_attn_fwd_TFlops"]) for r in rows]
    fa3_fwd = [float(r["fav3_fwd_TFlops"]) for r in rows]

    return grouped_bar_chart(
        t,
        [
            Series("FlexAttention", flex_fwd, t.accent_primary),
            Series("FA3", fa3_fwd, "#8a7256"),
        ],
        labels,
        y_label="TFlops",
        x_length=4.5,
        y_length=3.6,
        show_legend=True,
        legend_font_size=10,
    )


def build(scene: SlideBase) -> None:
    header = scene.section_header(
        "FlexAttention worked. Performance became the bottleneck."
    )

    adoption = _adoption_chart(scene)
    left_group = scene.labeled_panel(
        "Adoption", width=6.0, height=5.4, content=adoption
    )

    gap = _fa3_vs_flex_chart(scene)
    right_group = scene.labeled_panel(
        "The gap widened", width=6.2, height=5.4, content=gap
    )

    Group(left_group, right_group).arrange(RIGHT, buff=0.5).move_to(
        ORIGIN + DOWN * 0.15
    )

    caption = scene.bullet_list(
        "1,000+ repos and dozens of papers validated the API",
        "Researchers could prototype quickly but hit a performance wall",
        "On Blackwell, the old Triton path fell even further behind",
        font_size=24,
    ).to_edge(DOWN, buff=0.4)

    scene.play(FadeIn(header[0], shift=UP * 0.15), Create(header[1]))
    scene.play(
        LaggedStart(
            FadeIn(left_group, shift=LEFT * 0.18),
            FadeIn(right_group, shift=RIGHT * 0.18),
            lag_ratio=0.14,
        )
    )
    scene.play(
        LaggedStart(*[FadeIn(row, shift=UP * 0.1) for row in caption], lag_ratio=0.12)
    )
    scene.next_slide()
    scene.clear_stage()
