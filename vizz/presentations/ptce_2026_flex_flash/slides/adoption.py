from __future__ import annotations

from typing import TYPE_CHECKING

from manim import *

if TYPE_CHECKING:
    from vizz.presentations.components import SlideBase

from vizz.presentations.charts import cumulative_line_chart, load_csv
from vizz.presentations.ptce_2026_flex_flash import DATA_ROOT


def _load_adoption_data() -> tuple[list[str], list[int], list[int]]:
    rows = load_csv(DATA_ROOT / "flex_adoption_monthly.csv")
    _MONTH_ABBR = {
        "01": "Jan",
        "02": "Feb",
        "03": "Mar",
        "04": "Apr",
        "05": "May",
        "06": "Jun",
        "07": "Jul",
        "08": "Aug",
        "09": "Sep",
        "10": "Oct",
        "11": "Nov",
        "12": "Dec",
    }
    months: list[str] = []
    for r in rows:
        mm, yy = r["month"][5:7], r["month"][2:4]
        is_last = r is rows[-1]
        months.append(
            f"{_MONTH_ABBR[mm]}\n'{yy}"
            if mm in ("01", "04", "07", "10") or is_last
            else ""
        )
    counts = [int(r["count"]) for r in rows]
    cumulative = [int(r["cumulative"]) for r in rows]
    return months, counts, cumulative


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = scene.section_header("Rapid Adoption")
    months, counts, cumulative = _load_adoption_data()

    cum_chart = cumulative_line_chart(
        t,
        months,
        [float(c) for c in cumulative],
        line_color=t.accent_primary,
        y_label="Cumulative repos",
        x_length=6.5,
        y_length=4.2,
        show_values=True,
        value_font_size=10,
    )

    chart_frame = scene.panel(width=7.9, height=5.75)
    chart_title = scene.meta_text(
        "Adoption since launch",
        font_size=max(t.meta_font_size + 1, 15),
        color=t.accent_primary,
        uppercase=False,
    )
    cum_chart.scale_to_fit_width(chart_frame.width - 0.7)
    if cum_chart.height > chart_frame.height - 1.2:
        cum_chart.scale_to_fit_height(chart_frame.height - 1.2)

    chart_title.move_to(chart_frame.get_top() + DOWN * 0.34)
    chart_title.align_to(chart_frame.get_left() + RIGHT * 0.32, LEFT)
    cum_chart.move_to(chart_frame.get_center() + DOWN * 0.15)

    chart_panel = Group(chart_frame, chart_title, cum_chart)

    bullets = scene.bullet_list(
        "1,000+ repos adopted the API",
        "Dozens of papers validated\nthe programming model",
        "Researchers could prototype\ncustom attention quickly",
        font_size=25,
    )
    bullets.arrange(DOWN, aligned_edge=LEFT, buff=0.4)

    content = Group(chart_panel, bullets).arrange(RIGHT, buff=0.55, aligned_edge=UP)
    content.next_to(header, DOWN, buff=0.26)
    content.set_x(0)
    content.shift(DOWN * 0.12)

    scene.play(FadeIn(header[0], shift=UP * 0.15), Create(header[1]))
    scene.play(FadeIn(chart_panel, shift=UP * 0.12))
    scene.next_slide()
    for row in bullets:
        scene.play(FadeIn(row, shift=RIGHT * 0.12))
        scene.next_slide()
    scene.clear_stage()
