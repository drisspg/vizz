from __future__ import annotations

import re
from typing import TYPE_CHECKING

from manim import *

if TYPE_CHECKING:
    from vizz.presentations.components import SlideBase

from vizz.presentations.charts import (
    Series,
    _build_legend,
    grouped_bar_chart,
    load_csv,
    seq_label,
)
from vizz.presentations.ptce_2026_flex_flash import DATA_ROOT

_SHAPE_RE = re.compile(r"\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)")
TRITON_COLOR = "#5f7684"
WARM_COMPARE_COLOR = "#8a5a34"
CUDNN_COLOR = "#9a8b7c"


def _parse_seq(shape_str: str) -> int:
    m = _SHAPE_RE.search(shape_str)
    return int(m.group(3)) if m else 0


def _parse_hkv(shape_str: str) -> int:
    m = _SHAPE_RE.search(shape_str)
    return int(m.group(4)) if m else 0


def _parse_d(shape_str: str) -> int:
    m = _SHAPE_RE.search(shape_str)
    return int(m.group(6)) if m else 0


def _fa3_vs_flex_chart(scene: SlideBase) -> Group:
    t = scene.theme
    rows = load_csv(DATA_ROOT / "flex_vs_fav3_causal.csv")
    rows.sort(key=lambda r: _parse_seq(r["shape(B,Hq,M,Hkv,N,D)"]))

    labels = [seq_label(_parse_seq(r["shape(B,Hq,M,Hkv,N,D)"])) for r in rows]
    flex_fwd = [float(r["flex_attn_fwd_TFlops"]) for r in rows]
    fa3_fwd = [float(r["fav3_fwd_TFlops"]) for r in rows]
    flex_bwd = [float(r["flex_attn_bwd_TFlops"]) for r in rows]
    fa3_bwd = [float(r["fav3_bwd_TFlops"]) for r in rows]

    y_max = max(max(flex_fwd), max(fa3_fwd), max(flex_bwd), max(fa3_bwd)) * 1.15

    series_defs = [
        Series("FlexAttention", flex_fwd, TRITON_COLOR),
        Series("FA3", fa3_fwd, WARM_COMPARE_COLOR),
    ]

    fwd = grouped_bar_chart(
        t,
        series_defs,
        labels,
        y_label="TFlops",
        x_length=4.2,
        y_length=3.4,
        y_max=y_max,
        show_legend=False,
    )
    bwd = grouped_bar_chart(
        t,
        [
            Series("FlexAttention", flex_bwd, TRITON_COLOR),
            Series("FA3", fa3_bwd, WARM_COMPARE_COLOR),
        ],
        labels,
        y_label="TFlops",
        x_length=4.2,
        y_length=3.4,
        y_max=y_max,
        show_legend=False,
    )

    legend = _build_legend(t, series_defs, font_size=11)
    fwd_title = scene.meta_text(
        "Forward", font_size=13, color=t.muted_text, uppercase=False
    )
    bwd_title = scene.meta_text(
        "Backward", font_size=13, color=t.muted_text, uppercase=False
    )
    fwd_col = VGroup(fwd_title, fwd).arrange(DOWN, buff=0.12)
    bwd_col = VGroup(bwd_title, bwd).arrange(DOWN, buff=0.12)
    chart_pair = VGroup(fwd_col, bwd_col).arrange(RIGHT, buff=0.4)
    chart_content = VGroup(chart_pair, legend)
    legend.next_to(chart_pair, DOWN, buff=0.12)
    legend.set_x(chart_pair.get_center()[0])

    return _wrap_in_panel(scene, "H100: FA3 vs FlexAttention (Causal)", chart_content)


def _cudnn_vs_triton_chart(scene: SlideBase) -> Group:
    t = scene.theme
    rows = load_csv(DATA_ROOT / "gb200_triton_cudnn.csv")
    rows = [
        r
        for r in rows
        if r["attn_type"] == "causal"
        and _parse_d(r["shape(B,Hq,M,Hkv,N,D)"]) == 128
        and _parse_hkv(r["shape(B,Hq,M,Hkv,N,D)"]) == 16
    ]
    rows.sort(key=lambda r: _parse_seq(r["shape(B,Hq,M,Hkv,N,D)"]))

    labels = [seq_label(_parse_seq(r["shape(B,Hq,M,Hkv,N,D)"])) for r in rows]
    triton_fwd = [float(r["flex_attn_fwd_TFlops"]) for r in rows]
    cudnn_fwd = [float(r["cudnn_fwd_TFlops"]) for r in rows]
    triton_bwd = [float(r["flex_attn_bwd_TFlops"]) for r in rows]
    cudnn_bwd = [float(r["cudnn_bwd_TFlops"]) for r in rows]

    y_max = max(max(triton_fwd), max(cudnn_fwd), max(triton_bwd), max(cudnn_bwd)) * 1.15

    series_defs = [
        Series("Flex (Triton)", triton_fwd, TRITON_COLOR),
        Series("cuDNN", cudnn_fwd, CUDNN_COLOR),
    ]

    fwd = grouped_bar_chart(
        t,
        series_defs,
        labels,
        y_label="TFlops",
        x_length=4.2,
        y_length=3.4,
        y_max=y_max,
        show_legend=False,
    )
    bwd = grouped_bar_chart(
        t,
        [
            Series("Flex (Triton)", triton_bwd, TRITON_COLOR),
            Series("cuDNN", cudnn_bwd, CUDNN_COLOR),
        ],
        labels,
        y_label="TFlops",
        x_length=4.2,
        y_length=3.4,
        y_max=y_max,
        show_legend=False,
    )

    legend = _build_legend(t, series_defs, font_size=11)
    fwd_title = scene.meta_text(
        "Forward", font_size=13, color=t.muted_text, uppercase=False
    )
    bwd_title = scene.meta_text(
        "Backward", font_size=13, color=t.muted_text, uppercase=False
    )
    fwd_col = VGroup(fwd_title, fwd).arrange(DOWN, buff=0.12)
    bwd_col = VGroup(bwd_title, bwd).arrange(DOWN, buff=0.12)
    chart_pair = VGroup(fwd_col, bwd_col).arrange(RIGHT, buff=0.4)
    chart_content = VGroup(chart_pair, legend)
    legend.next_to(chart_pair, DOWN, buff=0.12)
    legend.set_x(chart_pair.get_center()[0])

    return _wrap_in_panel(scene, "GB200: cuDNN vs Triton (Causal)", chart_content)


def _wrap_in_panel(scene: SlideBase, title: str, chart_content: VGroup) -> Group:
    t = scene.theme
    frame = scene.panel(width=10.0, height=5.0)
    title_text = scene.meta_text(
        title,
        font_size=max(t.meta_font_size + 1, 15),
        color=t.accent_primary,
        uppercase=False,
    )
    title_text.move_to(frame.get_top() + DOWN * 0.34)
    title_text.align_to(frame.get_left() + RIGHT * 0.32, LEFT)

    divider_y = frame.get_top()[1] - 0.62
    divider = Line(
        [frame.get_left()[0] + 0.32, divider_y, 0],
        [frame.get_right()[0] - 0.32, divider_y, 0],
        color=t.divider,
        stroke_width=1.2,
    )

    chart_content.scale_to_fit_width(min(chart_content.width, frame.width - 0.7))
    content_height_limit = divider.get_y() - frame.get_bottom()[1] - 0.4
    if chart_content.height > content_height_limit:
        chart_content.scale_to_fit_height(content_height_limit)
    chart_content.move_to(
        [frame.get_center()[0], (divider.get_y() + frame.get_bottom()[1]) / 2, 0]
    )

    return Group(frame, title_text, divider, chart_content)


def build(scene: SlideBase) -> None:
    header = scene.section_header("But performance became the bottleneck")

    h100_panel = _fa3_vs_flex_chart(scene)
    h100_panel.next_to(header, DOWN, buff=0.15)

    blackwell_panel = _cudnn_vs_triton_chart(scene)
    blackwell_panel.move_to(h100_panel)

    bullet_a = scene.bullet_row("FA3 pulled ahead on\nraw throughput", font_size=22)
    bullet_b = scene.bullet_row("Triton backend couldn't\nclose the gap", font_size=22)
    bullet_c = scene.bullet_row(
        "On Blackwell, the old path\nfell even further behind", font_size=22
    )

    top_row = VGroup(bullet_a, bullet_b).arrange(RIGHT, buff=1.5)
    bottom_row = VGroup(bullet_c)
    bullets = VGroup(top_row, bottom_row).arrange(DOWN, buff=0.2)
    bullets.next_to(h100_panel, DOWN, buff=0.2)

    scene.play(FadeIn(header[0], shift=UP * 0.15), Create(header[1]))
    scene.play(FadeIn(h100_panel, shift=UP * 0.12))
    scene.next_slide()
    scene.play(FadeIn(bullet_a, shift=UP * 0.1), FadeIn(bullet_b, shift=UP * 0.1))
    scene.next_slide()
    scene.play(
        FadeIn(bullet_c, shift=UP * 0.1),
        ReplacementTransform(h100_panel, blackwell_panel),
    )
    scene.next_slide()
    scene.clear_stage()
