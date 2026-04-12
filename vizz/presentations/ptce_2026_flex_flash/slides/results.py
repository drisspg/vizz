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
_ATTN_TYPES = ["noop", "causal", "alibi", "document_mask"]
TRITON_COLOR = "#5f7684"
FLASH_COLOR = "#8a5a34"


def _parse(shape_str: str) -> tuple[int, int, int]:
    m = _SHAPE_RE.search(shape_str)
    if not m:
        return (0, 0, 0)
    return int(m.group(3)), int(m.group(4)), int(m.group(6))


def _merge_flash_triton(
    flash_path: str, triton_cudnn_path: str, *, d: int = 128, hkv: int = 16
) -> dict[str, list[dict]]:
    flash_rows = load_csv(DATA_ROOT / flash_path)
    triton_rows = load_csv(DATA_ROOT / triton_cudnn_path)

    flash_by_key: dict[tuple, dict] = {}
    for r in flash_rows:
        seq, h, dd = _parse(r["shape(B,Hq,M,Hkv,N,D)"])
        if dd == d and h == hkv:
            flash_by_key[(r["attn_type"], r["shape(B,Hq,M,Hkv,N,D)"])] = r

    by_attn: dict[str, list[dict]] = {}
    for r in triton_rows:
        seq, h, dd = _parse(r["shape(B,Hq,M,Hkv,N,D)"])
        if dd != d or h != hkv or r["attn_type"] not in _ATTN_TYPES:
            continue
        key = (r["attn_type"], r["shape(B,Hq,M,Hkv,N,D)"])
        flash_r = flash_by_key.get(key)
        if not flash_r:
            continue
        merged = {
            "attn_type": r["attn_type"],
            "seq_len": seq,
            "triton_fwd": float(r["flex_attn_fwd_TFlops"]),
            "triton_bwd": float(r["flex_attn_bwd_TFlops"]),
            "flash_fwd": float(flash_r["flex_attn_fwd_TFlops"]),
            "flash_bwd": float(flash_r["flex_attn_bwd_TFlops"]),
        }
        by_attn.setdefault(r["attn_type"], []).append(merged)

    for rows in by_attn.values():
        rows.sort(key=lambda r: r["seq_len"])
    return by_attn


def _build_chart_stage(
    scene: SlideBase, title: str, flash_csv: str, triton_csv: str
) -> Group:
    t = scene.theme
    data = _merge_flash_triton(flash_csv, triton_csv)

    all_vals = [
        v
        for rows in data.values()
        for r in rows
        for v in (r["triton_fwd"], r["flash_fwd"], r["triton_bwd"], r["flash_bwd"])
    ]
    y_max = max(all_vals) * 1.15

    grid_rows = []
    for attn_type in _ATTN_TYPES:
        rows = data.get(attn_type, [])
        if not rows:
            continue
        labels = [seq_label(r["seq_len"]) for r in rows]
        attn_label = attn_type.replace("_", " ").title()

        fwd = grouped_bar_chart(
            t,
            [
                Series("Flex (Triton)", [r["triton_fwd"] for r in rows], TRITON_COLOR),
                Series("Flex (Flash)", [r["flash_fwd"] for r in rows], FLASH_COLOR),
            ],
            labels,
            y_label="TFlops",
            x_length=4.2,
            y_length=1.3,
            y_max=y_max,
            show_legend=False,
        )
        bwd = grouped_bar_chart(
            t,
            [
                Series("Flex (Triton)", [r["triton_bwd"] for r in rows], TRITON_COLOR),
                Series("Flex (Flash)", [r["flash_bwd"] for r in rows], FLASH_COLOR),
            ],
            labels,
            y_label="TFlops",
            x_length=4.2,
            y_length=1.3,
            y_max=y_max,
            show_legend=False,
        )

        fwd_title = scene.meta_text(
            f"{attn_label} — Forward", font_size=11, color=t.muted_text, uppercase=False
        )
        bwd_title = scene.meta_text(
            f"{attn_label} — Backward",
            font_size=11,
            color=t.muted_text,
            uppercase=False,
        )
        fwd_col = VGroup(fwd_title, fwd).arrange(DOWN, buff=0.08)
        bwd_col = VGroup(bwd_title, bwd).arrange(DOWN, buff=0.08)
        row = VGroup(fwd_col, bwd_col).arrange(RIGHT, buff=0.3)
        grid_rows.append(row)

    series_defs = [
        Series("Flex (Triton)", [], TRITON_COLOR),
        Series("Flex (Flash)", [], FLASH_COLOR),
    ]
    legend = _build_legend(t, series_defs, font_size=13)

    chart_grid = VGroup(*grid_rows).arrange(DOWN, buff=0.15)

    frame = scene.panel(width=12.2, height=6.1)
    label = scene.meta_text(
        title, font_size=18, color=t.accent_primary, uppercase=False
    )
    label.align_to(frame, LEFT).next_to(frame, UP, buff=0.12)
    legend.next_to(frame, UP, buff=0.12).align_to(frame, RIGHT)

    footnote = scene.meta_text(
        "* Detailed benchmark methodology and reproduction setup are in the blog post.",
        font_size=10,
        color=t.muted_text,
        uppercase=False,
    )
    footnote.next_to(frame, DOWN, buff=0.14)
    footnote.align_to(frame.get_right() + LEFT * 0.02, RIGHT)

    chart_grid.scale_to_fit_width(min(chart_grid.width, frame.width - 0.6))
    if chart_grid.height > frame.height - 0.3:
        chart_grid.scale_to_fit_height(frame.height - 0.3)
    chart_grid.move_to(frame.get_center())

    return Group(label, legend, frame, chart_grid, footnote)


def build(scene: SlideBase) -> None:
    header = scene.section_header("Results: strong wins now on Blackwell and Hopper")

    hopper_stage = _build_chart_stage(
        scene, "Hopper", "aws_h200_flash.csv", "aws_h200_triton_cudnn.csv"
    )
    blackwell_stage = _build_chart_stage(
        scene, "Blackwell", "gb200_flash.csv", "gb200_triton_cudnn.csv"
    )
    for stage in (hopper_stage, blackwell_stage):
        stage.next_to(header, DOWN, buff=0.26)

    scene.play(FadeIn(header[0], shift=UP * 0.15), Create(header[1]))
    scene.play(
        FadeIn(hopper_stage[0], shift=UP * 0.1),
        FadeIn(hopper_stage[1], shift=UP * 0.1),
        FadeIn(hopper_stage[2], shift=UP * 0.1),
        FadeIn(hopper_stage[3], shift=UP * 0.08),
        FadeIn(hopper_stage[4], shift=UP * 0.04),
    )
    scene.next_slide()
    scene.play(
        Transform(hopper_stage[0], blackwell_stage[0]),
        Transform(hopper_stage[1], blackwell_stage[1]),
        Transform(hopper_stage[2], blackwell_stage[2]),
        Transform(hopper_stage[3], blackwell_stage[3]),
        Transform(hopper_stage[4], blackwell_stage[4]),
        run_time=0.9,
    )
    scene.next_slide()
    scene.clear_stage()
