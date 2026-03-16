from __future__ import annotations

from typing import TYPE_CHECKING

from manim import *

if TYPE_CHECKING:
    from vizz.presentations.components import SlideBase

TILE_A_COLOR = "#9b7558"
TILE_B_COLOR = "#6f8f7b"
HIGH_HALF_COLOR = "#617a69"
LOW_HALF_COLOR = "#8a7256"
IDLE_COLOR = "#bcb4a8"
LABEL_COLOR = "#6b746d"

UNIT = 1.75
CELL_H = 0.8
GAP = 0.05
CORNER_R = 0.1

STEP_DESCRIPTIONS = [
    "The upper M split scores first on the top tensor-core lane.",
    "One beat later the lower M split scores while the upper-half softmax runs.",
    "Then the lower-half softmax runs while the upper-half output uses the other tensor-core lane.",
    "The next upper-half score starts one beat later while the lower-half output stays staggered underneath it.",
    "The cadence repeats: 1H output lands at t7, then 2H scoring starts at t8.",
]


def _cell(color: str, label: str, op: str, width: float = UNIT) -> VGroup:
    rect = RoundedRectangle(
        corner_radius=CORNER_R,
        width=width,
        height=CELL_H,
        fill_color=color,
        fill_opacity=1,
        stroke_width=0,
    )
    label_t = Text(label, font_size=14, color=WHITE, weight=BOLD)
    op_t = Text(op, font_size=11, color="#e0e0e0")
    VGroup(label_t, op_t).arrange(DOWN, buff=0.04).move_to(rect.get_center())
    return VGroup(rect, label_t, op_t)


def _idle_cell(width: float = UNIT) -> VGroup:
    rect = RoundedRectangle(
        corner_radius=CORNER_R,
        width=width,
        height=CELL_H,
        fill_color=IDLE_COLOR,
        fill_opacity=0.28,
        stroke_width=0,
    )
    dash = Text("—", font_size=16, color="#7d847d")
    dash.move_to(rect.get_center())
    return VGroup(rect, dash)


BADGE_H_BG = "#f0ece4"
BADGE_H_FG = "#3a4a3f"
BADGE_L_BG = "#2d3830"
BADGE_L_FG = "#e8e4dc"


def _half_badge(half: str | None, anchor: Mobject) -> VGroup | None:
    if half not in ("H", "L"):
        return None
    bg = BADGE_H_BG if half == "H" else BADGE_L_BG
    fg = BADGE_H_FG if half == "H" else BADGE_L_FG
    pill = RoundedRectangle(
        corner_radius=0.06,
        width=0.22,
        height=0.17,
        fill_color=bg,
        fill_opacity=1,
        stroke_width=0,
    )
    letter = Text(half, font_size=10, color=fg, weight=BOLD)
    letter.move_to(pill.get_center())
    badge = VGroup(pill, letter)
    badge.move_to(anchor.get_corner(UL) + RIGHT * 0.14 + DOWN * 0.12)
    return badge


def _loop_cell(
    color: str,
    label: str,
    op: str,
    width: float,
    height: float,
    half: str | None = None,
) -> VGroup:
    rect = RoundedRectangle(
        corner_radius=0.12,
        width=width,
        height=height,
        fill_color=color,
        fill_opacity=1,
        stroke_width=0,
    )
    label_t = Text(label, font_size=16, color=WHITE, weight=BOLD)
    op_t = Text(op, font_size=12, color="#e0e0e0")
    VGroup(label_t, op_t).arrange(DOWN, buff=0.04).move_to(rect.get_center())
    elements = [rect, label_t, op_t]
    badge = _half_badge(half, rect)
    if badge:
        elements.append(badge)
    return VGroup(*elements)


def _formula_cell(
    color: str,
    tex: str,
    width: float = UNIT,
    height: float = CELL_H,
    half: str | None = None,
) -> VGroup:
    rect = RoundedRectangle(
        corner_radius=CORNER_R,
        width=width,
        height=height,
        fill_color=color,
        fill_opacity=1,
        stroke_width=0,
    )
    formula = MathTex(tex, color=WHITE, font_size=24)
    formula.scale_to_fit_width(width - 0.16)
    if formula.height > height - 0.14:
        formula.scale_to_fit_height(height - 0.14)
    formula.move_to(rect.get_center())
    elements = [rect, formula]
    badge = _half_badge(half, rect)
    if badge:
        elements.append(badge)
    return VGroup(*elements)


def _ghost_cell(width: float = UNIT, height: float = CELL_H) -> VGroup:
    rect = RoundedRectangle(
        corner_radius=CORNER_R,
        width=width,
        height=height,
        stroke_color="#d7cec0",
        stroke_width=1.0,
        fill_color="#ffffff",
        fill_opacity=0.0,
    )
    inset = 0.12
    diag_a = DashedLine(
        rect.get_corner(UL) + RIGHT * inset + DOWN * inset,
        rect.get_corner(DR) + LEFT * inset + UP * inset,
        color="#dfd7ca",
        stroke_width=1.0,
        dash_length=0.06,
    )
    diag_b = DashedLine(
        rect.get_corner(DL) + RIGHT * inset + UP * inset,
        rect.get_corner(UR) + LEFT * inset + DOWN * inset,
        color="#dfd7ca",
        stroke_width=1.0,
        dash_length=0.06,
    )
    return VGroup(rect, diag_a, diag_b)


def _dots(n: int, active: int, color: str) -> VGroup:
    dots = VGroup()
    for i in range(n):
        c = color if i == active else "#d0d0d0"
        dots.add(Dot(radius=0.055, color=c))
    return dots.arrange(RIGHT, buff=0.18)


def _build_intro(scene: SlideBase, header: VGroup) -> None:
    t = scene.theme

    # ── Phase 1: Full attention matrix with CTA structure ──

    n_k = 8
    n_q = 8
    q_per_cta = 2
    n_cta = n_q // q_per_cta
    cell_w = 0.42
    cell_h = 0.32
    kv_h = 0.42
    gap = 0.04
    cta_colors = ["#5b8db8", "#6f8f7b", "#c4a35a", "#9b7558"]
    score_color = "#e8b87a"
    k_color = TILE_B_COLOR
    v_color = "#c4a35a"

    def _tile(label: str, color: str, w: float, h: float) -> VGroup:
        rect = RoundedRectangle(
            corner_radius=0.04,
            width=w,
            height=h,
            fill_color=color,
            fill_opacity=1,
            stroke_width=0,
        )
        txt = MathTex(label, font_size=11, color=WHITE)
        txt.move_to(rect)
        return VGroup(rect, txt)

    score_rows: list[VGroup] = []
    for row_i in range(n_q):
        cta_i = row_i // q_per_cta
        row_cells = VGroup(
            *[
                RoundedRectangle(
                    corner_radius=0.03,
                    width=cell_w,
                    height=cell_h,
                    fill_color=score_color,
                    fill_opacity=0.25 + 0.12 * (cta_i % 2),
                    stroke_width=0.3,
                    stroke_color="#c89050",
                )
                for _ in range(n_k)
            ]
        ).arrange(RIGHT, buff=gap)
        score_rows.append(row_cells)
    score_grid = VGroup(*score_rows).arrange(DOWN, buff=gap)

    k_cells = VGroup(
        *[_tile(f"K_{{{j}}}", k_color, cell_w, kv_h) for j in range(n_k)]
    ).arrange(RIGHT, buff=gap)

    q_cells = VGroup(
        *[
            _tile(f"Q_{{{i}}}", cta_colors[i // q_per_cta], cell_w, cell_h)
            for i in range(n_q)
        ]
    ).arrange(DOWN, buff=gap)

    v_cells = VGroup(
        *[_tile(f"V_{{{i}}}", v_color, cell_w, cell_h) for i in range(n_q)]
    ).arrange(DOWN, buff=gap)

    k_cells.next_to(score_grid, UP, buff=0.5)
    k_cells.set_x(score_grid.get_x())
    q_cells.next_to(score_grid, LEFT, buff=0.1)
    q_cells.set_y(score_grid.get_y())
    v_cells.next_to(score_grid, RIGHT, buff=0.1)
    v_cells.set_y(score_grid.get_y())

    cta_tags = VGroup()
    for c in range(n_cta):
        tag = Text(f"CTA {c}", font_size=9, color=cta_colors[c], weight=BOLD)
        top_y = q_cells[c * q_per_cta].get_y()
        bot_y = q_cells[c * q_per_cta + q_per_cta - 1].get_y()
        tag.move_to([q_cells.get_left()[0] - 0.35, (top_y + bot_y) / 2, 0])
        cta_tags.add(tag)

    arrow_y = (k_cells.get_bottom()[1] + score_grid.get_top()[1]) / 2
    inner_arrow = Arrow(
        [score_grid.get_left()[0], arrow_y, 0],
        [score_grid.get_right()[0], arrow_y, 0],
        color=LABEL_COLOR,
        stroke_width=1.5,
        tip_length=0.08,
    )
    inner_text = Text("inner loop", font_size=11, color=LABEL_COLOR)
    inner_text.next_to(inner_arrow, UP, buff=0.03)

    diagram = VGroup(
        score_grid, k_cells, q_cells, v_cells, cta_tags, inner_arrow, inner_text
    )
    diagram.move_to(ORIGIN)

    cap1 = Text(
        "1 CTA per Q chunk — each scans left to right through K/V tiles (inner loop)",
        font_size=20,
        color=t.text,
        weight=BOLD,
    )
    cap1.to_edge(DOWN, buff=0.6)

    scene.play(
        FadeIn(score_grid),
        FadeIn(k_cells, shift=DOWN * 0.08),
        FadeIn(q_cells, shift=RIGHT * 0.08),
        FadeIn(v_cells, shift=LEFT * 0.08),
        FadeIn(cta_tags, shift=RIGHT * 0.05),
        FadeIn(inner_arrow),
        FadeIn(inner_text),
        FadeIn(cap1, shift=UP * 0.04),
    )
    scene.next_slide()

    row0_hl = SurroundingRectangle(
        VGroup(q_cells[0], score_rows[0]),
        color=cta_colors[0],
        stroke_width=2.5,
        buff=0.05,
        corner_radius=0.06,
    )
    scene.play(Create(row0_hl), run_time=0.3)

    scan_hl = SurroundingRectangle(
        score_rows[0][0],
        color=t.accent_primary,
        stroke_width=2,
        buff=0.03,
        corner_radius=0.04,
    )
    scene.play(Create(scan_hl), run_time=0.2)
    for j in range(1, n_k):
        next_scan = SurroundingRectangle(
            score_rows[0][j],
            color=t.accent_primary,
            stroke_width=2,
            buff=0.03,
            corner_radius=0.04,
        )
        scene.play(Transform(scan_hl, next_scan), run_time=0.15)
    scene.play(FadeOut(scan_hl), run_time=0.2)
    scene.next_slide()

    # ── Phase 2: B200 timing mismatch ──

    pipe_h = 0.85

    def _pbox(label: str, sub: str, color: str, w: float) -> VGroup:
        rect = RoundedRectangle(
            corner_radius=0.12,
            width=w,
            height=pipe_h,
            fill_color=color,
            fill_opacity=1,
            stroke_width=0,
        )
        lbl = Text(label, font_size=16, color=WHITE, weight=BOLD)
        slbl = Text(sub, font_size=12, color="#e0e0e0")
        VGroup(lbl, slbl).arrange(DOWN, buff=0.04).move_to(rect)
        return VGroup(rect, lbl, slbl)

    t_mma_w = 1.3
    t_sm_w = 2.6

    ts = _pbox("Score", "512 cycles", TILE_B_COLOR, t_mma_w)
    tsm = _pbox("Softmax", "1024 cycles", TILE_A_COLOR, t_sm_w)
    to_ = _pbox("Output", "512 cycles", TILE_B_COLOR, t_mma_w)
    tpipe = VGroup(ts, tsm, to_).arrange(RIGHT, buff=0.05)
    tpipe.move_to(ORIGIN + UP * 0.2)

    idle_brace = Brace(tsm, UP, color="#b91c1c", buff=0.1)
    idle_label = Text("Tensor cores idle", font_size=16, color="#b91c1c", weight=BOLD)
    idle_label.next_to(idle_brace, UP, buff=0.06)

    cap2 = Text(
        "B200: softmax is 2× slower than MMA → tensor cores idle half the time",
        font_size=20,
        color="#8a7256",
        weight=BOLD,
    )
    cap2.to_edge(DOWN, buff=0.6)

    scene.play(
        FadeOut(diagram),
        FadeOut(row0_hl),
        FadeIn(tpipe, shift=UP * 0.1),
        ReplacementTransform(cap1, cap2),
    )
    scene.play(GrowFromCenter(idle_brace), FadeIn(idle_label, shift=DOWN * 0.04))
    scene.next_slide()

    ol_h = 0.7
    ol_mma = 1.2
    ol_sm = 2.4
    ol_gap = 0.05

    def _ol_cell(label: str, color: str, w: float, opa: float = 1.0) -> VGroup:
        rect = RoundedRectangle(
            corner_radius=0.1,
            width=w,
            height=ol_h,
            fill_color=color,
            fill_opacity=opa,
            stroke_width=0,
        )
        txt = Text(label, font_size=13, color=WHITE, weight=BOLD)
        txt.move_to(rect)
        return VGroup(rect, txt)

    h_s = _ol_cell("Score", HIGH_HALF_COLOR, ol_mma)
    h_sm = _ol_cell("Softmax", HIGH_HALF_COLOR, ol_sm, 0.55)
    h_o = _ol_cell("Output", HIGH_HALF_COLOR, ol_mma)
    h_row = VGroup(h_s, h_sm, h_o).arrange(RIGHT, buff=ol_gap)

    l_s = _ol_cell("Score", LOW_HALF_COLOR, ol_mma)
    l_sm = _ol_cell("Softmax", LOW_HALF_COLOR, ol_sm, 0.55)
    l_o = _ol_cell("Output", LOW_HALF_COLOR, ol_mma)
    l_row = VGroup(l_s, l_sm, l_o).arrange(RIGHT, buff=ol_gap)

    overlap = VGroup(h_row, l_row).arrange(DOWN, buff=0.1)
    l_row.shift(RIGHT * (ol_mma + ol_gap))

    h_tag = Text("High half", font_size=13, color=BADGE_H_FG, weight=BOLD)
    l_tag = Text("Low half", font_size=13, color=BADGE_L_BG, weight=BOLD)
    h_tag.next_to(h_row, LEFT, buff=0.25)
    l_tag.next_to(l_row, LEFT, buff=0.25)

    overlap_all = VGroup(overlap, h_tag, l_tag)
    overlap_all.move_to(tpipe.get_center())

    cap3 = Text(
        "Split Q in half → overlap softmax with the next MMA, filling the bubble",
        font_size=20,
        color=t.text,
        weight=BOLD,
    )
    cap3.move_to(cap2)

    scene.play(
        FadeOut(idle_brace),
        FadeOut(idle_label),
        ReplacementTransform(cap2, cap3),
        ReplacementTransform(ts, h_s),
        ReplacementTransform(tsm, h_sm),
        ReplacementTransform(to_, h_o),
        FadeIn(l_s, shift=RIGHT * 0.1),
        FadeIn(l_sm, shift=RIGHT * 0.1),
        FadeIn(l_o, shift=RIGHT * 0.1),
        FadeIn(h_tag, shift=LEFT * 0.05),
        FadeIn(l_tag, shift=LEFT * 0.05),
    )
    scene.next_slide()

    scene.play(
        FadeOut(h_row),
        FadeOut(l_row),
        FadeOut(h_tag),
        FadeOut(l_tag),
        FadeOut(cap3),
    )


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = scene.section_header("Why Blackwell required a redesign")

    scene.play(FadeIn(header[0], shift=UP * 0.15), Create(header[1]))
    scene.next_slide()
    _build_intro(scene, header)

    slot_w = 1.02
    slot_gap = 0.06
    slot_step = slot_w + slot_gap
    mufu_w = 2 * slot_w + slot_gap
    lane_gap = 0.14
    top_y = 0.72
    mid_y = top_y - (CELL_H + lane_gap)
    bot_y = mid_y - (CELL_H + lane_gap)

    def slot_x(slot: float) -> float:
        return (slot - 2.0) * slot_step

    top_lane = VGroup(
        Text("Tensor Cores", font_size=18, color=LABEL_COLOR, weight=BOLD),
        Text("high half", font_size=11, color=BADGE_H_FG, weight=BOLD),
    ).arrange(DOWN, buff=0.03)
    mid_lane = VGroup(
        Text("MUFU.EX2", font_size=18, color=LABEL_COLOR, weight=BOLD),
        Text("softmax", font_size=11, color="#b0b0b0"),
    ).arrange(DOWN, buff=0.03)
    bot_lane = VGroup(
        Text("Tensor Cores", font_size=18, color=LABEL_COLOR, weight=BOLD),
        Text("low half", font_size=11, color=BADGE_L_BG, weight=BOLD),
    ).arrange(DOWN, buff=0.03)

    ghost_top = VGroup(*[_ghost_cell(slot_w) for _ in range(9)])
    ghost_mid = VGroup(*[_ghost_cell(slot_w) for _ in range(9)])
    ghost_bot = VGroup(*[_ghost_cell(slot_w) for _ in range(9)])
    for i, ghost in enumerate(ghost_top):
        ghost.move_to([slot_x(i), top_y, 0])
    for i, ghost in enumerate(ghost_mid):
        ghost.move_to([slot_x(i), mid_y, 0])
    for i, ghost in enumerate(ghost_bot):
        ghost.move_to([slot_x(i), bot_y, 0])

    top_lane.next_to(ghost_top[0], LEFT, buff=0.38)
    mid_lane.next_to(ghost_mid[0], LEFT, buff=0.38)
    bot_lane.next_to(ghost_bot[0], LEFT, buff=0.38)

    t_labels = VGroup(
        *[Text(f"t{i}", font_size=13, color=LABEL_COLOR, weight=BOLD) for i in range(9)]
    )
    for i, tl in enumerate(t_labels):
        tl.move_to([slot_x(i), top_y + 0.56, 0])

    top_s0h = _formula_cell(TILE_A_COLOR, r"S_0^{H}=Q^{H}K_0", width=slot_w, half="H")
    top_s0h.move_to([slot_x(0), top_y, 0])
    bot_s0l = _formula_cell(TILE_A_COLOR, r"S_0^{L}=Q^{L}K_0", width=slot_w, half="L")
    bot_s0l.move_to([slot_x(1), bot_y, 0])
    mid_p0h = _formula_cell(
        TILE_A_COLOR,
        r"P_0^{H}=\operatorname{softmax}(S_0^{H})",
        width=mufu_w,
        half="H",
    )
    mid_p0h.move_to([slot_x(1.5), mid_y, 0])
    mid_p0l = _formula_cell(
        TILE_A_COLOR,
        r"P_0^{L}=\operatorname{softmax}(S_0^{L})",
        width=mufu_w,
        half="L",
    )
    mid_p0l.move_to([slot_x(3.5), mid_y, 0])
    top_o0h = _formula_cell(TILE_A_COLOR, r"O_0^{H}=P_0^{H}V_0", width=slot_w, half="H")
    top_o0h.move_to([slot_x(3), top_y, 0])
    top_s1h = _formula_cell(TILE_B_COLOR, r"S_1^{H}=Q^{H}K_1", width=slot_w, half="H")
    top_s1h.move_to([slot_x(4), top_y, 0])
    mid_p1h = _formula_cell(
        TILE_B_COLOR,
        r"P_1^{H}=\operatorname{softmax}(S_1^{H})",
        width=mufu_w,
        half="H",
    )
    mid_p1h.move_to([slot_x(5.5), mid_y, 0])
    bot_o0l = _formula_cell(TILE_A_COLOR, r"O_0^{L}=P_0^{L}V_0", width=slot_w, half="L")
    bot_o0l.move_to([slot_x(5), bot_y, 0])
    mid_p1l = _formula_cell(
        TILE_B_COLOR,
        r"P_1^{L}=\operatorname{softmax}(S_1^{L})",
        width=mufu_w,
        half="L",
    )
    mid_p1l.move_to([slot_x(7.5), mid_y, 0])
    bot_s1l = _formula_cell(TILE_B_COLOR, r"S_1^{L}=Q^{L}K_1", width=slot_w, half="L")
    bot_s1l.move_to([slot_x(6), bot_y, 0])
    top_o1h = _formula_cell(TILE_B_COLOR, r"O_1^{H}=P_1^{H}V_1", width=slot_w, half="H")
    top_o1h.move_to([slot_x(7), top_y, 0])
    top_s2h = _formula_cell(TILE_A_COLOR, r"S_2^{H}=Q^{H}K_2", width=slot_w, half="H")
    top_s2h.move_to([slot_x(8), top_y, 0])

    legend_a = VGroup(
        RoundedRectangle(
            corner_radius=0.05,
            width=0.28,
            height=0.18,
            fill_color=TILE_A_COLOR,
            fill_opacity=1,
            stroke_width=0,
        ),
        Text("Even steps", font_size=12, color=t.text),
    ).arrange(RIGHT, buff=0.08)
    legend_b = VGroup(
        RoundedRectangle(
            corner_radius=0.05,
            width=0.28,
            height=0.18,
            fill_color=TILE_B_COLOR,
            fill_opacity=1,
            stroke_width=0,
        ),
        Text("Odd steps", font_size=12, color=t.text),
    ).arrange(RIGHT, buff=0.08)
    legend_h_pill = RoundedRectangle(
        corner_radius=0.06,
        width=0.28,
        height=0.18,
        fill_color=BADGE_H_BG,
        fill_opacity=1,
        stroke_width=0,
    )
    legend_h_letter = Text("H", font_size=11, color=BADGE_H_FG, weight=BOLD)
    legend_h_letter.move_to(legend_h_pill.get_center())
    legend_h = VGroup(
        VGroup(legend_h_pill, legend_h_letter),
        Text("High half", font_size=12, color=t.text),
    ).arrange(RIGHT, buff=0.08)
    legend_l_pill = RoundedRectangle(
        corner_radius=0.06,
        width=0.28,
        height=0.18,
        fill_color=BADGE_L_BG,
        fill_opacity=1,
        stroke_width=0,
    )
    legend_l_letter = Text("L", font_size=11, color=BADGE_L_FG, weight=BOLD)
    legend_l_letter.move_to(legend_l_pill.get_center())
    legend_l = VGroup(
        VGroup(legend_l_pill, legend_l_letter),
        Text("Low half", font_size=12, color=t.text),
    ).arrange(RIGHT, buff=0.08)
    legend = VGroup(
        VGroup(legend_a, legend_b).arrange(RIGHT, buff=0.4),
        VGroup(legend_h, legend_l).arrange(RIGHT, buff=0.4),
    ).arrange(DOWN, aligned_edge=RIGHT, buff=0.08)
    legend.next_to(header, DOWN, buff=0.08).to_edge(RIGHT, buff=0.45)

    timeline_left = ghost_top[0].get_left()[0]
    timeline_right = ghost_top[-1].get_right()[0]
    cl = Line(
        [timeline_left, bot_y - 0.62, 0],
        [timeline_right, bot_y - 0.62, 0],
        color=LABEL_COLOR,
        stroke_width=1.2,
    )
    cl_tip = Triangle(fill_color=LABEL_COLOR, fill_opacity=1, stroke_width=0)
    cl_tip.scale(0.05).rotate(-PI / 2).next_to(cl, RIGHT, buff=0.01)

    cl_ticks = VGroup()
    cl_nums = VGroup()
    for i in range(10):
        x = timeline_left + i * slot_step
        if i == 9:
            x = timeline_right
        tick = Line(UP * 0.04, DOWN * 0.04, color=LABEL_COLOR, stroke_width=1.2)
        tick.move_to([x, cl.get_center()[1], 0])
        cl_ticks.add(tick)
    for i in range(9):
        mid_x = (cl_ticks[i].get_x() + cl_ticks[i + 1].get_x()) / 2
        num = Text("512", font_size=10, color=LABEL_COLOR)
        num.move_to([mid_x, cl.get_center()[1] - 0.16, 0])
        cl_nums.add(num)
    clock = VGroup(cl, cl_tip, cl_ticks, cl_nums)
    overview_group = VGroup(
        ghost_top,
        ghost_mid,
        ghost_bot,
        top_lane,
        mid_lane,
        bot_lane,
        t_labels,
        clock,
        top_s0h,
        bot_s0l,
        mid_p0h,
        mid_p0l,
        top_o0h,
        top_s1h,
        mid_p1h,
        bot_o0l,
        mid_p1l,
        bot_s1l,
        top_o1h,
        top_s2h,
    )
    overview_group.shift(LEFT * overview_group.get_center()[0])

    status_y = clock.get_bottom() + DOWN * 0.25

    scene.play(
        FadeIn(legend),
        FadeIn(top_lane, shift=LEFT * 0.08),
        FadeIn(mid_lane, shift=LEFT * 0.08),
        FadeIn(bot_lane, shift=LEFT * 0.08),
        FadeIn(t_labels),
        FadeIn(ghost_top),
        FadeIn(ghost_mid),
        FadeIn(ghost_bot),
        FadeIn(clock),
    )
    scene.next_slide()

    step_cells = [
        [top_s0h],
        [mid_p0h, bot_s0l],
        [top_o0h, mid_p0l],
        [top_s1h, bot_o0l],
        [mid_p1h, bot_s1l, mid_p1l, top_o1h, top_s2h],
    ]

    prev_dots_g = None
    prev_desc_g = None
    prev_hl = None

    for i in range(5):
        dots_g = _dots(5, i, t.accent_primary)
        dots_g.move_to(status_y)
        desc_g = Text(STEP_DESCRIPTIONS[i], font_size=15, color=t.text)
        desc_g.next_to(dots_g, DOWN, buff=0.1)

        hl = SurroundingRectangle(
            VGroup(*step_cells[i]),
            color=t.accent_primary,
            stroke_width=2.5,
            buff=0.05,
            corner_radius=0.12,
        )

        anims = [
            *[FadeIn(c, shift=RIGHT * 0.1) for c in step_cells[i]],
            FadeIn(dots_g),
            FadeIn(desc_g, shift=UP * 0.05),
            Create(hl),
        ]
        if prev_dots_g:
            anims += [FadeOut(prev_dots_g), FadeOut(prev_desc_g)]
        if prev_hl:
            anims.append(FadeOut(prev_hl))

        scene.play(*anims, run_time=0.65)
        scene.next_slide()
        prev_dots_g, prev_desc_g, prev_hl = dots_g, desc_g, hl

    scene.play(FadeOut(prev_dots_g), FadeOut(prev_desc_g), FadeOut(prev_hl))

    steady_state_window = SurroundingRectangle(
        VGroup(
            *ghost_top[3:9],
            *ghost_mid[3:9],
            *ghost_bot[3:9],
            top_o0h,
            top_s1h,
            mid_p0l,
            mid_p1h,
            bot_o0l,
            bot_s1l,
            top_o1h,
            top_s2h,
            mid_p1l,
        ),
        color=t.accent_primary,
        stroke_width=2.5,
        buff=0.08,
        corner_radius=0.16,
    )
    steady_state_msg = Text(
        "From t3 to t8 the kernel reaches steady state and repeats the same ping-pong cadence.",
        font_size=15,
        color=t.accent_primary,
        weight=BOLD,
    )
    steady_state_msg.move_to(status_y)
    scene.play(
        *[FadeOut(m, shift=LEFT * 0.04) for m in [top_s0h, bot_s0l, mid_p0h]],
        *[
            FadeOut(m, shift=LEFT * 0.04)
            for m in [
                *ghost_top[:3],
                *ghost_mid[:3],
                *ghost_bot[:3],
                *t_labels[:3],
            ]
        ],
        run_time=0.5,
    )
    scene.play(
        FadeIn(steady_state_window),
        FadeIn(steady_state_msg, shift=UP * 0.04),
        run_time=0.4,
    )
    scene.next_slide()
    scene.clear_stage()
