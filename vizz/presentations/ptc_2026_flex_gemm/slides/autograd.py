from manim import (
    LEFT,
    RIGHT,
    UP,
    AnimationGroup,
    Create,
    FadeIn,
    FadeOut,
    LaggedStart,
    Line,
    Rectangle,
    Transform,
    VGroup,
)

from vizz.presentations.components import SlideBase
from vizz.presentations.ptc_2026_flex_gemm.slides.common import code_block
from vizz.presentations.ptc_2026_flex_gemm.slides.common import header as common_header

# Adapted from FusedReluMLPFunction in
# ~/obsidian/Presentations/flex_gemm/flex_gemm_presentation_model_context.md.
# Named epilogues keep lines short, so the code can be set larger.
CODE = """
class ReluMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2):
        relu_aux = lambda acc: (acc.relu(), acc)
        h, pre = flex_gemm(torch.mm, (x, w1), relu_aux)
        ctx.save_for_backward(x, w1, w2, pre)
        return h @ w2

    @staticmethod
    def backward(ctx, dy):
        x, w1, w2, pre = ctx.saved_tensors
        relu_grad = lambda acc: acc * (pre > 0)
        dpre = flex_gemm(torch.mm, (dy, w2.T), relu_grad)
        return dpre @ w1.T, x.T @ dpre, pre.relu().T @ dy

y = ReluMLP.apply(x, w1, w2)
"""
SOURCE = CODE.strip().splitlines()


def _line(fragment: str) -> int:
    """Paragraph line index of the first source line containing `fragment`."""
    return next(i for i, text in enumerate(SOURCE) if fragment in text)


FORWARD = range(_line("def forward") - 1, _line("return h @ w2") + 1)
FORWARD = range(FORWARD.stop)
BACKWARD = range(_line("def backward") - 1, _line("return dpre") + 1)
USE = _line("ReluMLP.apply")
FUSED_FWD = (_line("relu_aux = "), _line("flex_gemm(torch.mm, (x, w1)"))
FUSED_BWD = (_line("relu_grad = "), _line("flex_gemm(torch.mm, (dy"))


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = common_header(scene, "Training: wrap it in autograd")
    code = code_block(scene, "forward-only today: you write the backward", CODE, 15)
    # Fill the slide: as large as fits beside the margin notes, centred vertically.
    code.scale_to_fit_width(10.0)
    top, bottom = header.get_bottom()[1] - 0.35, -3.75
    if code.height > top - bottom:
        code.scale_to_fit_height(top - bottom)
    code.move_to([0, (top + bottom) / 2, 0]).align_to(header, LEFT)
    label, block = code
    background, lines = block[0], block.code_lines
    for line in lines:
        line.set_opacity(0)

    def reveal(indices) -> LaggedStart:
        return LaggedStart(
            *[lines[i].animate.set_opacity(1) for i in indices],
            lag_ratio=0.12,
        )

    scene.play(FadeIn(header), FadeIn(label), FadeIn(background, shift=UP * 0.1))
    scene.play(reveal(FORWARD), run_time=1.0)
    scene.wait(0.2)
    scene.next_slide(
        notes="autograd.forward — Forward is one flex_gemm: relu fused, and the pre-activation comes out of the same kernel as an aux output."
    )

    scene.play(reveal(BACKWARD), run_time=0.9)
    scene.wait(0.2)
    scene.next_slide(
        notes="autograd.backward — Backward is another flex_gemm. relu-prime is not a prologue of the gradient GEMMs; it becomes the epilogue of the GEMM that produces d-hidden."
    )

    # Focus with a veil: paper-coloured bands over every line except the kept ones,
    # so dimming is uniform and moving focus is one smooth Transform.
    def veil(keep: set[int]) -> VGroup:
        bands = VGroup()
        for i, line in enumerate(lines):
            if i in keep or not line.submobjects:
                continue
            bands.add(
                Rectangle(
                    width=background.width - 0.12,
                    height=line.height,
                    stroke_width=0,
                    fill_color=background.get_fill_color(),
                    fill_opacity=0.78,
                ).move_to([background.get_x(), line.get_y(), 0])
            )
        return bands

    def note(line_index: tuple[int, ...], text: str, color: str) -> VGroup:
        tag = scene.meta_text(text, font_size=13, color=color)
        y = sum(lines[i].get_y() for i in line_index) / len(line_index)
        tag.next_to(block, RIGHT, buff=0.55).set_y(y)
        tick = Line(
            [block.get_right()[0] + 0.05, tag.get_y(), 0],
            tag.get_left() + LEFT * 0.1,
            color=color,
            stroke_width=1.4,
        )
        return VGroup(tick, tag)

    notes = VGroup(
        note(FUSED_FWD, "relu fused\npre-act as aux", t.accent_success),
        note(FUSED_BWD, "relu′ fused into\nthe dgrad epilogue", t.accent_success),
    )
    focus = veil({*FUSED_FWD, *FUSED_BWD, USE})
    scene.play(FadeIn(focus), run_time=0.6)
    scene.play(
        LaggedStart(
            *[
                AnimationGroup(Create(n[0]), FadeIn(n[1], shift=LEFT * 0.1))
                for n in notes
            ],
            lag_ratio=0.3,
        ),
        run_time=0.8,
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="autograd.fused — The only fused work is these two calls. Every fused op stays an epilogue; the mainloops stay untouched."
    )

    # Use it: the veil slides onto the class, and the one-liner arrives.
    usage = note(
        (USE,), "drop-in: trains like\nLinear → ReLU → Linear", t.accent_secondary
    )
    scene.play(
        Transform(focus, veil({USE})),
        FadeOut(notes, shift=RIGHT * 0.1),
        lines[USE].animate.set_opacity(1),
        run_time=0.8,
    )
    scene.play(Create(usage[0]), FadeIn(usage[1], shift=LEFT * 0.1), run_time=0.5)
    scene.wait(0.2)
    scene.next_slide(
        notes="autograd.use — Then it is a drop-in module. The other route, prototyped in TorchTitan, is a post-autograd FX rewrite that keeps the traced ATen backward. Compiled backward through flex_gemm raises instead of silently returning None grads."
    )
