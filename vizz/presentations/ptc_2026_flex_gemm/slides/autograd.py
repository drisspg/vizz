from manim import DOWN, LEFT, RIGHT, Create, FadeIn, SurroundingRectangle, VGroup

from vizz.presentations.components import SlideBase
from vizz.presentations.ptc_2026_flex_gemm.slides.common import code_block
from vizz.presentations.ptc_2026_flex_gemm.slides.common import header as common_header

# Adapted from FusedReluMLPFunction in
# ~/obsidian/Presentations/flex_gemm/flex_gemm_presentation_model_context.md.
CODE = """
class ReluMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2):
        h, pre = flex_gemm(torch.mm, (x, w1), lambda acc: (acc.relu(), acc))
        ctx.save_for_backward(x, w1, w2, pre)
        return h @ w2

    @staticmethod
    def backward(ctx, dy):
        x, w1, w2, pre = ctx.saved_tensors
        dpre = flex_gemm(torch.mm, (dy, w2.T), lambda acc: acc * (pre > 0))
        return dpre @ w1.T, x.T @ dpre, pre.relu().T @ dy

y = ReluMLP.apply(x, w1, w2)    # trains like Linear → ReLU → Linear
"""
# Paragraph line indices include blank lines.
FORWARD_LINE, BACKWARD_LINE, USE_LINE = 3, 10, 13


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = common_header(scene, "Training: wrap the fused pieces in autograd")
    code = code_block(scene, "forward-only today, so you own the backward", CODE, 15)
    code.scale_to_fit_width(9.6)
    code.next_to(header, DOWN, buff=0.9).align_to(header, LEFT)
    lines = code[1].code_lines

    scene.play(FadeIn(header), FadeIn(code))
    scene.wait(0.2)
    scene.next_slide(
        notes="autograd.code — flex_gemm is forward-only, so training means an autograd.Function. The pieces are just flex_gemm calls."
    )

    def callout(line: int, text: str) -> VGroup:
        frame = SurroundingRectangle(
            lines[line],
            color=t.accent_success,
            buff=0.03,
            corner_radius=0.03,
            stroke_width=2,
        )
        label = scene.meta_text(text, font_size=13, color=t.accent_success)
        label.next_to(code[1], RIGHT, buff=0.35).set_y(frame.get_y())
        return VGroup(frame, label)

    fwd = callout(FORWARD_LINE, "relu fused\npre-act saved as aux")
    bwd = callout(BACKWARD_LINE, "relu′ fused into\nthe dgrad epilogue")
    scene.play(
        Create(fwd[0]), FadeIn(fwd[1]), Create(bwd[0]), FadeIn(bwd[1]), run_time=0.7
    )
    scene.wait(0.2)
    scene.next_slide(
        notes="autograd.fused — Forward: relu fused into mm1, and the pre-activation comes out of the same kernel as an aux output. Backward: relu-prime is not a prologue of the gradient GEMMs; it becomes the epilogue of the GEMM that produces d-hidden. Every fused op stays an epilogue."
    )

    use = SurroundingRectangle(
        lines[USE_LINE],
        color=t.accent_secondary,
        buff=0.03,
        corner_radius=0.03,
        stroke_width=2,
    )
    scene.play(Create(use), run_time=0.5)
    scene.wait(0.2)
    scene.next_slide(
        notes="autograd.use — Then it is a drop-in module. In TorchTitan we also prototyped the other route: a post-autograd FX rewrite that keeps the traced ATen backward. Compiled backward through flex_gemm raises instead of silently returning None grads."
    )
