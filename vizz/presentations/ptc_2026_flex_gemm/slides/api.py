from manim import DOWN, LEFT, RIGHT, UP, FadeIn, VGroup

from vizz.presentations.components import SlideBase
from vizz.presentations.ptc_2026_flex_gemm.slides.common import code_block
from vizz.presentations.ptc_2026_flex_gemm.slides.common import header as common_header

CODE = """
from torch._higher_order_ops import flex_gemm

def epilogue(acc):                          # acc: [M, N]
    pre = acc.float() + bias                # captured load
    return F.relu(pre).to(acc.dtype), pre   # main, aux

out, pre_act = flex_gemm(
    torch.mm, (a, b),                       # GEMM + args
    epilogue,
    kernel_options={"backend": "QUACK"},
)
"""


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = common_header(scene, "The API: an epilogue is just a PyTorch function")
    code = code_block(scene, "flex_gemm(gemm, gemm_args, epilogue)", CODE, font_size=17)
    code.next_to(header, DOWN, buff=0.45).align_to(header, LEFT)
    scene.play(FadeIn(header), FadeIn(code))
    scene.wait(0.2)
    scene.next_slide(
        notes="api.code — Same idea as FlexAttention's score_mod: you get a hook over the accumulator, write eager PyTorch, and the compiler inlines it into the GEMM kernel."
    )

    def callout(title: str, body: str, color: str) -> VGroup:
        head = scene.meta_text(title, font_size=16, color=color)
        text = scene.body_text(body, font_size=19)
        return VGroup(head, text).arrange(DOWN, aligned_edge=LEFT, buff=0.1)

    callouts = VGroup(
        callout(
            "closures",
            "Captured tensors become\nbias / residual / scale loads",
            t.muted_text,
        ),
        callout(
            "tuple returns",
            "Extra outputs are stored from\nthe same tile: pre-act, masks,\nfp8 scales",
            t.muted_text,
        ),
        callout(
            "gemm families",
            "the op you already call:\nmm · bmm · scaled_mm\ngrouped_mm",
            t.muted_text,
        ),
        callout(
            "eager semantics",
            "Same result as running\nthe epilogue on mm's output",
            t.accent_secondary,
        ),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
    callouts.next_to(code, RIGHT, buff=0.4).align_to(code, UP).shift(DOWN * 0.1)
    scene.play(FadeIn(callouts, shift=LEFT * 0.1, lag_ratio=0.2), run_time=1.0)
    scene.wait(0.2)
    scene.next_slide(
        notes="api.features — Closures give global-memory loads. Tuple returns give aux stores. The GEMM itself is an ordinary PyTorch op. The reference semantics are just eager: epilogue(torch.mm(a, b))."
    )
