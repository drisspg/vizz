from pathlib import Path

from manim import DOWN, LEFT, RIGHT, UP, FadeIn, ImageMobject, VGroup

from vizz.presentations.components import SlideBase
from vizz.presentations.ptc_2026_flex_gemm.slides.common import code_block
from vizz.presentations.ptc_2026_flex_gemm.slides.common import header as common_header

PLATE = Path(__file__).parents[1] / "assets" / "philox_round.png"

# Condensed from agent_space/sr_epilogue/fused_philox.py (pytorch checkout on the
# devgpu). Correctness demo on SM100a/SM103a; no latency claim.
CODE = """
def epilogue(acc):
    # Philox4x32-10 on (row, col, invocation)
    # 10 x (mul.hi/lo, xor) in inline PTX
    bits = philox(rows, cols, seed, invocation)
    return inline_asm_elementwise(
        acc.float(), bits,
        asm_str="cvt.rs.satfinite.bf16x2.f32 ...",
        constraints="=h,f,r", dtype=torch.bfloat16,
    )

out = flex_gemm(torch.mm, (a, b), epilogue,
                kernel_options={"backend": "QUACK"})
"""


def build(scene: SlideBase) -> None:
    t = scene.theme
    header = common_header(scene, "Things you might not expect: an RNG in the epilogue")
    code = code_block(
        scene, "gemm + philox + stochastic bf16 rounding", CODE, font_size=14
    )
    code.next_to(header, DOWN, buff=0.4).align_to(header, LEFT)
    facts = VGroup(
        scene.meta_text(
            "one kernel · user-land code · no new op", color=t.accent_success
        ),
        scene.meta_text(
            "own counter stream, not torch.Generator · sm100a/sm103a", font_size=13
        ),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
    facts.next_to(code, DOWN, buff=0.3).align_to(code, LEFT)

    scene.play(FadeIn(header), FadeIn(code), FadeIn(facts))
    scene.wait(0.2)
    scene.next_slide(
        notes="philox.code — Stochastic rounding needs random bits per output element. The epilogue computes a counter-based Philox RNG from the output coordinate and an invocation counter, then uses the Blackwell cvt.rs instruction to round the fp32 accumulator to bf16. All of it is PyTorch plus inline_asm_elementwise; nothing in FlexGEMM knows about RNGs."
    )

    plate = ImageMobject(str(PLATE)).scale_to_fit_width(6.3)
    plate.next_to(code, RIGHT, buff=0.3).align_to(code, UP)
    if plate.get_right()[0] > 6.9:
        plate.scale_to_fit_width(6.9 - plate.get_left()[0])
    scene.play(FadeIn(plate, shift=LEFT * 0.1), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(
        notes="philox.round — One Philox round: two wide multiplies, a cross-wired xor with the key, repeated ten times. Word zero feeds the rounding instruction. This is a correctness demo, not a benchmark, and it does not consume PyTorch's generator stream."
    )
