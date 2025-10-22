"""Scale calculation formulas for different quantization formats.

Shows how amax is converted into scales for FP8, MXFP4, and MXFP8 formats,
starting with abstract formulas then filling in the actual max values.

To run:
manim-slides render vizz/quant/amax_to_scale_animation.py AmaxToScaleAnimation -ql
manim-slides present AmaxToScaleAnimation
"""

from manim import *
from manim_slides import Slide

from vizz.quant.mx_base import COLORS


class AmaxToScaleAnimation(Slide):
    """Visualize scale calculation formulas for different formats."""

    def construct(self):
        title = Text(
            "Scale Calculation Formulas",
            font_size=40,
            color=COLORS["text"],
            weight=BOLD,
        )
        title.to_edge(UP, buff=0.4)
        self.add(title)

        abstract_fp8 = MathTex(
            r"\text{scale}_{\text{fp8}} = \frac{\text{FP8\_MAX}}{\text{amax}}",
            font_size=38,
        )
        abstract_fp8.set_color_by_tex("FP8", COLORS["fp8_color"])
        abstract_fp8.set_color_by_tex("amax", COLORS["max_value"])

        abstract_mxfp4 = MathTex(
            r"\text{scale}_{\text{mxfp4}} = 2^{\left\lfloor \log_2\left(\frac{\text{FP4\_MAX}}{\text{amax}}\right) \right\rfloor}",
            font_size=38,
        )
        abstract_mxfp4.set_color_by_tex("FP4", COLORS["nvfp_color"])
        abstract_mxfp4.set_color_by_tex("amax", COLORS["max_value"])

        abstract_mxfp8 = MathTex(
            r"\text{scale}_{\text{mxfp8}} = 2^{\left\lfloor \log_2\left(\frac{\text{FP8\_MAX}}{\text{amax}}\right) \right\rfloor}",
            font_size=38,
        )
        abstract_mxfp8.set_color_by_tex("FP8", COLORS["fp8_color"])
        abstract_mxfp8.set_color_by_tex("amax", COLORS["max_value"])

        equations = VGroup(abstract_fp8, abstract_mxfp4, abstract_mxfp8)
        equations.arrange(DOWN, buff=0.8)
        equations.move_to(ORIGIN)

        self.play(FadeIn(equations), run_time=0.8)
        self.next_slide()

        concrete_fp8 = MathTex(
            r"\text{scale}_{\text{fp8}} = \frac{448.0}{\text{amax}}",
            font_size=38,
        )
        concrete_fp8.set_color_by_tex("448", COLORS["fp8_color"])
        concrete_fp8.set_color_by_tex("amax", COLORS["max_value"])
        concrete_fp8.move_to(abstract_fp8.get_center())

        concrete_mxfp4 = MathTex(
            r"\text{scale}_{\text{mxfp4}} = 2^{\left\lfloor \log_2\left(\frac{6.0}{\text{amax}}\right) \right\rfloor}",
            font_size=38,
        )
        concrete_mxfp4.set_color_by_tex("6.0", COLORS["nvfp_color"])
        concrete_mxfp4.set_color_by_tex("amax", COLORS["max_value"])
        concrete_mxfp4.move_to(abstract_mxfp4.get_center())

        concrete_mxfp8 = MathTex(
            r"\text{scale}_{\text{mxfp8}} = 2^{\left\lfloor \log_2\left(\frac{448.0}{\text{amax}}\right) \right\rfloor}",
            font_size=38,
        )
        concrete_mxfp8.set_color_by_tex("448", COLORS["fp8_color"])
        concrete_mxfp8.set_color_by_tex("amax", COLORS["max_value"])
        concrete_mxfp8.move_to(abstract_mxfp8.get_center())

        self.play(
            Transform(abstract_fp8, concrete_fp8),
            Transform(abstract_mxfp4, concrete_mxfp4),
            Transform(abstract_mxfp8, concrete_mxfp8),
            run_time=1.5,
        )
        self.next_slide()

        intuition_title = Text(
            "What do these scales mean?",
            font_size=36,
            color=COLORS["text"],
            weight=BOLD,
        )
        intuition_title.to_edge(UP, buff=0.4)

        scale_greater = Text(
            "scale > 1  →  Stretch values to fill the type's dynamic range",
            font_size=28,
            color=COLORS["computed_scale"],
        )
        scale_greater.move_to(ORIGIN + UP * 0.5)

        scale_less = Text(
            "scale < 1  →  Compress values within the type's range",
            font_size=28,
            color=COLORS["tensor_data"],
        )
        scale_less.next_to(scale_greater, DOWN, buff=0.4)

        self.play(
            FadeOut(title),
            FadeOut(equations),
            FadeIn(intuition_title),
            FadeIn(scale_greater),
            FadeIn(scale_less),
            run_time=1.0,
        )
        self.next_slide()
