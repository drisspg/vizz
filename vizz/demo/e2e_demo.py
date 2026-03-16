from manim import DOWN, LEFT, RIGHT, UP, FadeIn, Square, Text, VGroup
from manim_slides import Slide


class E2EIntro(Slide):
    """Presenter Notes:
    Use this slide to verify presenter notes are visible.
    Press next to test click and keyboard navigation.
    """

    def construct(self) -> None:
        title = Text("Manimtation E2E Demo", font_size=64)
        subtitle = Text("Slide 1: Intro", font_size=32).next_to(title, DOWN)
        self.play(FadeIn(title), FadeIn(subtitle))
        self.next_slide()


class E2EFlow(Slide):
    """Presenter Notes:
    This slide validates transitions and next-slide preview.
    Confirm timer continues while navigating.
    """

    def construct(self) -> None:
        title = Text("Slide 2: Flow", font_size=52).to_edge(UP)
        left = Square(side_length=1.5, color="#2563eb").shift(LEFT * 3)
        mid = Square(side_length=1.5, color="#f59e0b")
        right = Square(side_length=1.5, color="#16a34a").shift(RIGHT * 3)
        labels = VGroup(
            Text("Author", font_size=24).next_to(left, DOWN),
            Text("Build", font_size=24).next_to(mid, DOWN),
            Text("Present", font_size=24).next_to(right, DOWN),
        )
        self.play(
            FadeIn(title), FadeIn(left), FadeIn(mid), FadeIn(right), FadeIn(labels)
        )
        self.next_slide()


class E2EOutro(Slide):
    """Presenter Notes:
    End-of-deck behavior: verify final slide and back navigation.
    """

    def construct(self) -> None:
        line1 = Text("Slide 3: Done", font_size=56)
        line2 = Text(
            "Export PDF after build to test export path", font_size=30
        ).next_to(line1, DOWN)
        self.play(FadeIn(line1), FadeIn(line2))
        self.next_slide()
