from manim import DOWN, LEFT, FadeIn, VGroup

from vizz.presentations.components import SlideBase


def build(scene: SlideBase) -> None:
    mark = scene.meta_text(
        "A small idea, clearly explained", color=scene.theme.accent_primary
    )
    title = scene.title_text("From sketch\nto slide", font_size=76)
    subtitle = scene.body_text(
        "Draw the idea. Choose the story. Build the visual.", font_size=28
    )
    takeaway = scene.body_text(
        "One message per slide. One meaningful change per click.", font_size=22
    )
    content = VGroup(mark, title, subtitle, takeaway).arrange(
        DOWN, aligned_edge=LEFT, buff=0.42
    )
    content.move_to(LEFT * 0.4)
    scene.play(FadeIn(content), run_time=0.7)
    scene.wait(0.2)
    scene.next_slide(notes="Start with the idea, not the animation API.")
