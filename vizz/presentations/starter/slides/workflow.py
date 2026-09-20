from itertools import pairwise

from manim import DOWN, RIGHT, Arrow, Create, FadeIn, Group

from vizz.presentations.components import SlideBase


def build(scene: SlideBase) -> None:
    header = scene.section_header("Make the explanation visible")
    cards = Group(
        *[
            scene.labeled_panel(
                label,
                width=3.6,
                height=2.4,
                content=scene.body_text(body, font_size=25),
            )
            for label, body in [
                ("01 / Sketch", "Boxes, arrows,\nlabels, intent"),
                ("02 / Story", "Meaning, order,\none takeaway"),
                ("03 / Visual", "Editable shapes,\nclear reveals"),
            ]
        ]
    ).arrange(RIGHT, buff=0.7)
    cards.shift(DOWN * 0.1)
    arrows = [
        Arrow(
            left.get_right(),
            right.get_left(),
            buff=0.12,
            color=scene.theme.accent_primary,
        )
        for left, right in pairwise(cards)
    ]
    takeaway = scene.body_text("Keep the meaning. Improve the visual.", font_size=30)
    takeaway.next_to(cards, DOWN, buff=0.65)

    scene.play(FadeIn(header), FadeIn(cards[0]), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(notes="A screenshot plus exact labels is enough for a first pass.")
    scene.play(Create(arrows[0]), FadeIn(cards[1]), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(notes="Agree on the story before spending time on polish.")
    scene.play(Create(arrows[1]), FadeIn(cards[2]), FadeIn(takeaway), run_time=0.6)
    scene.wait(0.2)
    scene.next_slide(notes="Review each pause state, then render the whole deck.")
