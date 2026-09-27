"""Repo-local deck scaffolding and rendering; previews never overwrite live slides."""

import html
import importlib
import json
import keyword
import os
import re
import shutil
import tomllib
from enum import Enum
from pathlib import Path

import av
import typer
from manim import tempconfig

from vizz import review
from vizz.presentations.theme import NUGGETS_DARK_THEME, NUGGETS_LIGHT_THEME

ROOT = Path(__file__).resolve().parents[1]
STARTER = Path(__file__).resolve().parent / "presentations" / "starter"
app = typer.Typer(
    no_args_is_help=True, help="Create, render, and review Vizz presentations."
)


class Quality(str, Enum):
    low = "l"
    medium = "m"
    high = "h"


def deck_path(name: str) -> Path:
    if Path.cwd().resolve() != ROOT:
        raise typer.BadParameter(f"Run from the repository root: {ROOT}")
    if not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", name) or keyword.iskeyword(
        name
    ):
        raise typer.BadParameter("Use a lowercase Python name, e.g. attention_intro.")
    return ROOT / "vizz" / "presentations" / name


class Template(str, Enum):
    starter = "starter"
    patterns = "patterns"


class ThemeChoice(str, Enum):
    deck = "deck"
    light = "light"
    dark = "dark"


@app.command()
def new(name: str, template: Template = Template.starter) -> None:
    """Copy a deck template, brief, and sketch handoff without overwriting work."""
    destination = deck_path(name)
    if destination.exists() or destination.with_suffix(".py").exists():
        raise typer.BadParameter(
            f"Already exists: {destination} (directory or Python module)"
        )
    scene_name = "".join(part.capitalize() for part in name.split("_")) + "Deck"
    for manifest in destination.parent.glob("*/deck.toml"):
        if tomllib.loads(manifest.read_text())["scene"] == scene_name:
            raise typer.BadParameter(
                f"Scene name {scene_name} already used by {manifest.parent.name}; choose another name."
            )
    source = STARTER.with_name(template.value)
    source_scene = tomllib.loads((source / "deck.toml").read_text())["scene"]
    shutil.copytree(
        source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for item in (destination / "build.py", destination / "deck.toml"):
        item.write_text(
            item.read_text()
            .replace(
                f"vizz.presentations.{template.value}", f"vizz.presentations.{name}"
            )
            .replace(source_scene, scene_name)
        )
    typer.echo(f"Created {destination.relative_to(ROOT)}")
    first_slide = "focus" if template == Template.patterns else "workflow"
    typer.echo(
        f"Edit brief.md and scenes.md, then: uv run vizz preview {name} --slide {first_slide}"
    )


def render_deck(
    name: str,
    slide: str,
    quality: Quality,
    *,
    preview: bool,
    theme: ThemeChoice = ThemeChoice.deck,
    beat_log: list[dict] | None = None,
) -> tuple[Path, str]:
    directory = deck_path(name)
    manifest = directory / "deck.toml"
    if not manifest.is_file():
        raise typer.BadParameter(
            f"No deck.toml in {directory}; create a deck with 'vizz new NAME'."
        )
    scene_name = tomllib.loads(manifest.read_text())["scene"]
    module = importlib.import_module(f"vizz.presentations.{name}.build")
    if slide and slide not in module.SLIDES:
        raise typer.BadParameter(
            f"Unknown slide {slide!r}; choose: {', '.join(module.SLIDES)}"
        )
    output = (
        ROOT / "media" / "review" / name / (slide or "all")
        if preview
        else ROOT / "media" / name
    )
    if theme != ThemeChoice.deck:
        output /= theme.value
    slides = output / "slides" if preview else ROOT / "slides"
    previous = os.environ.get("SLIDE")
    if slide:
        os.environ["SLIDE"] = slide
    else:
        os.environ.pop("SLIDE", None)
    try:
        with tempconfig(
            {
                "quality": {
                    Quality.low: "low_quality",
                    Quality.medium: "medium_quality",
                    Quality.high: "high_quality",
                }[quality],
                "media_dir": str(output),
                "renderer": "cairo",
                "preview": False,
            }
        ):
            scene = getattr(module, scene_name)(output_folder=slides)
            if theme != ThemeChoice.deck:
                scene.theme = (
                    NUGGETS_DARK_THEME
                    if theme == ThemeChoice.dark
                    else NUGGETS_LIGHT_THEME
                )
            if preview:
                scene.skip_reversing = True
            if beat_log is not None:
                scene.beat_log = beat_log
            scene.render()
    finally:
        if previous is None:
            os.environ.pop("SLIDE", None)
        else:
            os.environ["SLIDE"] = previous
    return slides / f"{scene_name}.json", scene_name


@app.command()
def render(
    name: str, quality: Quality = Quality.low, theme: ThemeChoice = ThemeChoice.deck
) -> None:
    """Render a complete deck for presenting/exporting (l, m, or h quality)."""
    manifest, scene_name = render_deck(name, "", quality, preview=False, theme=theme)
    typer.echo(f"Slides: {manifest}")
    typer.echo(f"Present: uv run manim-slides present {scene_name}")


def write_gallery(manifest: Path, output: Path) -> Path:
    """Extract exact last frames of pause clips, not the often-empty deck ending."""
    output.mkdir(parents=True, exist_ok=True)
    cards = []
    for number, slide in enumerate(json.loads(manifest.read_text())["slides"], start=1):
        with av.open(str(slide["file"])) as video:
            last_frame = None
            for frame in video.decode(video=0):
                last_frame = frame
            if last_frame is None:
                raise ValueError(f"No video frames in {slide['file']}")
            filename = f"beat-{number:02d}.png"
            last_frame.to_image().save(output / filename)
        notes = html.escape(slide.get("notes", ""))
        cards.append(
            f'<figure><img src="{filename}" alt="Beat {number}"><figcaption>Beat {number}: {notes}</figcaption></figure>'
        )
    if not cards:
        raise ValueError(
            f"No slides in {manifest}; add animations and next_slide() calls."
        )
    index = output / "index.html"
    index.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Vizz pause-state review</title><style>"
        "body{font:18px system-ui;background:#eee;margin:2rem;max-width:1100px}"
        "figure{margin:0 0 2rem}img{width:100%;border:1px solid #ccc}"
        "figcaption{padding:.5rem 0;white-space:pre-wrap}</style>"
        "<h1>Pause-state review</h1><p>Review each reveal for clipping, labels, and meaning. "
        "Watch the video too when reviewing motion.</p>" + "".join(cards) + "</html>"
    )
    return index


@app.command()
def preview(
    name: str,
    slide: str = typer.Option(
        "", help="A SLIDES registry key; omit to review the whole deck."
    ),
    theme: ThemeChoice = ThemeChoice.deck,
) -> None:
    """Render a low-quality HTML/PNG review gallery, isolated from live slides."""
    beats: list[dict] = []
    manifest, _ = render_deck(
        name, slide, Quality.low, preview=True, theme=theme, beat_log=beats
    )
    index = write_gallery(manifest, manifest.parent.parent / "frames")
    if theme == ThemeChoice.deck:
        build = importlib.import_module(f"vizz.presentations.{name}.build")
        review.merge_render(ROOT, name, list(build.SLIDES), beats, index.parent)
    typer.echo(f"Review: {index}")
    typer.echo(f"Open: open {index}")


@app.command("review")
def review_command(
    name: str,
    port: int = typer.Option(8765, help="Local port for the review page."),
    render: bool = typer.Option(
        False, help="Render the whole deck first (automatic when no frames exist)."
    ),
) -> None:
    """Serve a markup page: edit wording and pin/box/draw comments on each pause."""
    deck_path(name)
    if render or not (review.review_dir(ROOT, name) / "state.json").is_file():
        preview(name, slide="", theme=ThemeChoice.deck)
    review.serve(ROOT, name, port)


feedback_app = typer.Typer(
    no_args_is_help=True, help="Read and act on review feedback."
)
app.add_typer(feedback_app, name="feedback")


@feedback_app.command("list")
def feedback_list(
    name: str,
    all_items: bool = typer.Option(False, "--all", help="Include resolved items."),
    as_json: bool = typer.Option(False, "--json", help="Print raw JSON."),
) -> None:
    """Print open feedback (with annotated image paths) for an agent to act on."""
    deck_path(name)
    items = review.load_feedback(ROOT, name)
    if not all_items:
        items = [item for item in items if item["status"] in review.OPEN_STATUSES]
    if as_json:
        typer.echo(json.dumps(items, indent=1, ensure_ascii=False))
    else:
        typer.echo(review.format_feedback(ROOT, name, items))


@feedback_app.command("apply-wording")
def feedback_apply_wording(name: str) -> None:
    """Apply open wording edits whose old text appears exactly once in the deck."""
    deck_path(name)
    for item in review.apply_wording(ROOT, name):
        typer.echo(f"applied {item['id']}: {item['reply']}")
    for item in review.load_feedback(ROOT, name):
        if item["kind"] in {"wording", "notes"} and item["status"] == "question":
            typer.echo(f"needs agent {item['id']}: {item['reply']}")


@feedback_app.command("wait")
def feedback_wait(
    name: str,
    timeout: float = typer.Option(3600, help="Seconds to wait before giving up."),
) -> None:
    """Block until the review page sends feedback; print it and claim it."""
    deck_path(name)
    submission = review.wait_for_submission(ROOT, name, timeout)
    if submission is None:
        typer.echo("No submission before timeout.")
        raise typer.Exit(2)
    items = [
        item
        for item in review.load_feedback(ROOT, name)
        if item["id"] in submission["items"]
    ]
    typer.echo(f"Submission {submission['id']}: {len(items)} item(s)")
    if submission.get("applied_wording"):
        typer.echo(
            f"Wording already applied: {', '.join(submission['applied_wording'])}"
        )
    typer.echo(review.format_feedback(ROOT, name, items))


@feedback_app.command("done")
def feedback_done(
    name: str,
    message: str = typer.Option("", help="Summary shown on the review page."),
) -> None:
    """Mark the claimed submission finished so the page shows the update."""
    deck_path(name)
    try:
        submission = review.finish_submission(ROOT, name, message)
    except LookupError as error:
        raise typer.BadParameter(str(error)) from None
    typer.echo(f"{submission['id']}: done")


@feedback_app.command("archive")
def feedback_archive(name: str) -> None:
    """Move done/wontfix items to archive.json and delete unused review media."""
    deck_path(name)
    typer.echo(f"archived {review.archive(ROOT, name)} item(s)")


@feedback_app.command("resolve")
def feedback_resolve(
    name: str,
    item_id: str,
    status: str = typer.Option(..., help="fixed, question, wontfix, or open."),
    reply: str = typer.Option("", help="One-line reply shown in the review page."),
) -> None:
    """Record the outcome of one feedback item."""
    deck_path(name)
    if status not in {"fixed", "question", "wontfix", "open"}:
        raise typer.BadParameter("status must be fixed, question, wontfix, or open")
    try:
        item = review.resolve(ROOT, name, item_id, status, reply)
    except KeyError:
        raise typer.BadParameter(f"No feedback item {item_id!r}") from None
    typer.echo(f"{item['id']}: {item['status']}")


if __name__ == "__main__":
    app()
