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


@app.command()
def new(name: str) -> None:
    """Copy the starter deck, brief, and sketch handoff without overwriting work."""
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
    shutil.copytree(
        STARTER, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for item in (destination / "build.py", destination / "deck.toml"):
        item.write_text(
            item.read_text()
            .replace("vizz.presentations.starter", f"vizz.presentations.{name}")
            .replace("StarterDeck", scene_name)
        )
    typer.echo(f"Created {destination.relative_to(ROOT)}")
    typer.echo(
        f"Edit brief.md and scenes.md, then: uv run vizz preview {name} --slide workflow"
    )


def render_deck(
    name: str, slide: str, quality: Quality, *, preview: bool
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
            if preview:
                scene.skip_reversing = True
            scene.render()
    finally:
        if previous is None:
            os.environ.pop("SLIDE", None)
        else:
            os.environ["SLIDE"] = previous
    return slides / f"{scene_name}.json", scene_name


@app.command()
def render(name: str, quality: Quality = Quality.low) -> None:
    """Render a complete deck for presenting/exporting (l, m, or h quality)."""
    manifest, scene_name = render_deck(name, "", quality, preview=False)
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
    slide: str = typer.Option(..., help="A key from build.py's SLIDES registry."),
) -> None:
    """Render one slide at low quality and write an HTML/PNG pause-state gallery."""
    if not slide:
        raise typer.BadParameter("Choose a slide from build.py's SLIDES registry.")
    manifest, _ = render_deck(name, slide, Quality.low, preview=True)
    index = write_gallery(manifest, manifest.parent.parent / "frames")
    typer.echo(f"Review: {index}")
    typer.echo(f"Open: open {index}")
