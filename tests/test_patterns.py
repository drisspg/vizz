from contextlib import ExitStack
from pathlib import Path

import manimpango
import numpy as np
import pytest
from manim import Group, Rectangle, register_font
from typer.testing import CliRunner

from vizz import cli
from vizz.presentations.components import SlideBase
from vizz.presentations.layouts import Comparison
from vizz.presentations.patterns.build import PatternGallery
from vizz.presentations.theme import (
    LIGHT_THEME,
    NUGGETS_DARK_THEME,
    NUGGETS_LIGHT_THEME,
)


@pytest.mark.parametrize("theme", [NUGGETS_LIGHT_THEME, NUGGETS_DARK_THEME])
def test_fonts_are_available_during_scene_construction(theme):
    class FontProbe(SlideBase):
        def build_slides(self):
            fonts = manimpango.list_fonts()
            assert self.theme.sans_font in fonts
            assert self.theme.mono_font in fonts
            assert self.title_text("Query 5").width > 0
            assert self.meta_text("key 2").height > 0

    scene = FontProbe()
    scene.theme = theme
    scene.construct()
    assert scene.camera.background_color.to_hex().lower() == theme.background


def test_comparison_preserves_relative_scale_and_alignment():
    left = Rectangle(width=8, height=4)
    right = Rectangle(width=4, height=2)
    with ExitStack() as fonts:
        for font in NUGGETS_LIGHT_THEME.font_files:
            fonts.enter_context(register_font(font))
        layout = Comparison(
            left,
            right,
            labels=("Baseline", "Candidate"),
            theme=NUGGETS_LIGHT_THEME,
            width=10,
            height=3,
            gap=1,
        )
    assert left.width / right.width == pytest.approx(2)
    assert left.height / right.height == pytest.approx(2)
    assert left.width == pytest.approx(4.5)
    assert left.get_y() == right.get_y()
    assert layout.labels[0].get_y() == layout.labels[1].get_y()
    assert layout.left is left and layout.right is right
    assert left.get_right()[0] < layout.divider.get_x() < right.get_left()[0]
    before = left.get_center().copy()
    layout.shift(np.array([1, 2, 0]))
    np.testing.assert_allclose(left.get_center(), before + [1, 2, 0])


@pytest.mark.parametrize(
    "kwargs", [{"width": 0}, {"height": 0}, {"gap": -1}, {"gap": 12}]
)
def test_comparison_rejects_invalid_dimensions(kwargs):
    with pytest.raises(ValueError, match="width > gap"):
        Comparison(
            Rectangle(), Rectangle(), labels=("A", "B"), theme=LIGHT_THEME, **kwargs
        )


def test_comparison_rejects_aliasing_or_empty_content():
    rect = Rectangle()
    for other in (rect, Group()):
        with pytest.raises(ValueError, match="distinct, nonempty"):
            Comparison(rect, other, labels=("A", "B"), theme=LIGHT_THEME)


def test_patterns_template_uses_renamed_imports(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cli.app, ["new", "my_patterns", "--template", "patterns"]
    )
    assert result.exit_code == 0, result.output
    assert "--slide focus" in result.output
    root = tmp_path / "vizz/presentations/my_patterns"
    source = (root / "build.py").read_text()
    assert "from vizz.presentations.my_patterns.slides" in source
    assert "class MyPatternsDeck" in source
    assert "PatternGallery" not in source
    assert (root / "slides/tiles.py").is_file()
    assert (root / "brief.md").is_file()
    compile(source, str(root / "build.py"), "exec")


@pytest.mark.parametrize(
    "choice,expected",
    [
        (cli.ThemeChoice.light, NUGGETS_LIGHT_THEME),
        (cli.ThemeChoice.dark, NUGGETS_DARK_THEME),
    ],
)
@pytest.mark.parametrize("preview", [True, False])
def test_theme_override_is_instance_local_and_preview_outputs_are_separate(
    monkeypatch, choice, expected, preview
):
    original = PatternGallery.theme
    seen = []

    def fake_render(self):
        seen.append(self.theme)

    monkeypatch.setattr(PatternGallery, "render", fake_render)
    manifest, _ = cli.render_deck(
        "patterns",
        "focus" if preview else "",
        cli.Quality.low,
        preview=preview,
        theme=choice,
    )
    assert seen == [expected]
    assert PatternGallery.theme is original
    if preview:
        assert (
            manifest.parent
            == cli.ROOT / "media/review/patterns/focus" / choice.value / "slides"
        )
    else:
        assert manifest.parent == cli.ROOT / "slides"


def test_explicit_and_default_preview_themes_do_not_collide(monkeypatch):
    monkeypatch.setattr(PatternGallery, "render", lambda self: None)
    manifests = [
        cli.render_deck(
            "patterns", "focus", cli.Quality.low, preview=True, theme=choice
        )[0]
        for choice in cli.ThemeChoice
    ]
    assert len(set(manifests)) == 3
    assert all(isinstance(item, Path) for item in manifests)
