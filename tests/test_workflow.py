import json
import os
import tomllib
from pathlib import Path

import av
import pytest
from manim import config
from PIL import Image
from typer.testing import CliRunner

from vizz import cli
from vizz.presentations.starter.build import StarterDeck

runner = CliRunner()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_new_deck_is_editable_and_does_not_overwrite(repo):
    result = runner.invoke(cli.app, ["new", "my_talk"])
    assert result.exit_code == 0, result.output
    deck = repo / "vizz/presentations/my_talk"
    assert tomllib.loads((deck / "deck.toml").read_text())["scene"] == "MyTalkDeck"
    code = (deck / "build.py").read_text()
    assert "vizz.presentations.my_talk.slides" in code
    assert "class MyTalkDeck" in code
    compile(code, str(deck / "build.py"), "exec")
    assert (deck / "sketches").is_dir()
    assert not list(deck.rglob("*.pyc"))
    assert (deck / "scenes.md").is_file()
    (deck / "brief.md").write_text("User's brief")
    result = runner.invoke(cli.app, ["new", "my_talk"])
    assert result.exit_code != 0
    assert "Already exists" in result.output
    assert (deck / "brief.md").read_text() == "User's brief"


@pytest.mark.parametrize(
    "name", ["../escape", "bad-name", "BadName", "1talk", "class", "bad__name"]
)
def test_invalid_deck_name(repo, name):
    result = runner.invoke(cli.app, ["new", name])
    assert result.exit_code != 0
    assert "lowercase Python name" in result.output


def test_new_deck_does_not_shadow_shared_module(repo):
    module = repo / "vizz/presentations/theme.py"
    module.parent.mkdir(parents=True)
    module.write_text("# Shared theme")
    result = runner.invoke(cli.app, ["new", "theme"])
    assert result.exit_code != 0
    assert "Already exists" in result.output
    assert not module.with_suffix("").exists()
    assert module.read_text() == "# Shared theme"


def test_new_deck_rejects_scene_name_collision(repo):
    assert runner.invoke(cli.app, ["new", "talk1"]).exit_code == 0
    result = runner.invoke(cli.app, ["new", "talk_1"])
    assert result.exit_code != 0
    assert "Scene name Talk1Deck already used" in result.output
    assert not (repo / "vizz/presentations/talk_1").exists()


def test_wrong_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["new", "talk"])
    assert result.exit_code != 0
    assert "repository root" in result.output


def test_unknown_deck():
    result = runner.invoke(cli.app, ["render", "does_not_exist"])
    assert result.exit_code != 0
    assert "No deck.toml" in result.output


def test_unknown_slide_fails_before_render(monkeypatch):
    def unexpected_render(self):
        pytest.fail("An invalid selection must not start rendering")

    monkeypatch.setattr(StarterDeck, "render", unexpected_render)
    result = runner.invoke(cli.app, ["preview", "starter", "--slide", "typo"])
    assert result.exit_code != 0
    assert "title, workflow" in result.output


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_render_isolation_and_environment_restoration(monkeypatch, preview, fail):
    monkeypatch.setenv("SLIDE", "inherited_selection")
    original_media_dir = config.media_dir
    seen = []

    def fake_render(self):
        seen.append(
            (os.environ.get("SLIDE"), Path(config.media_dir), self._output_folder)
        )
        if preview:
            assert self.skip_reversing is True
        if fail:
            raise RuntimeError("render failure")

    monkeypatch.setattr(StarterDeck, "render", fake_render)
    if fail:
        with pytest.raises(RuntimeError, match="render failure"):
            cli.render_deck(
                "starter", "title" if preview else "", cli.Quality.low, preview=preview
            )
    else:
        manifest, name = cli.render_deck(
            "starter", "title" if preview else "", cli.Quality.low, preview=preview
        )
        assert manifest.name == f"{name}.json"
    assert os.environ["SLIDE"] == "inherited_selection"
    assert config.media_dir == original_media_dir
    selection, media, slides = seen[0]
    assert selection == ("title" if preview else None)
    if preview:
        assert media == cli.ROOT / "media/review/starter/title"
        assert slides == media / "slides"
    else:
        assert slides == cli.ROOT / "slides"


def test_gallery_extracts_last_frame_and_escapes_notes(tmp_path):
    clip = tmp_path / "beat.mp4"
    with av.open(str(clip), "w") as video:
        stream = video.add_stream("libx264", rate=15)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for color in ["red", "green", "blue"]:
            frame = av.VideoFrame.from_image(Image.new("RGB", (32, 32), color))
            for packet in stream.encode(frame):
                video.mux(packet)
        for packet in stream.encode():
            video.mux(packet)
    manifest = tmp_path / "Deck.json"
    manifest.write_text(
        json.dumps({"slides": [{"file": str(clip), "notes": "<script>oops</script>"}]})
    )
    gallery = cli.write_gallery(manifest, tmp_path / "frames")
    assert "&lt;script&gt;" in gallery.read_text()
    with Image.open(gallery.parent / "beat-01.png") as image:
        red, green, blue = image.getpixel((16, 16))
        assert blue > 240 and red < 10 and green < 10


def test_empty_gallery_is_an_error(tmp_path):
    manifest = tmp_path / "Empty.json"
    manifest.write_text('{"slides": []}')
    with pytest.raises(ValueError, match="No slides"):
        cli.write_gallery(manifest, tmp_path / "frames")
