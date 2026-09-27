import importlib.util
import json
import types

import pytest
from PIL import Image
from typer.testing import CliRunner

from vizz import cli, review
from vizz.presentations.components import SlideBase
from vizz.presentations.theme import NUGGETS_LIGHT_THEME

runner = CliRunner()
DECK = "talk"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "vizz/presentations" / DECK / "slides").mkdir(parents=True)
    return tmp_path


def write_frames(directory, count):
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(1, count + 1):
        Image.new("RGB", (8, 8), (index * 20, 0, 0)).save(
            directory / f"beat-{index:02d}.png"
        )


def test_merge_groups_frames_by_slide_and_keeps_previous_render(repo):
    beats = [
        {"slide": "title", "notes": "a", "texts": []},
        {"slide": "api", "notes": "b", "texts": []},
        {"slide": "api", "notes": "c", "texts": []},
    ]
    write_frames(repo / "all", 3)
    state = review.merge_render(repo, DECK, ["title", "api"], beats, repo / "all")
    assert [b["notes"] for b in state["slides"]["api"]["beats"]] == ["b", "c"]
    first = state["slides"]["api"]["beats"][0]["image"]

    # Re-rendering one slide replaces only that slide and remembers the old frame.
    write_frames(repo / "api_only", 2)
    state = review.merge_render(
        repo, DECK, ["title", "api"], beats[1:], repo / "api_only"
    )
    api = state["slides"]["api"]["beats"][0]
    assert api["previous_image"] == first and api["image"] != first
    assert (review.review_dir(repo, DECK) / api["image"]).is_file()
    assert state["slides"]["title"]["beats"][0]["notes"] == "a"


def test_merge_rejects_frame_count_mismatch(repo):
    write_frames(repo / "all", 2)
    with pytest.raises(ValueError, match="cannot map frames"):
        review.merge_render(repo, DECK, ["title"], [{"slide": "title"}], repo / "all")


def test_apply_wording_replaces_only_unique_source_text(repo):
    slide = repo / "vizz/presentations" / DECK / "slides/title.py"
    slide.write_text('A = "Bringing flexible\\nepilogues"\nB = "dup"\nC = "dup"\n')
    items = [
        {
            "id": "w1",
            "kind": "wording",
            "status": "open",
            "slide": "title",
            "old": "Bringing flexible\nepilogues",
            "new": "Flexible\nepilogues",
        },
        {"id": "w2", "kind": "wording", "status": "open", "old": "dup", "new": "x"},
        {"id": "w3", "kind": "notes", "status": "open", "old": "missing", "new": "y"},
    ]
    review.save_feedback(repo, DECK, items)
    applied = review.apply_wording(repo, DECK)
    assert [item["id"] for item in applied] == ["w1"]
    assert 'A = "Flexible\\nepilogues"' in slide.read_text()
    assert slide.read_text().count('"dup"') == 2
    status = {
        i["id"]: (i["status"], i["reply"]) for i in review.load_feedback(repo, DECK)
    }
    assert status["w2"][0] == "question" and "2 times" in status["w2"][1]
    assert status["w3"][0] == "question" and "not found" in status["w3"][1]


def test_pause_records_slide_key_and_editable_source_text(tmp_path):
    fake = types.SimpleNamespace(theme=NUGGETS_LIGHT_THEME)
    label = SlideBase.meta_text(fake, "Flex slot")
    # describe_pause is called from inside a slide module's build().
    source = tmp_path / "title.py"
    source.write_text(
        "from vizz import review\n\n"
        "def build(scene):\n    return review.describe_pause(scene, 'hello')\n"
    )
    spec = importlib.util.spec_from_file_location(
        "vizz.presentations.talk.slides.title", source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Deck:
        __module__ = "vizz.presentations.talk.build"

    deck = Deck()
    deck.mobjects = [label]
    pause = module.build(deck)
    assert pause["slide"] == "title" and pause["notes"] == "hello"
    [text] = pause["texts"]
    assert text["text"] == "Flex slot"  # source text, not the uppercased render
    x0, y0, x1, y1 = text["box"]
    assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1


def test_feedback_cli_lists_open_items_and_records_resolution(repo):
    review.save_feedback(
        repo,
        DECK,
        [
            {
                "id": "v1",
                "kind": "visual",
                "status": "open",
                "slide": "title",
                "beat": 2,
                "tag": "layout",
                "text": "line up the box",
                "marks": [
                    {"type": "box", "x0": 0.5, "y0": 0.5, "x1": 0.75, "y1": 0.25}
                ],
            },
            {"id": "v2", "kind": "visual", "status": "done", "text": "old"},
        ],
    )
    result = runner.invoke(cli.app, ["feedback", "list", DECK])
    assert result.exit_code == 0, result.output
    assert "line up the box" in result.output and "v2" not in result.output
    assert "box scene (0.0, 0.0) to" in result.output

    result = runner.invoke(
        cli.app,
        ["feedback", "resolve", DECK, "v1", "--status", "fixed", "--reply", "centred"],
    )
    assert result.exit_code == 0, result.output
    [item, _] = json.loads(review.feedback_path(repo, DECK).read_text())
    assert (item["status"], item["reply"]) == ("fixed", "centred")
    assert (
        runner.invoke(
            cli.app, ["feedback", "resolve", DECK, "nope", "--status", "fixed"]
        ).exit_code
        != 0
    )


def test_submission_handshake_applies_wording_and_hands_rest_to_agent(repo):
    slide = repo / "vizz/presentations" / DECK / "slides/title.py"
    slide.write_text('A = "Old title"\n')
    review.save_feedback(
        repo,
        DECK,
        [
            {
                "id": "w",
                "kind": "wording",
                "status": "open",
                "slide": "title",
                "old": "Old title",
                "new": "New title",
            },
            {
                "id": "v",
                "kind": "visual",
                "status": "open",
                "slide": "title",
                "text": "move the box",
            },
        ],
    )
    submission = review.submit(repo, DECK)
    assert submission["applied_wording"] == ["w"] and submission["items"] == ["v"]
    assert "New title" in slide.read_text()

    claimed = review.wait_for_submission(repo, DECK, timeout=0)
    assert claimed["id"] == submission["id"] and claimed["status"] == "in_progress"
    assert review.wait_for_submission(repo, DECK, timeout=0) is None  # already claimed

    # A second send only includes items created since the first.
    assert review.submit(repo, DECK)["items"] == []
    done = review.finish_submission(repo, DECK, "moved it")
    assert (done["status"], done["message"]) == ("done", "moved it")
    with pytest.raises(LookupError):
        review.finish_submission(repo, DECK, "again")


def test_archive_moves_finished_items_and_prunes_unused_frames(repo):
    write_frames(repo / "all", 1)
    state = review.merge_render(
        repo, DECK, ["title"], [{"slide": "title"}], repo / "all"
    )
    current = state["slides"]["title"]["beats"][0]["image"]
    directory = review.review_dir(repo, DECK)
    stale = directory / "frames/title/old-01.png"
    kept_by_item = directory / "frames/title/older-01.png"
    done_note = directory / "annotations/d.png"
    for path in (stale, kept_by_item, done_note):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
    review.save_feedback(
        repo,
        DECK,
        [
            {
                "id": "d",
                "kind": "visual",
                "status": "done",
                "annotated": "annotations/d.png",
            },
            {
                "id": "o",
                "kind": "visual",
                "status": "open",
                "image": "frames/title/older-01.png",
            },
        ],
    )
    assert review.archive(repo, DECK) == 1
    assert [i["id"] for i in review.load_feedback(repo, DECK)] == ["o"]
    archived = json.loads(
        review.feedback_path(repo, DECK).with_name("archive.json").read_text()
    )
    assert [i["id"] for i in archived] == ["d"]
    assert not stale.exists() and not done_note.exists()
    assert kept_by_item.exists() and (directory / current).exists()
