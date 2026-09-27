"""Review loop: per-pause frames, a markup page, and a feedback file agents can act on.

Two feedback lanes share `vizz/presentations/<deck>/review/feedback.json`:
  * wording: exact old -> new text edits, applied mechanically by `apply_wording`;
  * visual: pins, boxes, and pen marks with a note, handed to an agent.
"""

from __future__ import annotations

import base64
import inspect
import json
import shutil
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from manim import Code, MarkupText, Mobject, Text, config

APP = Path(__file__).resolve().parent / "review_app" / "index.html"
OPEN_STATUSES = {"open", "question"}


# ---------------------------------------------------------------------------
# Recording pauses during a preview render


def _slide_key(deck_package: str) -> str:
    """Name of the slide module whose build() is on the stack, or "" if none."""
    marker = f"{deck_package}.slides."
    for frame in inspect.stack(context=0):
        module = frame.frame.f_globals.get("__name__", "")
        if module.startswith(marker):
            return module[len(marker) :].split(".")[0]
    return ""


def _normalized_box(mobject: Mobject) -> list[float]:
    width, height = config.frame_width, config.frame_height
    left, bottom = mobject.get_corner([-1, -1, 0])[:2]
    right, top = mobject.get_corner([1, 1, 0])[:2]
    return [
        round((left + width / 2) / width, 4),
        round((height / 2 - top) / height, 4),
        round((right + width / 2) / width, 4),
        round((height / 2 - bottom) / height, 4),
    ]


def _texts(mobject: Mobject, found: list[dict]) -> None:
    if isinstance(mobject, Code):
        # SlideBase.themed_code stores the source; other Code objects are opaque.
        source = getattr(mobject, "source_text", "")
        if source:
            found.append({"text": source, "box": _normalized_box(mobject)})
        return
    if isinstance(mobject, Text | MarkupText):
        text = (
            getattr(mobject, "source_text", None)
            or getattr(mobject, "original_text", None)
            or mobject.text
        )
        if text.strip():
            found.append({"text": text, "box": _normalized_box(mobject)})
        return
    for child in mobject.submobjects:
        _texts(child, found)


def describe_pause(scene, notes: str) -> dict:
    """One review entry for the scene's current pause state."""
    deck_package = type(scene).__module__.rsplit(".", 1)[0]
    texts: list[dict] = []
    for mobject in scene.mobjects:
        _texts(mobject, texts)
    return {"slide": _slide_key(deck_package), "notes": notes, "texts": texts}


# ---------------------------------------------------------------------------
# Review state: latest frames per slide, keeping one previous render for diffs


def review_dir(root: Path, deck: str) -> Path:
    return root / "media" / "review" / deck / "review"


def feedback_path(root: Path, deck: str) -> Path:
    return root / "vizz" / "presentations" / deck / "review" / "feedback.json"


def load_state(root: Path, deck: str) -> dict:
    path = review_dir(root, deck) / "state.json"
    return json.loads(path.read_text()) if path.is_file() else {"slides": {}}


def merge_render(
    root: Path, deck: str, order: list[str], beats: list[dict], frames: Path
) -> dict:
    """Copy a render's frames into the review state, grouped by slide key."""
    images = sorted(frames.glob("beat-*.png"))
    if len(images) != len(beats):
        raise ValueError(
            f"{len(images)} frames but {len(beats)} recorded pauses; cannot map frames to slides"
        )
    directory = review_dir(root, deck)
    state = load_state(root, deck)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    grouped: dict[str, list[tuple[dict, Path]]] = {}
    for beat, image in zip(beats, images, strict=True):
        grouped.setdefault(beat["slide"] or "unknown", []).append((beat, image))
    for key, items in grouped.items():
        target = directory / "frames" / key
        target.mkdir(parents=True, exist_ok=True)
        previous = state["slides"].get(key, {}).get("beats", [])
        entries = []
        for index, (beat, image) in enumerate(items, start=1):
            name = f"{stamp}-{index:02d}.png"
            shutil.copyfile(image, target / name)
            entry = {**beat, "image": f"frames/{key}/{name}"}
            if index <= len(previous):
                entry["previous_image"] = previous[index - 1]["image"]
            entries.append(entry)
        state["slides"][key] = {"stamp": stamp, "beats": entries}
    state["order"] = order
    state["deck"] = deck
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "state.json").write_text(json.dumps(state, indent=1))
    return state


# ---------------------------------------------------------------------------
# Feedback file


def load_feedback(root: Path, deck: str) -> list[dict]:
    path = feedback_path(root, deck)
    return json.loads(path.read_text()) if path.is_file() else []


def save_feedback(root: Path, deck: str, items: list[dict]) -> None:
    path = feedback_path(root, deck)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=1, ensure_ascii=False) + "\n")


def to_scene(x: float, y: float) -> tuple[float, float]:
    """Normalized frame position (0..1, y down) to Manim scene units."""
    return (
        round((x - 0.5) * config.frame_width, 2),
        round((0.5 - y) * config.frame_height, 2),
    )


def describe_mark(mark: dict) -> str:
    kind = mark["type"]
    if kind == "pin":
        return f"pin at scene {to_scene(mark['x'], mark['y'])}"
    if kind == "box":
        return f"box scene {to_scene(mark['x0'], mark['y0'])} to {to_scene(mark['x1'], mark['y1'])}"
    if kind == "arrow" and "x0" in mark:
        start, end = to_scene(mark["x0"], mark["y0"]), to_scene(mark["x1"], mark["y1"])
        return f"arrow from scene {start} to {end}"
    points = mark.get("points", [])
    if not points:
        return kind
    return f"{kind} from scene {to_scene(*points[0])} to {to_scene(*points[-1])}"


def format_feedback(root: Path, deck: str, items: list[dict]) -> str:
    """Markdown summary of feedback items for an agent turn."""
    directory = review_dir(root, deck)
    lines = []
    for item in items:
        where = item.get("slide") or "deck"
        if item.get("beat"):
            where += f" beat {item['beat']}"
        lines.append(f"- [{item['id']}] {item['kind']} · {where} · {item['status']}")
        if item.get("text"):
            lines.append(f"  note: {item['text']}")
        if item["kind"] in {"wording", "notes"}:
            lines.append(f"  old: {item.get('old', '')!r}")
            lines.append(f"  new: {item.get('new', '')!r}")
        for mark in item.get("marks", []):
            lines.append(f"  mark: {describe_mark(mark)}")
        for key in ("annotated", "image"):
            if item.get(key):
                lines.append(f"  {key}: {directory / item[key]}")
        if item.get("reply"):
            lines.append(f"  reply: {item['reply']}")
    return "\n".join(lines) if lines else "No feedback items."


def _source_literals(text: str) -> list[str]:
    """Forms a runtime string can take in Python source."""
    forms = [text, text.replace("\n", "\\n")]
    return list(dict.fromkeys(forms))


def apply_wording(root: Path, deck: str) -> list[dict]:
    """Apply open wording/notes edits whose old text is unique in the deck sources."""
    items = load_feedback(root, deck)
    sources = sorted((root / "vizz" / "presentations" / deck).rglob("*.py"))
    applied = []
    for item in items:
        if item["kind"] not in {"wording", "notes"} or item["status"] != "open":
            continue
        old, new = item.get("old", ""), item.get("new", "")
        hits = []
        for path in sources:
            code = path.read_text()
            for form in _source_literals(old):
                count = code.count(form) if form else 0
                if count:
                    hits.append((path, form, count))
                    break
        total = sum(count for _, _, count in hits)
        if total != 1:
            item["status"] = "question"
            item["reply"] = (
                "old text not found in deck sources"
                if total == 0
                else f"old text appears {total} times; needs an agent"
            )
            continue
        path, form, _ = hits[0]
        replacement = new.replace("\n", "\\n") if form != old else new
        path.write_text(path.read_text().replace(form, replacement))
        item["status"] = "applied"
        item["reply"] = f"replaced in {path.relative_to(root)}"
        applied.append(item)
    save_feedback(root, deck, items)
    return applied


def resolve(root: Path, deck: str, item_id: str, status: str, reply: str) -> dict:
    items = load_feedback(root, deck)
    for item in items:
        if item["id"] == item_id:
            item["status"] = status
            item["reply"] = reply
            item["resolved"] = datetime.now(UTC).isoformat(timespec="seconds")
            save_feedback(root, deck, items)
            return item
    raise KeyError(item_id)


def archive(root: Path, deck: str) -> int:
    """Move done/wontfix items to archive.json and prune media nothing references."""
    items = load_feedback(root, deck)
    finished = [item for item in items if item["status"] in {"done", "wontfix"}]
    if finished:
        path = feedback_path(root, deck).with_name("archive.json")
        history = json.loads(path.read_text()) if path.is_file() else []
        path.write_text(
            json.dumps(history + finished, indent=1, ensure_ascii=False) + "\n"
        )
        save_feedback(root, deck, [item for item in items if item not in finished])
    prune_media(root, deck)
    return len(finished)


def prune_media(root: Path, deck: str) -> int:
    """Delete review frames/annotations not used by the state or open feedback."""
    directory = review_dir(root, deck)
    keep = set()
    for slide in load_state(root, deck)["slides"].values():
        for beat in slide["beats"]:
            keep.update(filter(None, (beat.get("image"), beat.get("previous_image"))))
    for item in load_feedback(root, deck):
        keep.update(filter(None, (item.get("image"), item.get("annotated"))))
    removed = 0
    for sub in ("frames", "annotations"):
        for path in (directory / sub).rglob("*.png"):
            if path.relative_to(directory).as_posix() not in keep:
                path.unlink()
                removed += 1
    return removed


# ---------------------------------------------------------------------------
# Submissions: the handshake between the review page and an agent
#
# The page's "Send to agent" applies wording edits, stamps every unsent open item
# with a submission id, and appends a `pending` submission. An agent blocks in
# `wait_for_submission` (CLI: `vizz feedback wait`), acts on the items, resolves
# them, and calls `finish_submission` (CLI: `vizz feedback done`).


def submissions_path(root: Path, deck: str) -> Path:
    return feedback_path(root, deck).with_name("submissions.json")


def load_submissions(root: Path, deck: str) -> list[dict]:
    path = submissions_path(root, deck)
    return json.loads(path.read_text()) if path.is_file() else []


def _save_submissions(root: Path, deck: str, submissions: list[dict]) -> None:
    path = submissions_path(root, deck)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(submissions, indent=1, ensure_ascii=False) + "\n")


def submit(root: Path, deck: str) -> dict:
    """Apply wording edits, then hand every unsent open item to the agent."""
    applied = apply_wording(root, deck)
    items = load_feedback(root, deck)
    now = datetime.now(UTC)
    submission = {
        "id": "s" + now.strftime("%Y%m%dT%H%M%S"),
        "created": now.isoformat(timespec="seconds"),
        "status": "pending",
        "items": [],
        "applied_wording": [item["id"] for item in applied],
    }
    for item in items:
        if item["status"] in OPEN_STATUSES and not item.get("submission"):
            item["submission"] = submission["id"]
            submission["items"].append(item["id"])
    save_feedback(root, deck, items)
    if submission["items"]:
        _save_submissions(root, deck, [*load_submissions(root, deck), submission])
    else:
        submission["status"] = "done"
        submission["message"] = "nothing needs an agent"
    return submission


def wait_for_submission(
    root: Path, deck: str, timeout: float, poll: float = 1.0
) -> dict | None:
    """Block until a pending submission exists; claim it and return it."""
    deadline = time.monotonic() + timeout
    while True:
        submissions = load_submissions(root, deck)
        for submission in submissions:
            if submission["status"] == "pending":
                submission["status"] = "in_progress"
                submission["claimed"] = datetime.now(UTC).isoformat(timespec="seconds")
                _save_submissions(root, deck, submissions)
                return submission
        if time.monotonic() >= deadline:
            return None
        time.sleep(poll)


def finish_submission(root: Path, deck: str, message: str) -> dict:
    submissions = load_submissions(root, deck)
    for submission in reversed(submissions):
        if submission["status"] == "in_progress":
            submission["status"] = "done"
            submission["message"] = message
            submission["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
            _save_submissions(root, deck, submissions)
            return submission
    raise LookupError("no submission in progress")


# ---------------------------------------------------------------------------
# Local review server


class _Renderer:
    """Runs one preview subprocess at a time so edited code is always reloaded."""

    def __init__(self, root: Path, deck: str) -> None:
        self.root, self.deck = root, deck
        self.lock = threading.Lock()
        self.running = ""
        self.started = 0.0
        self.log = ""

    def start(self, slide: str) -> bool:
        if not self.lock.acquire(blocking=False):
            return False
        self.running = slide or "all"
        self.started = time.monotonic()
        threading.Thread(target=self._run, args=(slide,), daemon=True).start()
        return True

    def _run(self, slide: str) -> None:
        command = [sys.executable, "-m", "vizz.cli", "preview", self.deck]
        if slide:
            command += ["--slide", slide]
        try:
            result = subprocess.run(
                command, cwd=self.root, capture_output=True, text=True, check=False
            )
            self.log = (result.stdout + result.stderr)[-4000:]
            if result.returncode:
                self.log = f"render failed ({result.returncode})\n" + self.log
        finally:
            self.running = ""
            self.lock.release()


def serve(root: Path, deck: str, port: int) -> None:
    directory = review_dir(root, deck)
    renderer = _Renderer(root, deck)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _send(self, body: bytes, kind: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value, status: int = 200) -> None:
            self._send(json.dumps(value).encode(), "application/json", status)

        def do_GET(self) -> None:
            path = unquote(urlparse(self.path).path)
            if path == "/":
                self._send(APP.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(
                    {
                        **load_state(root, deck),
                        "feedback": load_feedback(root, deck),
                        "submissions": load_submissions(root, deck)[-5:],
                        "rendering": renderer.running,
                        "render_seconds": round(time.monotonic() - renderer.started)
                        if renderer.running
                        else 0,
                        "log": renderer.log,
                    }
                )
            elif path.startswith("/files/"):
                target = (directory / path[len("/files/") :]).resolve()
                if directory.resolve() not in target.parents or not target.is_file():
                    self._send(b"not found", "text/plain", 404)
                else:
                    self._send(target.read_bytes(), "image/png")
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if path == "/api/feedback":
                save_feedback(root, deck, body["items"])
                self._json({"ok": True})
            elif path == "/api/annotation":
                data = body["png"].split(",", 1)[1]
                target = directory / "annotations" / f"{body['id']}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(base64.b64decode(data))
                self._json({"path": f"annotations/{body['id']}.png"})
            elif path == "/api/submit":
                submission = submit(root, deck)
                slides = sorted(
                    {
                        item["slide"]
                        for item in load_feedback(root, deck)
                        if item["id"] in submission["applied_wording"]
                        and item.get("slide")
                    }
                )
                if slides:
                    renderer.start(slides[0] if len(slides) == 1 else "")
                self._json(submission)
            elif path == "/api/archive":
                self._json({"archived": archive(root, deck)})
            elif path == "/api/render":
                started = renderer.start(body.get("slide", ""))
                self._json({"started": started}, 200 if started else 409)
            else:
                self._send(b"not found", "text/plain", 404)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Review: http://127.0.0.1:{port}/  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
