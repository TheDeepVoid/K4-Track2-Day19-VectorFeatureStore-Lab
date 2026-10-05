"""Render each executed notebook to HTML and screenshot it for the rubric.

`rubric.md` asks for "screenshots + notebook output for each criterion", and the
submission checklist wants at least one screenshot per notebook. This produces
them from the notebook itself — the rendered page a reviewer would see on
GitHub or in Jupyter — rather than a terminal transcript, so the evidence sits
next to the prose and code that produced it.

Run via `make screenshots`, or:

    python scripts/screenshot_evidence.py

For each executed `notebooks/*.ipynb`:
  1. `jupyter nbconvert --to html` with the `lab` template (the familiar Jupyter
     chrome: prompts, outputs, rendered markdown, tables).
  2. Headless Brave via Playwright loads the local file and screenshots the
     document, full height, capped so one notebook does not become a 20 000 px
     strip.

`--only NB` restricts the run; `--max-height N` changes the cap.

Non-fatal by design: a missing browser, a missing nbconvert, or a notebook that
has not been executed yet produces a warning and a skip, so this is safe to run
before `make notebooks`.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "submission" / "screenshots"
NB = ROOT / "notebooks"

BRAVE_CANDIDATES = [
    Path("/opt/brave-origin-bin/brave"),
    Path("/opt/brave.com/brave/brave-browser"),
    Path("/usr/bin/brave"),
    Path("/usr/bin/brave-browser"),
]

# Long notebooks get a very tall image. This is generous enough to show the
# bulk of the content while keeping the file a sane size.
DEFAULT_MAX_HEIGHT = 14000


def find_brave() -> Path | None:
    for p in BRAVE_CANDIDATES:
        if p.is_file() and p.stat().st_mode & 0o111:
            return p
    return None


def nbconvert_binary() -> Path | None:
    """Prefer the venv's nbconvert so the template matches the installed Jupyter."""
    for p in (ROOT / ".venv/bin/jupyter", ROOT / ".venv/bin/nbconvert"):
        if p.exists():
            return p
    found = shutil.which("jupyter") or shutil.which("nbconvert")
    return Path(found) if found else None


def is_executed(path: Path) -> bool:
    """True when at least one cell carries an output.

    An unexecuted notebook renders as source only, which is useless as evidence
    and worth skipping loudly rather than screenshotting.
    """
    import json

    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return any(c.get("outputs") for c in nb.get("cells", []) if c.get("cell_type") == "code")


def render_html(nb_path: Path, out_dir: Path) -> Path | None:
    """nbconvert one notebook to a standalone HTML file. Returns its path."""
    nbconvert = nbconvert_binary()
    if nbconvert is None:
        return None
    cmd = [str(nbconvert)]
    if nbconvert.name == "jupyter":
        cmd += ["nbconvert"]
    # --output-dir is essential: without it nbconvert writes the .html next to
    # the input notebook, so the temp directory stays empty and every render
    # looks like a failure.
    cmd += ["--to", "html", "--template", "lab",
            "--output-dir", str(out_dir), str(nb_path)]
    res = subprocess.run(cmd, cwd=str(out_dir), capture_output=True, text=True)
    produced = out_dir / (nb_path.stem + ".html")
    if res.returncode == 0 and produced.exists():
        return produced
    print(f"      nbconvert failed: {(res.stderr or res.stdout).strip()[:160]}")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", metavar="NB",
                    help="screenshot only these notebook stems, e.g. 01 04")
    ap.add_argument("--max-height", type=int, default=DEFAULT_MAX_HEIGHT,
                    help=f"cap image height in px (default {DEFAULT_MAX_HEIGHT})")
    args = ap.parse_args()

    brave = find_brave()
    if brave is None:
        print("  ! No Brave binary found. Looked in:")
        for p in BRAVE_CANDIDATES:
            print(f"      {p}")
        return 1

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  ! playwright not installed. Run: pip install playwright")
        return 1

    targets = sorted(NB.glob("[0-9]*.ipynb"))
    if args.only:
        wanted = {f"{n}" if not n[0].isdigit() else n for n in args.only}
        targets = [t for t in targets if t.stem[:2] in wanted]
    if not targets:
        print("  ! no executed notebooks found. Run `make notebooks` first.")
        return 1

    SHOTS.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=str(brave),
                args=["--no-sandbox", "--disable-dev-shm-usage",
                      "--force-color-profile=srgb"],
            )
            # width 1280 at 2x keeps Vietnamese diacritics legible when the
            # image is viewed scaled down in a GitHub markdown preview.
            page = browser.new_page(viewport={"width": 1280, "height": 1000},
                                    device_scale_factor=2)
            for nb_path in targets:
                stem = nb_path.stem
                print(f"  {stem}")
                if not is_executed(nb_path):
                    print("      skipped: no cell outputs — run `make notebooks`")
                    skipped += 1
                    continue

                html = render_html(nb_path, tmp)
                if html is None:
                    skipped += 1
                    continue

                page.goto(html.as_uri())
                page.wait_for_load_state("networkidle")
                dims = page.evaluate(
                    "() => ({w: document.documentElement.scrollWidth,"
                    " h: document.documentElement.scrollHeight})"
                )
                out = SHOTS / f"{stem}.png"
                page.screenshot(
                    path=str(out),
                    clip={"x": 0, "y": 0,
                          "width": dims["w"],
                          "height": min(dims["h"], args.max_height)},
                )
                truncated = "" if dims["h"] <= args.max_height else \
                    f"  (capped from {dims['h']}px)"
                print(f"      -> {out.relative_to(ROOT)}  "
                      f"{out.stat().st_size/1024:.0f} KB{truncated}")
                written += 1
            browser.close()

    print(f"\n  {written} notebook screenshot(s) written"
          + (f", {skipped} skipped" if skipped else "")
          + f" -> {SHOTS.relative_to(ROOT)}/")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())