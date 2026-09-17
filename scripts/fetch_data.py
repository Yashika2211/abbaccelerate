#!/usr/bin/env python3
"""Download the two public datasets — PROJECT_BRIEF.md §6.

Run once with a network:

    python scripts/fetch_data.py

Without a network, everything still works: the loaders fall back to the vendored
slices in ``data/samples`` (committed) and then to a synthetic generator. This
script exists to make the full datasets reproducible, not to make them required.

Sources:
  * NASA C-MAPSS turbofan degradation — Saxena & Goebel (2008), NASA Ames.
  * UCI AI4I 2020 predictive maintenance — Matzka (2020), DOI 10.24432/C5HS5C.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

CMAPSS_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
AI4I_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)

CMAPSS_FILES = [f"{kind}_FD00{i}.txt" for kind in ("train", "test", "RUL") for i in range(1, 5)]


def _download(url: str, timeout: int = 300) -> bytes:
    print(f"  fetching {url[:78]}...")
    request = urllib.request.Request(url, headers={"User-Agent": "kairos/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = response.read()
    print(f"  received {len(payload):,} bytes (sha256 {hashlib.sha256(payload).hexdigest()[:16]})")
    return payload


def fetch_cmapss(force: bool = False) -> bool:
    target = RAW / "cmapss"
    if (target / "train_FD001.txt").exists() and not force:
        print("C-MAPSS already present; skipping (use --force to refetch)")
        return True
    target.mkdir(parents=True, exist_ok=True)
    try:
        outer = zipfile.ZipFile(io.BytesIO(_download(CMAPSS_URL)))
        # The PHM archive nests the real CMAPSSData.zip one level down.
        inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
        inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
        written = 0
        for name in inner.namelist():
            base = Path(name).name
            if base in CMAPSS_FILES or base == "readme.txt":
                (target / base).write_bytes(inner.read(name))
                written += 1
        print(f"  wrote {written} files to {target}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {exc}", file=sys.stderr)
        print("  falling back to data/samples (vendored) at load time", file=sys.stderr)
        return False


def fetch_ai4i(force: bool = False) -> bool:
    target = RAW / "ai4i"
    if (target / "ai4i2020.csv").exists() and not force:
        print("AI4I already present; skipping (use --force to refetch)")
        return True
    target.mkdir(parents=True, exist_ok=True)
    try:
        archive = zipfile.ZipFile(io.BytesIO(_download(AI4I_URL)))
        name = next(n for n in archive.namelist() if n.endswith(".csv"))
        (target / "ai4i2020.csv").write_bytes(archive.read(name))
        print(f"  wrote {target / 'ai4i2020.csv'}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {exc}", file=sys.stderr)
        print("  falling back to data/samples (vendored) at load time", file=sys.stderr)
        return False


def verify() -> int:
    """Load both datasets through the real loaders and report what came back."""
    sys.path.insert(0, str(ROOT / "backend"))
    from kairos.data.loaders import load_ai4i, load_cmapss  # noqa: PLC0415

    problems = 0
    for name, loader in (("C-MAPSS", load_cmapss), ("AI4I", load_ai4i)):
        try:
            dataset = loader()
            summary = dataset.summary()
            print(
                f"  {name:9s} source={summary['source']:9s} rows={summary['rows']:>6,} "
                f"target={summary['target']!r} task={summary['task_type']}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:9s} FAILED: {exc}", file=sys.stderr)
            problems += 1
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="refetch even if present")
    parser.add_argument("--verify-only", action="store_true", help="skip download, just load")
    args = parser.parse_args()

    if not args.verify_only:
        print("== C-MAPSS ==")
        fetch_cmapss(force=args.force)
        print("== AI4I ==")
        fetch_ai4i(force=args.force)

    print("== verify ==")
    problems = verify()
    if problems:
        print(f"\n{problems} dataset(s) failed to load.", file=sys.stderr)
        return 1
    print("\nAll datasets load.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
