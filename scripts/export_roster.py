"""Write data/source_roster.csv and data/source_roster.md from src.sources only."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sources import SCHEMES  # noqa: E402

OUT_DIR = ROOT / "data"
CSV_PATH = OUT_DIR / "source_roster.csv"
MD_PATH = OUT_DIR / "source_roster.md"
FIELDS = ("scheme_class", "scheme_name", "url")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(SCHEMES)

    lines = [
        "# Source roster",
        "",
        "Generated from `src/sources.py`. Do not edit by hand.",
        "",
        "| scheme_class | scheme_name | url |",
        "| :--- | :--- | :--- |",
    ]
    for row in SCHEMES:
        lines.append(
            f"| {row['scheme_class']} | {row['scheme_name']} | {row['url']} |"
        )
    lines.append("")
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(SCHEMES)} rows to {CSV_PATH}")
    print(f"Wrote {MD_PATH}")


if __name__ == "__main__":
    main()
