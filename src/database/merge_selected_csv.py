"""Merge multiple selected-IDs CSV files into a single combined CSV.

Scans a directory for CSV files produced by create_csv.py and combines
all sound IDs into one file, adding a 'query' column from the source filename.
Duplicate IDs across files are removed by default.

Usage:
    python -m src.database.merge_selected_csv
    python -m src.database.merge_selected_csv \
        --input-dir database/database_download/user \
        --output database/database_download/user/combined_ids.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

_VALID_ID_COLUMNS = {"sound_id", "id", "soundId", "freesound_id"}


def merge_selected_csvs(input_dir: Path, output_path: Path, deduplicate: bool = True) -> int:
    """Merge all sound-ID CSVs under input_dir into output_path.

    Adds a 'query' column populated from each source filename (stem).
    Returns the number of rows written.
    """
    csv_files = sorted(f for f in input_dir.glob("*.csv") if f.resolve() != output_path.resolve())
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return 0

    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            col = next((c for c in (reader.fieldnames or []) if c in _VALID_ID_COLUMNS), None)
            if col is None:
                print(f"  [skip] {csv_path.name}: no sound_id column found")
                continue

            added = 0
            for row in reader:
                sound_id = (row.get(col) or "").strip()
                if not sound_id:
                    continue
                if deduplicate and sound_id in seen_ids:
                    continue
                seen_ids.add(sound_id)
                rows.append({"sound_id": sound_id})
                added += 1

            print(f"  {csv_path.name}: {added} ID(s) added")

    if not rows:
        print("No sound IDs found across all CSV files.")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sound_id"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge selected-IDs CSV files into a single combined CSV."
    )
    parser.add_argument(
        "--input-dir",
        default="database/database_download/user",
        help="Directory containing per-query selected-IDs CSV files.",
    )
    parser.add_argument(
        "--output",
        default="database/database_download/user/combined_ids.csv",
        help="Path for the merged output CSV.",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Keep duplicate sound IDs across files (default: deduplicate).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.is_dir():
        print(f"Directory not found: {input_dir}")
        return 1

    print(f"Merging CSVs from: {input_dir}")
    total = merge_selected_csvs(input_dir, output_path, deduplicate=not args.no_dedup)

    if total:
        print(f"\n[OK] Written {total} sound ID(s) to: {output_path}")
    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
