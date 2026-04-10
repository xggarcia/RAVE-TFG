"""End-to-end Freesound database creation workflow.

Flow per query row in input CSV:
1) Download preview candidates (from first_download_freesound.py logic)
2) Play each preview and ask user keep/skip (create_csv.py logic)
3) Save selected IDs CSV in database/database_download/user/
4) Download selected sounds into input_data/user_data/<queryText>/
5) Normalize volume of downloaded sounds to a consistent RMS level
6) Remove temporary preview folder (for example descSounds/<queryText>/)
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import replace
from pathlib import Path

try:
	from src.database_creation.create_csv import gather_candidates, run_selection, write_selected_ids_csv
	from src.database_creation.download_csv import _load_dotenv, download_sound_by_id
	from src.database_creation.first_download_freesound import (
		download_sounds_freesound,
		read_jobs_from_csv,
	)
	from src.database_creation.normalize_volume import normalize_directory
except ModuleNotFoundError:
	# Support direct execution: `python src/database_creation/create_database.py ...`
	from create_csv import gather_candidates, run_selection, write_selected_ids_csv
	from download_csv import _load_dotenv, download_sound_by_id
	from first_download_freesound import download_sounds_freesound, read_jobs_from_csv
	from normalize_volume import normalize_directory


def _build_unique_csv_path(base_dir: Path, query_text: str) -> Path:
	"""Return a non-colliding CSV path: query.csv, query_01.csv, query_02.csv, ..."""
	base_dir.mkdir(parents=True, exist_ok=True)
	base = query_text.strip() or "selection"

	candidate = base_dir / f"{base}.csv"
	if not candidate.exists():
		return candidate

	index = 1
	while True:
		candidate = base_dir / f"{base}_{index:02d}.csv"
		if not candidate.exists():
			return candidate
		index += 1


def _build_unique_query_dir(base_dir: Path, query_text: str) -> Path:
	"""Return a non-colliding folder path: query, query_01, query_02, ..."""
	base_dir.mkdir(parents=True, exist_ok=True)
	base = query_text.strip() or "selection"

	candidate = base_dir / base
	if not candidate.exists():
		return candidate

	index = 1
	while True:
		candidate = base_dir / f"{base}_{index:02d}"
		if not candidate.exists():
			return candidate
		index += 1


def _download_selected_ids(selected_ids: list[str], api_key: str, output_dir: Path) -> tuple[int, int]:
	"""Download selected IDs and return (success_count, total_count)."""
	output_dir.mkdir(parents=True, exist_ok=True)
	total = len(selected_ids)
	success = 0

	for idx, sound_id in enumerate(selected_ids, start=1):
		print(f"  [{idx}/{total}] Downloading selected sound {sound_id}")
		if download_sound_by_id(sound_id, api_key, output_dir, skip_existing=True):
			success += 1

	return success, total


def _cleanup_temp_query_folder(temp_root: Path, query_text: str) -> None:
	query_dir = temp_root / query_text
	if query_dir.exists() and query_dir.is_dir():
		shutil.rmtree(query_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Create final user dataset from Freesound workflow.")
	parser.add_argument(
		"--csv",
		default="",
		help="Input CSV with Freesound query jobs. If blank, you'll be prompted.",
	)
	parser.add_argument(
		"--selected-csv-dir",
		default="database/database_download/user",
		help="Directory where selected IDs CSV files are saved.",
	)
	parser.add_argument(
		"--final-root",
		default="input_data/user_data",
		help="Root folder where final audio is downloaded per queryText.",
	)
	return parser.parse_args()


def run_create_database_workflow(
	jobs_csv_path: Path,
	selected_csv_dir: Path,
	final_root: Path,
	temp_preview_root: Path | None = None,
) -> int:
	"""Run full dataset-creation workflow and return process status code."""
	try:
		jobs = read_jobs_from_csv(jobs_csv_path)
	except Exception as exc:
		print(f"Could not read jobs CSV: {exc}")
		return 1

	if not jobs:
		print("No valid query jobs found in input CSV.")
		return 0

	if temp_preview_root is not None:
		jobs = [replace(job, output_dir=temp_preview_root) for job in jobs]

	global_selected = 0
	global_downloaded = 0
	used_temp_roots: set[Path] = set()

	for job_index, job in enumerate(jobs, start=1):
		used_temp_roots.add(job.output_dir)
		print("\n" + "=" * 88)
		print(f"[{job_index}/{len(jobs)}] Query workflow: {job.query_text}")
		print("=" * 88)

		# 1) Download preview candidates for this query.
		saved_previews = download_sounds_freesound(job)
		if saved_previews <= 0:
			print(f"No preview candidates saved for query '{job.query_text}'. Skipping.")
			continue

		# 2) Ask user what to preserve for this query only.
		all_candidates = gather_candidates(job.output_dir)
		query_candidates = [c for c in all_candidates if c.query_text == job.query_text]
		if not query_candidates:
			print(f"No playable candidates found for query '{job.query_text}'.")
			_cleanup_temp_query_folder(job.output_dir, job.query_text)
			continue

		print(f"Reviewing {len(query_candidates)} preview candidate(s) for '{job.query_text}'")
		selected_ids = run_selection(query_candidates)
		global_selected += len(selected_ids)

		# 3) Store selected IDs CSV with unique filename in database/database_download/user.
		selected_csv_path = _build_unique_csv_path(selected_csv_dir, job.query_text)
		write_selected_ids_csv(selected_ids, selected_csv_path)
		print(f"Saved selected IDs CSV: {selected_csv_path}")

		# 4) Download selected sounds into input_data/user_data/<queryText>/.
		if selected_ids:
			final_query_dir = _build_unique_query_dir(final_root, job.query_text)
			success, total = _download_selected_ids(selected_ids, job.api_key, final_query_dir)
			global_downloaded += success
			print(f"Downloaded {success}/{total} selected sound(s) into {final_query_dir}")

			# 5) Normalize volume of all downloaded sounds to a consistent RMS level.
			print(f"Normalizing volume in {final_query_dir} ...")
			ok, n = normalize_directory(final_query_dir)
			print(f"Normalized {ok}/{n} file(s).")
		else:
			print("No sounds selected for this query. Final download skipped.")

		# 6) Remove temporary mid-download previews (descSounds/<queryText> style).
		_cleanup_temp_query_folder(job.output_dir, job.query_text)
		print(f"Removed temporary preview folder for query '{job.query_text}'")

	print("\n" + "=" * 88)
	for temp_root in sorted(used_temp_roots, key=lambda p: str(p)):
		if temp_root.exists() and temp_root.is_dir():
			shutil.rmtree(temp_root, ignore_errors=True)
			print(f"Removed temporary folder: {temp_root}")

	print("\n" + "=" * 88)
	print("Workflow completed")
	print(f"Total selected IDs: {global_selected}")
	print(f"Total final downloads: {global_downloaded}")
	print(f"Selected CSV folder: {selected_csv_dir}")
	print(f"Final audio root: {final_root}")
	return 0


def main() -> int:
	_load_dotenv()
	args = parse_args()

	csv_value = args.csv.strip()
	if not csv_value:
		csv_value = input("Path to query CSV file: ").strip()
	if not csv_value:
		print("No CSV path provided.")
		return 1

	jobs_csv_path = Path(csv_value)
	selected_csv_dir = Path(args.selected_csv_dir)
	final_root = Path(args.final_root)

	return run_create_database_workflow(
		jobs_csv_path=jobs_csv_path,
		selected_csv_dir=selected_csv_dir,
		final_root=final_root,
		temp_preview_root=None,
	)


if __name__ == "__main__":
	raise SystemExit(main())
