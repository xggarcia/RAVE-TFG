"""CSV-driven Freesound downloader.

This module reads one or more download jobs from a CSV file and downloads:
1) Sound preview files (.mp3)
2) Descriptor JSON files from Freesound analysis endpoint

Each CSV row can define these columns (all optional except queryText/API key):
	queryText, API_Key, outputDir, topNResults, duration, tag, featureExt, descriptors

Notes:
- `tag` supports multiple tags in one field (comma, semicolon or pipe-separated).
- Any column can be blank. Defaults are applied where possible.
- If `API_Key` is blank, environment variable FREESOUND_API_KEY is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from src.database._freesound_api import (
	FREESOUND_BASE_URL,
	_build_filter,
	_clean_text,
	_download_file,
	_extract_preview_url,
	_fetch_analysis,
	_normalize_feature_ext,
	_parse_duration,
	_parse_int,
	_parse_multi_values,
	_request_json,
)

DEFAULT_DESCRIPTORS = [
	"spectral_centroid",
	"dissonance",
	"spectral_rolloff",
	"log_attack_time",
	"inharmonicity",
	"tristimulus",
	"mfcc",
]


def _load_dotenv(dotenv_path: Path = Path(".env")) -> None:
	"""Load KEY=VALUE pairs from a .env file into os.environ if missing."""
	if not dotenv_path.exists() or not dotenv_path.is_file():
		return

	for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue

		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip().strip('"').strip("'")
		if key and key not in os.environ:
			os.environ[key] = value


@dataclass
class DownloadJob:
	query_text: str
	api_key: str
	output_dir: Path
	top_n_results: int = 5
	duration: Optional[tuple[float, float]] = None
	tags: Optional[list[str]] = None
	feature_ext: str = ".json"
	descriptors: Optional[list[str]] = None
	license_filter: Optional[str] = None



def download_sounds_freesound(job: DownloadJob) -> int:
	"""Download previews and descriptor JSON files for one search job."""
	if not job.query_text:
		print("Skipping row: queryText is empty.")
		return 0

	if not job.api_key:
		print(f"Skipping '{job.query_text}': missing API key.")
		return 0

	descriptors = job.descriptors or DEFAULT_DESCRIPTORS
	filter_expr = _build_filter(job.duration, job.tags, job.license_filter)

	query_output_dir = job.output_dir / job.query_text
	if query_output_dir.exists():
		shutil.rmtree(query_output_dir)
	query_output_dir.mkdir(parents=True, exist_ok=True)

	search_url = f"{FREESOUND_BASE_URL}/search/text/"
	search_params = {
		"query": job.query_text,
		"sort": "score",
		"fields": "id,name,previews,url",
		"page_size": min(max(job.top_n_results, 1), 150),
	}
	if filter_expr:
		search_params["filter"] = filter_expr

	try:
		search_data = _request_json(search_url, job.api_key, params=search_params)
	except requests.RequestException as exc:
		print(f"Search failed for '{job.query_text}': {exc}")
		return 0

	results = search_data.get("results", [])
	if not results:
		print(f"No results found for '{job.query_text}' with the selected filters.")
		return 0

	max_candidates = min(job.top_n_results, len(results))
	downloaded_sounds: list[tuple[str, str]] = []
	downloaded_count = 0

	for index, sound in enumerate(results[:max_candidates], start=1):
		sound_id = sound.get("id")
		if sound_id is None:
			continue

		preview_url = _extract_preview_url(sound)
		if not preview_url:
			print(f"[{job.query_text}] candidate {index}: skipping id={sound_id} (no preview)")
			continue

		sound_dir = query_output_dir / str(sound_id)
		if sound_dir.exists():
			shutil.rmtree(sound_dir)
		sound_dir.mkdir(parents=True, exist_ok=True)

		mp3_name = preview_url.split("/")[-1]
		mp3_path = sound_dir / mp3_name
		feature_path = mp3_path.with_suffix(job.feature_ext)

		try:
			_download_file(preview_url, mp3_path)
			analysis = _fetch_analysis(int(sound_id), job.api_key, descriptors)

			features = {}
			for desc in descriptors:
				# Keep notebook-compatible structure: each descriptor maps to a one-element list.
				value = analysis.get(desc, 0.0)
				if isinstance(value, dict) and "mean" in value:
					value = value["mean"]
				features[desc] = [value]

			with feature_path.open("w", encoding="utf-8") as file_obj:
				json.dump(features, file_obj)

			downloaded_sounds.append((str(sound_id), str(sound.get("url", ""))))
			downloaded_count += 1
			print(f"[{job.query_text}] saved {downloaded_count}/{max_candidates}: id={sound_id}")

		except requests.RequestException as exc:
			print(f"[{job.query_text}] failed id={sound_id}: {exc}")
			if sound_dir.exists():
				shutil.rmtree(sound_dir)
		except OSError as exc:
			print(f"[{job.query_text}] file error for id={sound_id}: {exc}")
			if sound_dir.exists():
				shutil.rmtree(sound_dir)

	sound_list_path = query_output_dir / f"{job.query_text}_SoundList.txt"
	with sound_list_path.open("w", encoding="utf-8") as file_obj:
		for sound_id, sound_url in downloaded_sounds:
			file_obj.write(f"{sound_id}\t{sound_url}\n")

	print(
		f"[{job.query_text}] done: saved {downloaded_count} sound(s) in {query_output_dir}"
	)
	return downloaded_count


def read_jobs_from_csv(csv_path: Path) -> list[DownloadJob]:
	"""Build a list of DownloadJob items from a CSV file.

	Supported columns (all optional except queryText/API key):
	- queryText
	- API_Key
	- outputDir
	- topNResults
	- duration
	- tag
	- featureExt
	- descriptors
	"""
	if not csv_path.exists():
		raise FileNotFoundError(f"CSV file not found: {csv_path}")

	jobs: list[DownloadJob] = []
	with csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
		reader = csv.DictReader(file_obj)
		for row_index, row in enumerate(reader, start=2):
			query_text = _clean_text(row.get("queryText"))
			api_key = _clean_text(row.get("API_Key")) or os.getenv("FREESOUND_API_KEY", "")
			output_dir = Path(_clean_text(row.get("outputDir")) or "descSounds")
			top_n = _parse_int(row.get("topNResults"), default=5)
			duration = _parse_duration(row.get("duration"))
			tags = _parse_multi_values(row.get("tag"))
			feature_ext = _normalize_feature_ext(row.get("featureExt"))
			descriptors = _parse_multi_values(row.get("descriptors"))
			license_filter = _clean_text(row.get("license")) or None

			if not query_text:
				print(f"Row {row_index}: empty queryText, skipping.")
				continue

			jobs.append(
				DownloadJob(
					query_text=query_text,
					api_key=api_key,
					output_dir=output_dir,
					top_n_results=top_n,
					duration=duration,
					tags=tags,
					feature_ext=feature_ext,
					descriptors=descriptors,
					license_filter=license_filter,
				)
			)

	return jobs


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Download Freesound data from CSV jobs.")
	parser.add_argument(
		"--csv",
		required=True,
		help="Path to CSV file with download job rows.",
	)
	return parser.parse_args()


def main() -> int:
	_load_dotenv()

	args = parse_args()
	csv_path = Path(args.csv)

	try:
		jobs = read_jobs_from_csv(csv_path)
	except Exception as exc:
		print(f"Failed to read CSV: {exc}")
		return 1

	if not jobs:
		print("No valid jobs found in CSV.")
		return 0

	total_saved = 0
	for job in jobs:
		total_saved += download_sounds_freesound(job)

	print(f"All jobs finished. Total downloaded sounds: {total_saved}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
