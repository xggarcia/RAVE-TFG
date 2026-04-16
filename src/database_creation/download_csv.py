"""Download Freesound audio from a CSV of selected sound IDs.

Expected CSV format (header required):
    sound_id
    12345
    67890

The script tries to download original audio first (`/sounds/{id}/download/`).
If original download fails, it falls back to preview MP3 when available.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Optional

import requests

try:
	from src.database_creation.freesound_auth import get_access_token, load_oauth2_credentials
except ModuleNotFoundError:
	from freesound_auth import get_access_token, load_oauth2_credentials

FREESOUND_BASE_URL = "https://freesound.org/apiv2"


def _load_dotenv(dotenv_path: Path = Path(".env")) -> None:
	"""Load KEY=VALUE pairs from .env into os.environ (without overriding existing vars)."""
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


def _headers(api_key: str) -> dict[str, str]:
	return {"Authorization": f"Token {api_key}"}


def _safe_filename(name: str) -> str:
	# Keep simple and cross-platform-safe.
	clean = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip()
	return clean or "sound"


def _extract_extension_from_type(type_value: Optional[str]) -> str:
	if not type_value:
		return ""
	ext = type_value.strip().lower()
	if not ext:
		return ""
	if not ext.startswith("."):
		ext = f".{ext}"
	return ext


def _read_sound_ids(csv_path: Path) -> list[str]:
	if not csv_path.exists():
		raise FileNotFoundError(f"CSV file not found: {csv_path}")

	ids: list[str] = []
	with csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
		reader = csv.DictReader(file_obj)
		if not reader.fieldnames:
			raise ValueError("CSV has no header.")

		valid_columns = {"sound_id", "id", "soundId", "freesound_id"}
		column = next((name for name in reader.fieldnames if name in valid_columns), None)
		if column is None:
			raise ValueError(
				"CSV must contain one of these headers: sound_id, id, soundId, freesound_id"
			)

		for row in reader:
			raw_id = (row.get(column) or "").strip()
			if raw_id:
				ids.append(raw_id)

	return ids


def _get_sound_info(sound_id: str, api_key: str) -> dict:
	url = f"{FREESOUND_BASE_URL}/sounds/{sound_id}/"
	params = {
		"fields": "id,name,type,previews,url",
	}
	response = requests.get(url, headers=_headers(api_key), params=params, timeout=20)
	response.raise_for_status()
	return response.json()


def _stream_download(url: str, headers: dict[str, str], output_path: Path) -> None:
	with requests.get(url, headers=headers, stream=True, timeout=60) as response:
		response.raise_for_status()
		with output_path.open("wb") as file_obj:
			for chunk in response.iter_content(chunk_size=8192):
				if chunk:
					file_obj.write(chunk)


def _preview_url_from_payload(payload: dict) -> Optional[str]:
	previews = payload.get("previews") or {}
	return (
		previews.get("preview_hq_mp3")
		or previews.get("preview-hq-mp3")
		or previews.get("preview_lq_mp3")
		or previews.get("preview-lq-mp3")
	)


def download_sound_by_id(sound_id: str, api_key: str, output_dir: Path, skip_existing: bool) -> bool:
	"""Download one sound by id.

	Tries the original file via OAuth2 Bearer token (full quality, any duration).
	Falls back to HQ preview MP3 if OAuth2 is not configured or download fails.
	"""
	try:
		sound_info = _get_sound_info(sound_id, api_key)
	except requests.RequestException as exc:
		print(f"[{sound_id}] failed to fetch sound info: {exc}")
		return False

	name = _safe_filename(str(sound_info.get("name", f"sound_{sound_id}")))
	ext = _extract_extension_from_type(sound_info.get("type"))
	if ext and not name.lower().endswith(ext):
		name = f"{name}{ext}"

	original_path = output_dir / name
	preview_path = output_dir / f"{Path(name).stem}_preview.mp3"

	if skip_existing and original_path.exists():
		print(f"[{sound_id}] already exists, skipping: {original_path.name}")
		return True

	# 1) Try original download via OAuth2 (requires FREESOUND_CLIENT_SECRET in .env).
	credentials = load_oauth2_credentials()
	if credentials:
		original_url = f"{FREESOUND_BASE_URL}/sounds/{sound_id}/download/"
		try:
			access_token = get_access_token(*credentials)
			_stream_download(original_url, {"Authorization": f"Bearer {access_token}"}, original_path)
			print(f"[{sound_id}] downloaded original: {original_path.name}")
			return True
		except requests.RequestException as exc:
			print(f"[{sound_id}] original download failed: {exc}")
			if original_path.exists():
				original_path.unlink(missing_ok=True)
	else:
		print(f"[{sound_id}] OAuth2 not configured (no FREESOUND_CLIENT_SECRET). Falling back to preview.")

	# 2) Fallback: HQ preview download (30s MP3, no OAuth2 required).
	preview_url = _preview_url_from_payload(sound_info)
	if not preview_url:
		print(f"[{sound_id}] no preview available.")
		return False

	if skip_existing and preview_path.exists():
		print(f"[{sound_id}] preview already exists, skipping: {preview_path.name}")
		return True

	try:
		_stream_download(preview_url, {}, preview_path)
		print(f"[{sound_id}] downloaded preview: {preview_path.name}")
		return True
	except requests.RequestException as exc:
		print(f"[{sound_id}] preview download failed: {exc}")
		if preview_path.exists():
			preview_path.unlink(missing_ok=True)
		return False


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Download Freesound sounds from IDs CSV.")
	parser.add_argument(
		"--csv",
		required=True,
		help="Path to CSV file containing sound IDs.",
	)
	parser.add_argument(
		"--output-dir",
		required=True,
		help="Folder where sounds will be downloaded.",
	)
	parser.add_argument(
		"--api-key",
		default="",
		help="Freesound API key. If blank, uses FREESOUND_API_KEY env var.",
	)
	parser.add_argument(
		"--skip-existing",
		action="store_true",
		help="Skip files that already exist in output folder.",
	)
	return parser.parse_args()


def main() -> int:
	_load_dotenv()
	args = parse_args()

	csv_path = Path(args.csv)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	api_key = (args.api_key or os.getenv("FREESOUND_API_KEY", "")).strip()
	if not api_key:
		print("Missing API key. Set --api-key or FREESOUND_API_KEY in .env")
		return 1

	try:
		sound_ids = _read_sound_ids(csv_path)
	except Exception as exc:
		print(f"Could not read csv: {exc}")
		return 1

	if not sound_ids:
		print("No sound IDs found in CSV.")
		return 0

	print(f"Found {len(sound_ids)} sound id(s). Downloading into: {output_dir}")
	success = 0
	for idx, sound_id in enumerate(sound_ids, start=1):
		print(f"\n[{idx}/{len(sound_ids)}] Processing {sound_id}")
		if download_sound_by_id(sound_id, api_key, output_dir, args.skip_existing):
			success += 1

	print("\n" + "=" * 72)
	print(f"Completed. Successful downloads: {success}/{len(sound_ids)}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
