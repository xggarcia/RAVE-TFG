"""Private HTTP and CSV-parsing helpers for the Freesound downloader."""
from __future__ import annotations

import re
from typing import Iterable, Optional

import requests


FREESOUND_BASE_URL = "https://freesound.org/apiv2"

_LICENSE_MAP: dict[str, str] = {
	"cc0": "Creative Commons 0",
	"attribution": "Attribution",
	"noncommercial": "Attribution Noncommercial",
	"sampling": "Sampling+",
}


def _clean_text(value: Optional[str]) -> str:
	return (value or "").strip()


def _parse_int(value: Optional[str], default: int) -> int:
	txt = _clean_text(value)
	if not txt:
		return default
	try:
		parsed = int(txt)
		return parsed if parsed > 0 else default
	except ValueError:
		return default


def _parse_duration(value: Optional[str]) -> Optional[tuple[float, float]]:
	"""Parse duration from formats like '(0,3)', '0,3', '[0 3]' or blank."""
	txt = _clean_text(value)
	if not txt:
		return None
	nums = re.findall(r"[-+]?\d*\.?\d+", txt)
	if len(nums) < 2:
		return None
	low, high = float(nums[0]), float(nums[1])
	if low > high:
		low, high = high, low
	return (low, high)


def _parse_multi_values(value: Optional[str]) -> Optional[list[str]]:
	"""Parse values split by ',', ';' or '|' and remove blanks."""
	txt = _clean_text(value)
	if not txt:
		return None
	parts = [part.strip() for part in re.split(r"[,;|]", txt)]
	values = [part for part in parts if part]
	return values or None


def _normalize_feature_ext(value: Optional[str]) -> str:
	txt = _clean_text(value)
	if not txt:
		return ".json"
	return txt if txt.startswith(".") else f".{txt}"


def _headers(api_key: str) -> dict[str, str]:
	return {"Authorization": f"Token {api_key}"}


def _build_filter(
	duration: Optional[tuple[float, float]],
	tags: Optional[Iterable[str]],
	license_filter: Optional[str] = None,
) -> str:
	parts: list[str] = []
	if duration is not None:
		parts.append(f"duration:[{duration[0]} TO {duration[1]}]")
	if tags:
		parts.extend(f"tag:{tag}" for tag in tags if tag)
	if license_filter:
		keys = [k.strip().lower() for k in license_filter.split("|") if k.strip()]
		license_clauses = [f'license:"{_LICENSE_MAP[k]}"' for k in keys if k in _LICENSE_MAP]
		if len(license_clauses) == 1:
			parts.append(license_clauses[0])
		elif len(license_clauses) > 1:
			parts.append("(" + " OR ".join(license_clauses) + ")")
	return " ".join(parts)


def _request_json(url: str, api_key: str, params: Optional[dict] = None) -> dict:
	response = requests.get(url, headers=_headers(api_key), params=params, timeout=20)
	response.raise_for_status()
	return response.json()


def _download_file(url: str, destination) -> None:
	response = requests.get(url, timeout=40, stream=True)
	response.raise_for_status()
	with destination.open("wb") as file_obj:
		for chunk in response.iter_content(chunk_size=8192):
			if chunk:
				file_obj.write(chunk)


def _extract_preview_url(sound_payload: dict) -> Optional[str]:
	previews = sound_payload.get("previews") or {}
	return (
		previews.get("preview_hq_mp3")
		or previews.get("preview-hq-mp3")
		or previews.get("preview_lq_mp3")
		or previews.get("preview-lq-mp3")
		or sound_payload.get("preview_hq_mp3")
		or sound_payload.get("preview-hq-mp3")
		or sound_payload.get("preview_lq_mp3")
		or sound_payload.get("preview-lq-mp3")
	)


def _fetch_analysis(sound_id: int, api_key: str, descriptors: list[str]) -> dict:
	analysis_url = f"{FREESOUND_BASE_URL}/sounds/{sound_id}/analysis/"
	params = {
		"descriptors": ",".join(descriptors),
		"normalized": 1,
	}
	return _request_json(analysis_url, api_key, params=params)
