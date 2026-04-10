"""Interactive selector for downloaded Freesound previews.

The script scans a directory of previously downloaded sounds (for example `descSounds/`),
plays each audio preview, and asks whether to keep it.

When a sound is accepted, its Freesound ID (folder name) is written to a CSV file.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

import sounddevice as sd
import soundfile as sf


@dataclass
class SoundCandidate:
	query_text: str
	sound_id: str
	audio_path: Path


def _find_audio_file(sound_dir: Path) -> Path | None:
	for pattern in ("*.mp3", "*.wav", "*.ogg", "*.flac"):
		matches = sorted(sound_dir.glob(pattern))
		if matches:
			return matches[0]
	return None


def gather_candidates(download_root: Path) -> list[SoundCandidate]:
	"""Collect all audio candidates from folders like root/query/sound_id/audiofile."""
	if not download_root.exists() or not download_root.is_dir():
		return []

	candidates: list[SoundCandidate] = []
	for query_dir in sorted(path for path in download_root.iterdir() if path.is_dir()):
		for sound_dir in sorted(path for path in query_dir.iterdir() if path.is_dir()):
			audio_path = _find_audio_file(sound_dir)
			if audio_path is None:
				continue
			candidates.append(
				SoundCandidate(
					query_text=query_dir.name,
					sound_id=sound_dir.name,
					audio_path=audio_path,
				)
			)
	return candidates


def _play_audio_internal(audio_path: Path) -> bool:
	"""Start audio playback non-blocking. Returns True when successful."""
	try:
		data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
		sd.play(data, sample_rate)
		return True
	except Exception:
		return False


def _open_audio_default_player(audio_path: Path) -> None:
	"""Fallback: open audio with default OS player."""
	if platform.system() == "Windows":
		os.startfile(str(audio_path))  # type: ignore[attr-defined]
	elif platform.system() == "Darwin":
		subprocess.run(["open", str(audio_path)], check=False)
	else:
		subprocess.run(["xdg-open", str(audio_path)], check=False)


def play_audio(audio_path: Path) -> None:
	"""Play one audio file; fallback to default player if local playback fails."""
	played = _play_audio_internal(audio_path)
	if not played:
		print("Could not play via sounddevice/soundfile; opening default player...")
		_open_audio_default_player(audio_path)


def ask_user_action() -> str:
	"""Prompt user for action and return normalized command."""
	while True:
		answer = input("Keep this sound? [y]es / [n]o / [r]eplay / [q]uit: ").strip().lower()
		if answer in {"y", "n", "r", "q"}:
			return answer
		print("Please type y, n, r or q.")


def run_selection(candidates: list[SoundCandidate]) -> list[str]:
	"""Interactive review loop returning selected sound IDs."""
	selected_ids: list[str] = []
	total = len(candidates)

	for idx, candidate in enumerate(candidates, start=1):
		print("\n" + "=" * 72)
		print(f"[{idx}/{total}] Query: {candidate.query_text} | Sound ID: {candidate.sound_id}")
		print(f"File: {candidate.audio_path}")

		play_audio(candidate.audio_path)

		while True:
			action = ask_user_action()
			sd.stop()
			if action == "r":
				play_audio(candidate.audio_path)
				continue
			if action == "y":
				selected_ids.append(candidate.sound_id)
				print(f"Added sound id {candidate.sound_id}")
			elif action == "n":
				print("Skipped")
			elif action == "q":
				print("Stopping early and saving current selection...")
				return selected_ids
			break

	return selected_ids


def write_selected_ids_csv(selected_ids: list[str], output_csv: Path) -> None:
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	with output_csv.open("w", encoding="utf-8", newline="") as file_obj:
		writer = csv.writer(file_obj)
		writer.writerow(["sound_id"])
		for sound_id in selected_ids:
			writer.writerow([sound_id])


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Play downloaded sounds one by one and save selected IDs to CSV."
	)
	parser.add_argument(
		"--input-dir",
		default="descSounds",
		help="Folder containing downloaded sounds (default: descSounds).",
	)
	parser.add_argument(
		"--output-csv",
		default="database/database_creation/selected_sound_ids.csv",
		help="CSV file where accepted sound IDs will be saved.",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	input_dir = Path(args.input_dir)
	output_csv = Path(args.output_csv)

	candidates = gather_candidates(input_dir)
	if not candidates:
		print(f"No audio candidates found in {input_dir}")
		return 0

	print(f"Found {len(candidates)} candidate sounds.")
	selected_ids = run_selection(candidates)
	write_selected_ids_csv(selected_ids, output_csv)

	print("\n" + "=" * 72)
	print(f"Saved {len(selected_ids)} selected sound ID(s) to: {output_csv}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
