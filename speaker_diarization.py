#!/usr/bin/env python3
"""Standalone Pyannote speaker diarization script.

Setup:
    pip install pyannote.audio torch torchaudio matplotlib
    export HUGGINGFACE_TOKEN="hf_your_token"

Usage:
    python speaker_diarization.py /path/to/meeting.wav
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a"}


def format_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def load_diarization_pipeline(model_name: str, hf_token: str):
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'pyannote.audio'. Install with:\n"
            "pip install pyannote.audio torch torchaudio matplotlib"
        ) from exc

    try:
        return Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not load model '{model_name}'. Check your Hugging Face token and internet access."
        ) from exc


def extract_segments(diarization) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start = float(segment.start)
        end = float(segment.end)
        segments.append(
            {
                "speaker": str(speaker),
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(max(0.0, end - start), 3),
                "start_hhmmss": format_timestamp(start),
                "end_hhmmss": format_timestamp(end),
            }
        )
    return segments


def save_timeline(segments: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'matplotlib'. Install with:\n"
            "pip install matplotlib"
        ) from exc

    speakers = sorted({segment["speaker"] for segment in segments})
    speaker_to_row = {speaker: index for index, speaker in enumerate(speakers)}

    fig, ax = plt.subplots(figsize=(12, max(3, len(speakers) * 0.8)))
    for segment in segments:
        row = speaker_to_row[segment["speaker"]]
        ax.broken_barh([(segment["start"], segment["duration"])], (row - 0.4, 0.8))

    max_end = max((segment["end"] for segment in segments), default=0.0)
    ax.set_xlim(0.0, max_end + 0.5)
    ax.set_ylim(-1, len(speakers))
    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels(speakers)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Speaker Timeline")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run speaker diarization (who spoke when) with Pyannote.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio_file", help="Path to an audio file (.mp3, .wav, .m4a)")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Output JSON file path. Defaults to <audio_name>_diarization.json",
    )
    parser.add_argument(
        "--timeline-png",
        default=None,
        help="Output timeline PNG path. Defaults to <audio_name>_timeline.png",
    )
    parser.add_argument(
        "--model",
        default="pyannote/speaker-diarization-3.1",
        help="Pyannote diarization model name",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face access token. If omitted, reads HUGGINGFACE_TOKEN or HF_TOKEN.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    audio_path = Path(args.audio_file).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if audio_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"Unsupported audio format '{audio_path.suffix}'. Supported formats: {supported}")

    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else audio_path.with_name(
        f"{audio_path.stem}_diarization.json"
    )
    timeline_png = (
        Path(args.timeline_png).expanduser().resolve()
        if args.timeline_png
        else audio_path.with_name(f"{audio_path.stem}_timeline.png")
    )

    hf_token = args.hf_token or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "A Hugging Face token is required for Pyannote models.\n"
            "Set HUGGINGFACE_TOKEN (or HF_TOKEN), or pass --hf-token."
        )

    print(f"[INFO] Loading diarization model: {args.model}")
    pipeline = load_diarization_pipeline(args.model, hf_token)

    print(f"[INFO] Processing audio: {audio_path}")
    try:
        diarization = pipeline(str(audio_path))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to diarize audio. Ensure the file is valid and readable: {audio_path}") from exc

    segments = extract_segments(diarization)
    if not segments:
        raise RuntimeError("No speaker segments were detected in this file.")

    speakers = sorted({segment["speaker"] for segment in segments})
    total_duration = round(max(segment["end"] for segment in segments), 3)

    print("\n=== Speaker Segments ===")
    for segment in segments:
        print(f"{segment['start_hhmmss']} - {segment['end_hhmmss']}  {segment['speaker']}")

    print("\n=== Summary ===")
    print(f"Speakers detected: {len(speakers)} ({', '.join(speakers)})")
    print(f"Segments: {len(segments)}")
    print(f"Audio duration (approx): {format_timestamp(total_duration)}")

    payload = {
        "input_file": str(audio_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "speaker_count": len(speakers),
        "speakers": speakers,
        "segment_count": len(segments),
        "total_duration_seconds": total_duration,
        "segments": segments,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    print(f"[INFO] JSON written to: {output_json}")

    timeline_png.parent.mkdir(parents=True, exist_ok=True)
    save_timeline(segments, timeline_png)
    print(f"[INFO] Timeline written to: {timeline_png}")

    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
