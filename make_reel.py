"""
Instagram Reel Generator — CLI
Baut ein Reel aus einem Ordner heraus. Nutzt reel_engine.py als Basis.

Ordnerstruktur:
  reels/mein_reel/
    titel.png
    fotos/foto1.jpg ...
    audio/track.mp3     (optional)

Aufruf:
  python make_reel.py reels/mein_reel
  python make_reel.py reels/mein_reel --duration 25 --beats-per-cut 3
"""

import argparse
import glob
import os
import sys

from reel_engine import render_reel

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG")
AUDIO_EXTS = ("*.mp3", "*.wav", "*.aac", "*.m4a", "*.ogg")


def glob_files(directory, extensions):
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description="Instagram Reel Generator (CLI)")
    parser.add_argument("reel_dir",          help="Pfad zum Reel-Ordner")
    parser.add_argument("--duration",        type=float, default=20.0)
    parser.add_argument("--crossfade",       type=float, default=0.35)
    parser.add_argument("--beats-per-cut",   type=int,   default=2)
    parser.add_argument("--titel-duration",  type=float, default=3.5)
    args = parser.parse_args()

    reel_dir = args.reel_dir.rstrip("/")
    if not os.path.isdir(reel_dir):
        sys.exit(f"Fehler: Ordner nicht gefunden: {reel_dir}")

    titel_path = os.path.join(reel_dir, "titel.png")
    if not os.path.isfile(titel_path):
        sys.exit(f"Fehler: Titelkarte nicht gefunden: {titel_path}")

    fotos  = glob_files(os.path.join(reel_dir, "fotos"),  IMAGE_EXTS)
    audios = glob_files(os.path.join(reel_dir, "audio"),  AUDIO_EXTS)

    if not fotos:
        sys.exit(f"Fehler: Keine Fotos in {reel_dir}/fotos/ gefunden.")

    reel_name   = os.path.basename(reel_dir)
    output_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    output_path = os.path.join(output_dir, f"{reel_name}.mp4")

    print(f"\n{'='*50}")
    print(f"  Reel  : {reel_name}")
    print(f"  Fotos : {len(fotos)}")
    print(f"  Audio : {os.path.basename(audios[0]) if audios else '—'}")
    print(f"{'='*50}\n")

    def progress(step, total, msg):
        print(f"  [{step}/{total}] {msg}")

    render_reel(
        titel_source    = titel_path,
        foto_sources    = fotos,
        output_path     = output_path,
        audio_path      = audios[0] if audios else None,
        total_duration  = args.duration,
        titel_duration  = args.titel_duration,
        crossfade       = args.crossfade,
        beats_per_cut   = args.beats_per_cut,
        progress_callback=progress,
    )

    print(f"\n✓ Gespeichert: {output_path}")


if __name__ == "__main__":
    main()
