"""
Reel Engine — Kernlogik für Video-Generierung.
Nutzt moviepy 2.x API.
"""

import os
import random
import numpy as np
import librosa
from PIL import Image
from moviepy import (
    AudioFileClip,
    VideoClip,
    concatenate_videoclips,
)
from moviepy.video.fx import FadeIn, FadeOut
from moviepy.audio.fx import AudioFadeOut

W, H   = 1080, 1920
FPS    = 30
MAX_DURATION   = 30.0
AUDIO_FADE_OUT = 2.0

KB_ZOOM_MIN_FOTO  = 1.05
KB_ZOOM_MAX_FOTO  = 1.18
KB_ZOOM_MIN_TITEL = 1.02
KB_ZOOM_MAX_TITEL = 1.08


# ── Bild-Hilfsfunktionen ──────────────────────────────────────────────────────

def cover_crop_array(img: Image.Image, tw: int, th: int) -> np.ndarray:
    iw, ih = img.size
    scale = max(tw / iw, th / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    l = (nw - tw) // 2
    t = (nh - th) // 2
    return np.array(img.crop((l, t, l + tw, t + th)))


def ken_burns_clip(
    image_source,
    duration: float,
    zoom_min: float = KB_ZOOM_MIN_FOTO,
    zoom_max: float = KB_ZOOM_MAX_FOTO,
    seed: int | None = None,
) -> VideoClip:
    if seed is not None:
        random.seed(seed)

    if isinstance(image_source, str):
        img = Image.open(image_source).convert("RGB")
    else:
        img = image_source.convert("RGB")

    zoom_start = random.uniform(zoom_min, zoom_max)
    zoom_end   = random.uniform(zoom_min, zoom_max)
    while abs(zoom_end - zoom_start) < 0.03:
        zoom_end = random.uniform(zoom_min, zoom_max)

    pan_xs, pan_ys = random.uniform(-1, 1), random.uniform(-1, 1)
    pan_xe, pan_ye = random.uniform(-1, 1), random.uniform(-1, 1)

    canvas_w = int(W * zoom_max * 1.05)
    canvas_h = int(H * zoom_max * 1.05)
    arr_big  = cover_crop_array(img, canvas_w, canvas_h)

    def make_frame(t: float) -> np.ndarray:
        p    = t / duration if duration > 0 else 0.0
        zoom = zoom_start + (zoom_end - zoom_start) * p
        px   = pan_xs + (pan_xe - pan_xs) * p
        py   = pan_ys + (pan_ye - pan_ys) * p

        cw = int(W * zoom)
        ch = int(H * zoom)
        ox = (canvas_w - cw) // 2
        oy = (canvas_h - ch) // 2
        cx = canvas_w // 2 + int(px * ox)
        cy = canvas_h // 2 + int(py * oy)

        x1 = max(0, cx - cw // 2)
        y1 = max(0, cy - ch // 2)
        x2 = min(canvas_w, x1 + cw)
        y2 = min(canvas_h, y1 + ch)

        patch = arr_big[y1:y2, x1:x2]
        return np.array(Image.fromarray(patch).resize((W, H), Image.LANCZOS))

    return VideoClip(make_frame, duration=duration).with_fps(FPS)


# ── Beat-Analyse ──────────────────────────────────────────────────────────────

def analyse_beats(
    audio_path: str,
    max_duration: float = MAX_DURATION,
) -> tuple[float, np.ndarray]:
    y, sr = librosa.load(audio_path, sr=None, mono=True, duration=max_duration)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    return bpm, beat_times


def beat_durations(
    beat_times: np.ndarray,
    max_duration: float,
    beats_per_cut: int,
    titel_duration: float,
) -> list[float]:
    bt = beat_times[beat_times > titel_duration]
    cut_times = [titel_duration] + list(bt[beats_per_cut - 1 :: beats_per_cut])
    cut_times = [t for t in cut_times if t < max_duration]
    cut_times.append(max_duration)
    return [cut_times[i + 1] - cut_times[i] for i in range(len(cut_times) - 1)]


def even_durations(n: int, total: float, titel_dur: float) -> list[float]:
    dur = (total - titel_dur) / n
    return [dur] * n


# ── Crossfade-Montage ─────────────────────────────────────────────────────────

def dissolve(clips: list, crossfade: float):
    if len(clips) == 1:
        return clips[0]
    result = []
    for i, clip in enumerate(clips):
        if i == 0:
            clip = clip.with_effects([FadeOut(crossfade)])
        elif i == len(clips) - 1:
            clip = clip.with_effects([FadeIn(crossfade)])
        else:
            clip = clip.with_effects([FadeIn(crossfade), FadeOut(crossfade)])
        result.append(clip)
    return concatenate_videoclips(result, method="compose", padding=-crossfade)


# ── Haupt-Render-Funktion ─────────────────────────────────────────────────────

def render_reel(
    titel_source,
    foto_sources: list,
    output_path: str,
    audio_path: str | None  = None,
    total_duration: float   = 20.0,
    titel_duration: float   = 3.5,
    crossfade: float        = 0.35,
    beats_per_cut: int      = 2,
    progress_callback=None,
):
    n_fotos = len(foto_sources)
    n_total = 1 + n_fotos
    step    = 0

    def progress(msg):
        nonlocal step
        step += 1
        if progress_callback:
            progress_callback(step, n_total + 2, msg)

    # Segmentlängen
    if audio_path:
        bpm, beat_times = analyse_beats(audio_path, MAX_DURATION)
        foto_durs = beat_durations(beat_times, MAX_DURATION, beats_per_cut, titel_duration)
        if len(foto_durs) > n_fotos:
            foto_durs = foto_durs[:n_fotos]
        elif len(foto_durs) < n_fotos:
            foto_sources = foto_sources[:len(foto_durs)]
    else:
        foto_durs = even_durations(n_fotos, min(total_duration, MAX_DURATION), titel_duration)

    # Clips rendern
    progress("Rendere Titelkarte …")
    titel_clip = ken_burns_clip(
        titel_source, titel_duration,
        zoom_min=KB_ZOOM_MIN_TITEL, zoom_max=KB_ZOOM_MAX_TITEL,
    )

    foto_clips = []
    for i, (src, dur) in enumerate(zip(foto_sources, foto_durs)):
        name = os.path.basename(src) if isinstance(src, str) else f"Foto {i+1}"
        progress(f"Rendere {name} …")
        foto_clips.append(ken_burns_clip(src, dur, seed=i))

    # Montage
    progress("Füge Crossfades zusammen …")
    final = dissolve([titel_clip] + foto_clips, crossfade)
    if final.duration > MAX_DURATION:
        final = final.subclipped(0, MAX_DURATION)

    # Audio
    if audio_path:
        audio = AudioFileClip(audio_path).subclipped(0, final.duration)
        audio = audio.with_effects([AudioFadeOut(AUDIO_FADE_OUT)])
        final = final.with_audio(audio)

    # Export
    progress("Exportiere MP4 …")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    final.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        bitrate="8000k",
        audio_bitrate="192k",
        threads=os.cpu_count(),
        preset="slow",
        ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"],
        logger=None,
    )

    return output_path
