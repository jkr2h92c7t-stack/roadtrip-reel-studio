"""
Reel Engine — Kernlogik für Video-Generierung.
moviepy 2.x | Ken-Burns mit Ease-in/out | Cinematic Grades | Vignette | Titel-Reveal
"""

import os
import random
import numpy as np
import librosa
from PIL import Image, ImageEnhance
from moviepy import AudioFileClip, VideoClip, concatenate_videoclips
from moviepy.video.fx import FadeIn, FadeOut
from moviepy.audio.fx import AudioFadeOut

W, H = 1080, 1920
FPS  = 30
MAX_DURATION   = 30.0
AUDIO_FADE_OUT = 2.5

# Ruhigere Ken-Burns Defaults
KB_ZOOM_MIN_FOTO  = 1.02
KB_ZOOM_MAX_FOTO  = 1.10
KB_ZOOM_MIN_TITEL = 1.12   # Titel startet reingezoomt, zoomt raus
KB_ZOOM_MAX_TITEL = 1.00   # Ziel: volle Größe


# ── Easing ────────────────────────────────────────────────────────────────────

def ease_in_out_cubic(t: float) -> float:
    """Cubic ease-in-out: langsam starten, beschleunigen, langsam enden."""
    if t < 0.5:
        return 4 * t * t * t
    return 1 - (-2 * t + 2) ** 3 / 2


def ease_out_cubic(t: float) -> float:
    return 1 - (1 - t) ** 3


# ── Vignette (einmal berechnet, dann wiederverwendet) ─────────────────────────

def make_vignette_mask(w: int, h: int, strength: float = 0.55) -> np.ndarray:
    """Erzeugt eine Vignette-Maske [h, w, 1] mit Werten 0–1."""
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    # Elliptische Distanz (9:16 Format berücksichtigen)
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    # Weiche Vignette mit Cosinus-Kurve
    vig = np.cos(np.clip(dist * strength * np.pi / 2, 0, np.pi / 2))
    return vig[:, :, np.newaxis].astype(np.float32)


_VIGNETTE_CACHE: dict[float, np.ndarray] = {}

def get_vignette(strength: float) -> np.ndarray:
    if strength not in _VIGNETTE_CACHE:
        _VIGNETTE_CACHE[strength] = make_vignette_mask(W, H, strength)
    return _VIGNETTE_CACHE[strength]


# ── Color Grading ─────────────────────────────────────────────────────────────

def apply_grade(arr: np.ndarray, grade: str) -> np.ndarray:
    """Wendet einen cinematischen Farbstich auf ein RGB-Array an."""
    if grade == "neutral" or not grade:
        return arr

    img = Image.fromarray(arr)
    r, g, b = img.split()

    if grade == "warm":
        # Goldene Stunde: Reds/Yellows angehoben, Blau reduziert
        r = r.point(lambda x: min(255, int(x * 1.07 + 6)))
        g = g.point(lambda x: min(255, int(x * 1.02 + 2)))
        b = b.point(lambda x: max(0,   int(x * 0.88)))

    elif grade == "cool":
        # Blaustich, kühl und klar
        r = r.point(lambda x: max(0,   int(x * 0.90)))
        g = g.point(lambda x: min(255, int(x * 1.01)))
        b = b.point(lambda x: min(255, int(x * 1.10 + 6)))

    elif grade == "teal_orange":
        # Klassischer Cinema-Look: Shadows teal, Highlights orange
        r = r.point(lambda x: min(255, int(x * 1.12) if x > 110 else int(x * 0.88)))
        g = g.point(lambda x: min(255, int(x * 0.96 + 3)))
        b = b.point(lambda x: min(255, int(x * 0.88) if x > 110 else int(x * 1.14)))

    elif grade == "muted":
        # Verblasster Film-Look: leicht entsättigt, angehobene Schwarzpunkte
        img2 = ImageEnhance.Color(img).enhance(0.82)
        img2 = ImageEnhance.Brightness(img2).enhance(1.04)
        return np.array(img2)

    return np.array(Image.merge("RGB", (r, g, b)))


# ── Bild-Hilfsfunktionen ──────────────────────────────────────────────────────

def cover_crop_array(img: Image.Image, tw: int, th: int) -> np.ndarray:
    iw, ih = img.size
    scale = max(tw / iw, th / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    l = (nw - tw) // 2
    t = (nh - th) // 2
    return np.array(img.crop((l, t, l + tw, t + th)))


# ── Ken-Burns Clip ────────────────────────────────────────────────────────────

def ken_burns_clip(
    image_source,
    duration: float,
    zoom_start: float = KB_ZOOM_MIN_FOTO,
    zoom_end:   float = KB_ZOOM_MAX_FOTO,
    easing     = ease_in_out_cubic,
    grade: str = "warm",
    vignette_strength: float = 0.55,
    seed: int | None = None,
) -> VideoClip:
    """Ken-Burns mit Ease-in/out, Color Grade und Vignette."""
    if seed is not None:
        random.seed(seed)

    if isinstance(image_source, str):
        img = Image.open(image_source).convert("RGB")
    else:
        img = image_source.convert("RGB")

    # Zufällige Pan-Richtung
    pan_xs, pan_ys = random.uniform(-0.7, 0.7), random.uniform(-0.7, 0.7)
    pan_xe, pan_ye = random.uniform(-0.7, 0.7), random.uniform(-0.7, 0.7)

    max_zoom = max(zoom_start, zoom_end) * 1.05
    canvas_w = int(W * max_zoom)
    canvas_h = int(H * max_zoom)
    arr_big  = cover_crop_array(img, canvas_w, canvas_h)

    vig = get_vignette(vignette_strength)

    def make_frame(t: float) -> np.ndarray:
        p    = easing(t / duration if duration > 0 else 0.0)
        zoom = zoom_start + (zoom_end - zoom_start) * p
        px   = pan_xs + (pan_xe - pan_xs) * p
        py   = pan_ys + (pan_ye - pan_ys) * p

        cw = int(W * zoom)
        ch = int(H * zoom)
        ox = max(0, (canvas_w - cw) // 2)
        oy = max(0, (canvas_h - ch) // 2)
        cx = canvas_w // 2 + int(px * ox)
        cy = canvas_h // 2 + int(py * oy)

        x1 = max(0, cx - cw // 2)
        y1 = max(0, cy - ch // 2)
        x2 = min(canvas_w, x1 + cw)
        y2 = min(canvas_h, y1 + ch)

        patch = arr_big[y1:y2, x1:x2]
        frame = np.array(Image.fromarray(patch).resize((W, H), Image.LANCZOS))

        # Color Grade
        frame = apply_grade(frame, grade)

        # Vignette
        if vignette_strength > 0:
            frame = np.clip(frame * vig, 0, 255).astype(np.uint8)

        return frame

    return VideoClip(make_frame, duration=duration).with_fps(FPS)


# ── Titelkarten-Animation ─────────────────────────────────────────────────────

def titel_reveal_clip(
    image_source,
    duration: float,
    grade: str = "warm",
    vignette_strength: float = 0.55,
) -> VideoClip:
    """
    Titelkarte: Zoom-out (1.12→1.0) + Aufhellen (dunkel→hell).
    Wirkt als würde die Karte langsam "erscheinen".
    """
    if isinstance(image_source, str):
        img = Image.open(image_source).convert("RGB")
    else:
        img = image_source.convert("RGB")

    canvas_w = int(W * 1.20)
    canvas_h = int(H * 1.20)
    arr_big  = cover_crop_array(img, canvas_w, canvas_h)
    vig      = get_vignette(vignette_strength * 0.8)   # Vignette etwas weicher auf Titel

    def make_frame(t: float) -> np.ndarray:
        p      = ease_out_cubic(t / duration if duration > 0 else 0.0)
        zoom   = 1.14 - 0.14 * p          # 1.14 → 1.00
        bright = 0.55 + 0.45 * p          # 55 % → 100 % Helligkeit

        cw = int(W * zoom)
        ch = int(H * zoom)
        cx, cy = canvas_w // 2, canvas_h // 2

        x1 = max(0, cx - cw // 2)
        y1 = max(0, cy - ch // 2)
        x2 = min(canvas_w, x1 + cw)
        y2 = min(canvas_h, y1 + ch)

        patch = arr_big[y1:y2, x1:x2]
        frame = np.array(Image.fromarray(patch).resize((W, H), Image.LANCZOS))

        # Helligkeit animieren
        frame = np.clip(frame * bright, 0, 255).astype(np.uint8)

        # Color Grade
        frame = apply_grade(frame, grade)

        # Vignette
        if vignette_strength > 0:
            frame = np.clip(frame * vig, 0, 255).astype(np.uint8)

        return frame

    return VideoClip(make_frame, duration=duration).with_fps(FPS)


# ── Beat-Analyse ──────────────────────────────────────────────────────────────

def analyse_beats(audio_path: str, max_duration: float = MAX_DURATION):
    y, sr = librosa.load(audio_path, sr=None, mono=True, duration=max_duration)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return float(np.atleast_1d(tempo)[0]), beat_times


def beat_durations(beat_times, max_duration, beats_per_cut, titel_duration):
    bt = beat_times[beat_times > titel_duration]
    cut_times = [titel_duration] + list(bt[beats_per_cut - 1 :: beats_per_cut])
    cut_times = [t for t in cut_times if t < max_duration]
    cut_times.append(max_duration)
    return [cut_times[i + 1] - cut_times[i] for i in range(len(cut_times) - 1)]


def even_durations(n, total, titel_dur):
    return [(total - titel_dur) / n] * n


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
    audio_path: str | None = None,
    total_duration: float  = 30.0,
    titel_duration: float  = 4.5,
    crossfade: float       = 0.5,
    beats_per_cut: int     = 3,
    grade: str             = "warm",
    vignette_strength: float = 0.55,
    progress_callback=None,
):
    n_fotos = len(foto_sources)
    step    = 0

    def progress(msg):
        nonlocal step
        step += 1
        if progress_callback:
            progress_callback(step, n_fotos + 3, msg)

    # Segmentlängen
    if audio_path:
        _, beat_times = analyse_beats(audio_path, MAX_DURATION)
        foto_durs = beat_durations(beat_times, MAX_DURATION, beats_per_cut, titel_duration)
        if len(foto_durs) > n_fotos:
            foto_durs = foto_durs[:n_fotos]
        elif len(foto_durs) < n_fotos:
            foto_sources = foto_sources[:len(foto_durs)]
    else:
        foto_durs = even_durations(n_fotos, min(total_duration, MAX_DURATION), titel_duration)

    # Titelkarte
    progress("Rendere Titelkarte …")
    t_clip = titel_reveal_clip(
        titel_source, titel_duration,
        grade=grade, vignette_strength=vignette_strength,
    )

    # Fotos
    foto_clips = []
    for i, (src, dur) in enumerate(zip(foto_sources, foto_durs)):
        name = os.path.basename(src) if isinstance(src, str) else f"Foto {i+1}"
        progress(f"Rendere {name} …")
        # Abwechselnd rein- und rauszoomen für mehr Abwechslung
        if i % 2 == 0:
            z_start, z_end = KB_ZOOM_MIN_FOTO, KB_ZOOM_MAX_FOTO
        else:
            z_start, z_end = KB_ZOOM_MAX_FOTO, KB_ZOOM_MIN_FOTO
        foto_clips.append(ken_burns_clip(
            src, dur,
            zoom_start=z_start, zoom_end=z_end,
            grade=grade, vignette_strength=vignette_strength,
            seed=i,
        ))

    # Montage
    progress("Füge Crossfades zusammen …")
    final = dissolve([t_clip] + foto_clips, crossfade)
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
