"""
Roadtrip Reel Studio — Streamlit Web-App
Hochladen → Sortieren → Reel generieren → Herunterladen
Projekte werden in GitHub gespeichert (Ring-Buffer, max. 8 Entwürfe).
"""

import base64
import io
import json
import os
import tempfile

import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from streamlit_sortables import sort_items

from reel_engine import analyse_beats, render_reel
from github_storage import list_projects, save_project, load_project

# ── Seitenkonfiguration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Roadtrip Reel Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d0d0d; color: #f0f0f0; }
  [data-testid="stSidebar"] { background: #141414; border-right: 1px solid #2a2a2a; }
  [data-testid="stSidebarContent"] { padding-top: 1.5rem; }

  .reel-header {
    background: linear-gradient(135deg, #1a1a1a 0%, #222 100%);
    border: 1px solid #333; border-radius: 16px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem; position: relative; overflow: hidden;
  }
  .reel-header::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #ff6b35, #f7c59f, #ff6b35);
  }
  .reel-header h1 {
    font-size: 2.2rem; font-weight: 800; margin: 0;
    background: linear-gradient(135deg, #ff6b35, #f7c59f);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .reel-header p { color: #888; margin: 0.4rem 0 0; font-size: 0.95rem; }

  .project-panel {
    background: #141414; border: 1px solid #2a2a2a;
    border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1.5rem;
  }
  .project-panel h4 { margin: 0 0 0.8rem; color: #ccc; font-size: 0.95rem; }

  .chip { display: inline-block; background: #2a2a2a; border: 1px solid #3a3a3a;
    border-radius: 20px; padding: 3px 12px; font-size: 0.78rem; color: #aaa; margin-right: 6px; }
  .chip.orange { border-color: #ff6b35; color: #ff6b35; }
  .chip.green  { border-color: #22c55e; color: #22c55e; }

  .stButton > button {
    width: 100%; background: linear-gradient(135deg, #ff6b35, #e55a24);
    color: white; border: none; border-radius: 10px;
    padding: 0.75rem 1.5rem; font-weight: 700; font-size: 1rem;
    transition: opacity 0.2s, transform 0.1s;
  }
  .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }
  .stButton > button[disabled] { background: #333 !important; color: #666 !important; }

  .stProgress > div > div { background: linear-gradient(90deg, #ff6b35, #f7c59f); }
  [data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
  }
  hr { border-color: #2a2a2a; }
  label[data-testid="stWidgetLabel"] { color: #ccc !important; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ── GitHub-Konfiguration ───────────────────────────────────────────────────────

def get_github_config():
    """Gibt (token, repo) aus Streamlit-Secrets oder None zurück."""
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo  = st.secrets.get("GITHUB_REPO", "jkr2h92c7t-stack/roadtrip-reel-studio")
        return token, repo
    except Exception:
        return None, None


# ── Session State ──────────────────────────────────────────────────────────────

for key, default in [
    ("foto_order",    []),
    ("render_done",   False),
    ("video_bytes",   None),
    ("video_name",    "reel.mp4"),
    ("beat_times",    None),
    ("bpm",           None),
    ("loaded_titel",  None),   # PIL.Image
    ("loaded_fotos",  None),   # [(name, PIL.Image), ...]
    ("loaded_audio",  None),   # (name, bytes)
    ("loaded_settings", None), # dict
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="reel-header">
  <h1>🎬 Roadtrip Reel Studio</h1>
  <p>Titelkarte + Fotos hochladen · Reihenfolge sortieren · Instagram-Reel generieren</p>
</div>
""", unsafe_allow_html=True)


# ── Projekt-Panel ──────────────────────────────────────────────────────────────

github_token, github_repo = get_github_config()
github_ok = bool(github_token and github_repo)

with st.container():
    st.markdown('<div class="project-panel">', unsafe_allow_html=True)
    st.markdown("#### 📁 Gespeicherte Projekte")

    if not github_ok:
        st.warning(
            "GitHub-Token nicht konfiguriert — Projekte können nicht gespeichert werden. "
            "Füge `GITHUB_TOKEN` in den Streamlit-Secrets hinzu."
        )
    else:
        col_proj_sel, col_proj_btn = st.columns([3, 1])
        with col_proj_sel:
            with st.spinner("Lade Projektliste …"):
                try:
                    saved_projects = list_projects(github_repo, github_token)
                except Exception as e:
                    saved_projects = []
                    st.error(f"Fehler beim Laden der Projekte: {e}")

            if saved_projects:
                proj_labels = [
                    f"{p['meta'].get('name', p['name'])}  —  {p['timestamp'][:4]}-{p['timestamp'][4:6]}-{p['timestamp'][6:8]} {p['timestamp'][9:11]}:{p['timestamp'][11:13]}"
                    for p in saved_projects
                ]
                selected_idx = st.selectbox(
                    "Projekt auswählen",
                    range(len(proj_labels)),
                    format_func=lambda i: proj_labels[i],
                    label_visibility="collapsed",
                )
            else:
                st.markdown('<span class="chip">Noch keine gespeicherten Projekte</span>', unsafe_allow_html=True)
                selected_idx = None

        with col_proj_btn:
            if saved_projects and selected_idx is not None:
                if st.button("⬆️ Laden", use_container_width=True):
                    with st.spinner("Lade Projekt von GitHub …"):
                        try:
                            proj_data = load_project(github_repo, github_token, saved_projects[selected_idx]["path"])
                            # In Session State laden
                            st.session_state.loaded_titel = Image.open(io.BytesIO(proj_data["titel_bytes"]))
                            st.session_state.loaded_fotos = [
                                (fname, Image.open(io.BytesIO(fbytes)))
                                for fname, fbytes in proj_data["foto_list"]
                            ]
                            if proj_data["audio_bytes"]:
                                st.session_state.loaded_audio = (proj_data["audio_name"], proj_data["audio_bytes"])
                            else:
                                st.session_state.loaded_audio = None
                            st.session_state.loaded_settings = proj_data["meta"].get("settings", {})
                            st.session_state.render_done = False
                            st.session_state.video_bytes = None
                            st.success("Projekt geladen!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Fehler: {e}")

        if saved_projects:
            n = len(saved_projects)
            st.markdown(
                f'<span class="chip">{n}/{8} Slots belegt</span>'
                f'<span class="chip orange">Ältestes wird bei Slot 9 überschrieben</span>',
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)


# ── Sidebar: Einstellungen ─────────────────────────────────────────────────────

loaded_s = st.session_state.loaded_settings or {}

with st.sidebar:
    st.markdown("### ⚙️ Einstellungen")
    st.markdown("---")

    titel_dur     = st.slider("Titelkarten-Dauer (s)",      2.0, 7.0, loaded_s.get("titel_dur",     4.5), 0.5)
    total_dur     = st.slider("Gesamtlänge ohne Audio (s)", 15,  30,  loaded_s.get("total_dur",      30),  1)
    crossfade     = st.slider("Crossfade-Dauer (s)",        0.1, 1.2, loaded_s.get("crossfade",      0.5), 0.05)
    beats_per_cut = st.slider("Beats pro Schnitt",          1,   6,   loaded_s.get("beats_per_cut",  3),   1)

    st.markdown("---")
    st.markdown("### 🎨 Cinematic Look")

    grade_options = {
        "warm":        "🌅 Warm — goldene Stunde",
        "teal_orange": "🎬 Teal & Orange — Kino-Klassiker",
        "cool":        "🧊 Cool — klare Luft",
        "muted":       "🎞️ Muted — verblasster Film",
        "neutral":     "⬜ Neutral — kein Farbstich",
    }
    saved_grade = loaded_s.get("grade", "warm")
    grade_idx   = list(grade_options.keys()).index(saved_grade) if saved_grade in grade_options else 0
    grade = st.selectbox(
        "Farbgebung",
        options=list(grade_options.keys()),
        format_func=lambda k: grade_options[k],
        index=grade_idx,
    )

    vignette_strength = st.slider(
        "Vignette", 0.0, 1.0, loaded_s.get("vignette_strength", 0.55), 0.05,
        help="Dunkler Rand-Effekt — gibt Tiefe"
    )

    st.markdown("---")
    st.markdown("### 🎞️ Ken-Burns")
    kb_min = st.slider("Zoom Minimum", 1.00, 1.08, loaded_s.get("kb_min", 1.02), 0.01)
    kb_max = st.slider("Zoom Maximum", 1.04, 1.20, loaded_s.get("kb_max", 1.10), 0.01)

    st.markdown("---")
    if github_ok:
        project_name = st.text_input("📝 Projektname", placeholder="z. B. Tag3_Cefalù")
    st.markdown('<p style="color:#555;font-size:0.75rem;">1080×1920 · 30 fps · H.264 · max. 30 s</p>', unsafe_allow_html=True)


# ── Hauptbereich ───────────────────────────────────────────────────────────────

col_upload, col_preview, col_audio = st.columns([1.1, 1.6, 1.3], gap="large")


# ── Spalte 1: Upload ───────────────────────────────────────────────────────────

with col_upload:
    st.markdown("#### 🖼️ Titelkarte")
    titel_file = st.file_uploader(
        "titel_upload", type=["png", "jpg", "jpeg"],
        label_visibility="collapsed", key="titel_uploader",
    )

    # Aus geladenem Projekt übernehmen wenn kein Upload
    titel_img = None
    if titel_file:
        titel_img = Image.open(titel_file)
    elif st.session_state.loaded_titel:
        titel_img = st.session_state.loaded_titel
        st.markdown('<span class="chip green">✓ Aus Projekt geladen</span>', unsafe_allow_html=True)

    if titel_img:
        st.image(titel_img, use_container_width=True, caption="Titelkarte")

    st.markdown("---")
    st.markdown("#### 📷 Fotos (5–10)")
    foto_files = st.file_uploader(
        "fotos_upload", type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True, label_visibility="collapsed", key="foto_uploader",
    )

    # Aus geladenem Projekt übernehmen
    use_loaded_fotos = False
    if foto_files:
        foto_imgs = [(f.name, Image.open(f)) for f in foto_files]
        if len(foto_imgs) > 10:
            foto_imgs = foto_imgs[:10]
    elif st.session_state.loaded_fotos:
        foto_imgs = st.session_state.loaded_fotos
        use_loaded_fotos = True
        st.markdown('<span class="chip green">✓ Aus Projekt geladen</span>', unsafe_allow_html=True)
    else:
        foto_imgs = []

    if foto_imgs:
        n = len(foto_imgs)
        color = "orange" if 5 <= n <= 10 else ""
        st.markdown(f'<span class="chip {color}">{n} Foto{"s" if n!=1 else ""}</span>', unsafe_allow_html=True)
        if n < 5:
            st.warning(f"Mindestens 5 Fotos empfohlen ({5 - n} fehlen).")

        new_names = [name for name, _ in foto_imgs]
        if new_names != [o["name"] for o in st.session_state.foto_order]:
            st.session_state.foto_order = [{"name": name, "idx": i} for i, name in enumerate(new_names)]


# ── Spalte 2: Vorschau & Sortierung ───────────────────────────────────────────

with col_preview:
    st.markdown("#### 🔀 Reihenfolge (Drag & Drop)")

    if foto_imgs and st.session_state.foto_order:
        foto_map = {name: img for name, img in foto_imgs}

        sorted_names = sort_items(
            [o["name"] for o in st.session_state.foto_order],
            direction="vertical",
            key="sortable_fotos",
        )

        cols_per_row = 3
        foto_sorted = [(n, foto_map[n]) for n in sorted_names if n in foto_map]

        rows = [foto_sorted[i:i+cols_per_row] for i in range(0, len(foto_sorted), cols_per_row)]
        for row_idx, row in enumerate(rows):
            grid_cols = st.columns(cols_per_row, gap="small")
            for col_idx, (fname, img) in enumerate(row):
                abs_idx = row_idx * cols_per_row + col_idx
                with grid_cols[col_idx]:
                    thumb = img.copy().convert("RGB")
                    iw, ih = thumb.size
                    scale = max(140 / iw, 249 / ih)
                    thumb = thumb.resize((int(iw*scale), int(ih*scale)), Image.LANCZOS)
                    l = (thumb.width - 140) // 2
                    t = (thumb.height - 249) // 2
                    thumb = thumb.crop((l, t, l+140, t+249))
                    st.image(thumb, caption=f"#{abs_idx+1} {fname[:14]}", use_container_width=True)

        st.session_state.foto_order = [{"name": n, "idx": i} for i, n in enumerate(sorted_names)]

    elif not foto_imgs:
        st.markdown(
            '<div style="color:#555;text-align:center;padding:4rem 0;">← Fotos hochladen oder Projekt laden</div>',
            unsafe_allow_html=True,
        )


# ── Spalte 3: Audio & Beat-Visualisierung ─────────────────────────────────────

with col_audio:
    st.markdown("#### 🎵 Musik (optional)")
    audio_file = st.file_uploader(
        "audio_upload", type=["mp3", "wav", "aac", "m4a", "ogg"],
        label_visibility="collapsed", key="audio_uploader",
    )

    # Aus geladenem Projekt
    audio_bytes_to_use = None
    audio_name_to_use  = None

    if audio_file:
        audio_bytes_to_use = audio_file.read()
        audio_name_to_use  = audio_file.name
        audio_file.seek(0)
        st.markdown(f'<span class="chip orange">✓ {audio_file.name}</span>', unsafe_allow_html=True)
    elif st.session_state.loaded_audio:
        audio_name_to_use, audio_bytes_to_use = st.session_state.loaded_audio
        st.markdown(f'<span class="chip green">✓ {audio_name_to_use} (aus Projekt)</span>', unsafe_allow_html=True)

    # Standardwerte
    audio_fade_in  = loaded_s.get("audio_fade_in",  0.0)
    audio_fade_out = loaded_s.get("audio_fade_out", 2.0)

    # audio_start aus Session State (wird von der Canvas-Komponente gesetzt)
    if "audio_start" not in st.session_state:
        st.session_state.audio_start = loaded_s.get("audio_start", 0.0)
    if loaded_s.get("audio_start") is not None and not st.session_state.get("_proj_loaded"):
        st.session_state.audio_start = loaded_s["audio_start"]
        st.session_state._proj_loaded = True

    if audio_bytes_to_use:
        cache_key = f"beats_{audio_name_to_use}_{len(audio_bytes_to_use)}"
        if getattr(st.session_state, "_audio_cache_key", None) != cache_key:
            with st.spinner("Analysiere Audio …"):
                with tempfile.NamedTemporaryFile(
                    suffix=os.path.splitext(audio_name_to_use)[1], delete=False
                ) as tmp:
                    tmp.write(audio_bytes_to_use)
                    tmp_path = tmp.name

                bpm, beat_times = analyse_beats(tmp_path)

                import soundfile as sf
                info = sf.info(tmp_path)

                # Wellenform des gesamten Tracks bei niedriger SR für Anzeige
                y_full, sr_full = librosa.load(tmp_path, sr=4000, mono=True)

                st.session_state.bpm          = bpm
                st.session_state.beat_times   = beat_times
                st.session_state.audio_total  = info.duration
                st.session_state.y_full       = y_full
                st.session_state.sr_full      = sr_full
                st.session_state._audio_cache_key = cache_key
                st.session_state._audio_tmp   = tmp_path

        bpm        = st.session_state.bpm
        beat_times = st.session_state.beat_times
        total_len  = st.session_state.audio_total
        y_full     = st.session_state.y_full
        sr_full    = st.session_state.sr_full

        st.markdown(
            f'<span class="chip orange">♩ {bpm:.0f} BPM</span>'
            f'<span class="chip">{int(total_len // 60)}:{int(total_len % 60):02d} min</span>',
            unsafe_allow_html=True,
        )

        window    = float(total_dur)
        max_start = max(0.0, total_len - window)

        # Waveform auf 600 Punkte downsampled für JS
        n_points  = 600
        step_w    = max(1, len(y_full) // n_points)
        wave_data = y_full[::step_w].tolist()

        # Beat-Positionen als Zeitstempel
        beat_list = beat_times[beats_per_cut - 1 :: beats_per_cut].tolist()

        # Audio als base64 für Web Audio API
        audio_b64 = base64.b64encode(audio_bytes_to_use).decode()
        audio_ext = os.path.splitext(audio_name_to_use)[1].lstrip(".") or "mp3"
        audio_mime = {"mp3": "audio/mpeg", "wav": "audio/wav",
                      "ogg": "audio/ogg", "aac": "audio/aac",
                      "m4a": "audio/mp4"}.get(audio_ext, "audio/mpeg")

        cur_start = float(st.session_state.audio_start)
        cur_start = max(0.0, min(cur_start, max_start))

        component_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #1a1a1a; font-family: sans-serif; padding: 8px; }}
  #waveCanvas {{ width: 100%; height: 110px; cursor: grab; display: block;
                 border-radius: 6px; background: #111; }}
  #waveCanvas.dragging {{ cursor: grabbing; }}
  .info {{ color: #888; font-size: 11px; margin: 6px 0 4px; }}
  .info span {{ color: #ff6b35; font-weight: 600; }}
  .controls {{ display: flex; gap: 8px; margin-top: 6px; align-items: center; }}
  button {{ background: #ff6b35; color: #fff; border: none; border-radius: 6px;
            padding: 5px 14px; font-size: 12px; cursor: pointer; font-weight: 600; }}
  button:hover {{ background: #e55a24; }}
  button.stop {{ background: #333; }}
  .fade-row {{ display: flex; gap: 12px; margin-top: 8px; }}
  .fade-col {{ flex: 1; }}
  .fade-col label {{ color: #888; font-size: 11px; display: block; margin-bottom: 3px; }}
  .fade-col input[type=range] {{ width: 100%; accent-color: #ff6b35; }}
  .fade-val {{ color: #ff6b35; font-size: 11px; }}
</style>
</head>
<body>
<canvas id="waveCanvas"></canvas>
<div class="info">
  Fenster: <span id="startLbl">0.0</span>s → <span id="endLbl">0.0</span>s
  &nbsp;·&nbsp; Ziehen zum Verschieben
</div>
<div class="controls">
  <button id="playBtn">▶ Vorschau</button>
  <button id="stopBtn" class="stop">■ Stop</button>
</div>

<script>
const WAVE      = {json.dumps(wave_data)};
const BEATS     = {json.dumps(beat_list)};
const TOTAL     = {total_len:.3f};
const WINDOW    = {window:.3f};
const MAX_START = {max_start:.3f};
const AUDIO_B64 = "{audio_b64}";
const AUDIO_MIME= "{audio_mime}";

let startPos = {cur_start:.3f};
let audioCtx = null, sourceNode = null, isPlaying = false;

// ── Canvas Setup ──────────────────────────────────────────────────────────────
const canvas = document.getElementById("waveCanvas");
const ctx    = canvas.getContext("2d");

function resize() {{
  canvas.width  = canvas.offsetWidth  * window.devicePixelRatio;
  canvas.height = canvas.offsetHeight * window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  draw();
}}

function timeToX(t) {{
  return (t / TOTAL) * canvas.offsetWidth;
}}

function draw() {{
  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  ctx.clearRect(0, 0, W, H);

  const endPos = startPos + WINDOW;
  const wx0 = timeToX(startPos), wx1 = timeToX(endPos);

  // Waveform
  const n = WAVE.length;
  const maxAmp = Math.max(...WAVE.map(Math.abs), 0.001);
  ctx.beginPath();
  for (let i = 0; i < n; i++) {{
    const x = (i / n) * W;
    const t = (i / n) * TOTAL;
    const inWin = t >= startPos && t <= endPos;
    const amp   = (WAVE[i] / maxAmp) * (H * 0.42);
    if (i === 0) ctx.moveTo(x, H/2 - amp);
    else ctx.lineTo(x, H/2 - amp);
  }}
  ctx.strokeStyle = "#2a2a2a";
  ctx.lineWidth = 1;
  ctx.stroke();

  // Waveform im Fenster heller
  ctx.beginPath();
  for (let i = 0; i < n; i++) {{
    const x = (i / n) * W;
    const t = (i / n) * TOTAL;
    if (t < startPos || t > endPos) continue;
    const amp = (WAVE[i] / maxAmp) * (H * 0.42);
    ctx.lineTo(x, H/2 - amp);
  }}
  ctx.strokeStyle = "#666";
  ctx.lineWidth = 1;
  ctx.stroke();

  // Fenster-Hintergrund
  ctx.fillStyle = "rgba(255,107,53,0.10)";
  ctx.fillRect(wx0, 0, wx1 - wx0, H);

  // Beats
  BEATS.forEach(bt => {{
    const x = timeToX(bt);
    const inWin = bt >= startPos && bt <= endPos;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.strokeStyle = inWin ? "rgba(255,107,53,0.85)" : "rgba(80,80,80,0.4)";
    ctx.lineWidth = inWin ? 1.2 : 0.7;
    ctx.stroke();
  }});

  // Fenster-Rahmen
  ctx.strokeStyle = "#ff6b35";
  ctx.lineWidth = 2;
  ctx.strokeRect(wx0 + 1, 1, wx1 - wx0 - 2, H - 2);

  // Labels
  document.getElementById("startLbl").textContent = startPos.toFixed(1);
  document.getElementById("endLbl").textContent   = (startPos + WINDOW).toFixed(1);
}}

// ── Drag ──────────────────────────────────────────────────────────────────────
let dragStartX = null, dragStartPos = null;

canvas.addEventListener("mousedown", e => {{
  dragStartX   = e.offsetX;
  dragStartPos = startPos;
  canvas.classList.add("dragging");
}});

window.addEventListener("mousemove", e => {{
  if (dragStartX === null) return;
  const rect   = canvas.getBoundingClientRect();
  const curX   = e.clientX - rect.left;
  const deltaT = ((curX - dragStartX) / canvas.offsetWidth) * TOTAL;
  startPos = Math.max(0, Math.min(MAX_START, dragStartPos + deltaT));
  draw();
}});

window.addEventListener("mouseup", () => {{
  if (dragStartX === null) return;
  dragStartX = null;
  canvas.classList.remove("dragging");
  // Wert an Streamlit zurückgeben
  window.parent.postMessage({{
    type: "streamlit:setComponentValue",
    value: startPos
  }}, "*");
}});

// Touch support
canvas.addEventListener("touchstart", e => {{
  dragStartX   = e.touches[0].clientX - canvas.getBoundingClientRect().left;
  dragStartPos = startPos;
}}, {{passive: true}});

canvas.addEventListener("touchmove", e => {{
  if (dragStartX === null) return;
  e.preventDefault();
  const curX   = e.touches[0].clientX - canvas.getBoundingClientRect().left;
  const deltaT = ((curX - dragStartX) / canvas.offsetWidth) * TOTAL;
  startPos = Math.max(0, Math.min(MAX_START, dragStartPos + deltaT));
  draw();
}}, {{passive: false}});

canvas.addEventListener("touchend", () => {{
  dragStartX = null;
  window.parent.postMessage({{
    type: "streamlit:setComponentValue",
    value: startPos
  }}, "*");
}});

// ── Audio Playback ────────────────────────────────────────────────────────────
function b64ToArrayBuffer(b64) {{
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
  return buf;
}}

async function play() {{
  if (isPlaying) stop();
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const buf = await audioCtx.decodeAudioData(b64ToArrayBuffer(AUDIO_B64));
  sourceNode = audioCtx.createBufferSource();
  sourceNode.buffer = buf;
  sourceNode.connect(audioCtx.destination);
  sourceNode.start(0, startPos, WINDOW);
  isPlaying = true;
  sourceNode.onended = () => {{ isPlaying = false; }};
}}

function stop() {{
  if (sourceNode) {{ try {{ sourceNode.stop(); }} catch(e) {{}} }}
  if (audioCtx)  {{ audioCtx.close(); }}
  isPlaying = false;
}}

document.getElementById("playBtn").addEventListener("click", play);
document.getElementById("stopBtn").addEventListener("click", stop);

// ── Init ──────────────────────────────────────────────────────────────────────
window.addEventListener("resize", resize);
resize();
</script>
</body>
</html>
"""
        result = components.html(component_html, height=220, scrolling=False)

        # Rückgabewert der Komponente (Drag-Position) in Session State übernehmen
        if result is not None:
            try:
                new_start = float(result)
                if 0.0 <= new_start <= max_start:
                    st.session_state.audio_start = new_start
            except (TypeError, ValueError):
                pass

        audio_start = float(st.session_state.audio_start)
        audio_end   = audio_start + window

        # ── Fade-in / Fade-out ────────────────────────────────────────────
        fc1, fc2 = st.columns(2)
        with fc1:
            audio_fade_in  = st.slider("Fade-in (s)",  0.0, 4.0,
                                       loaded_s.get("audio_fade_in", 0.0), 0.5)
        with fc2:
            audio_fade_out = st.slider("Fade-out (s)", 0.0, 4.0,
                                       loaded_s.get("audio_fade_out", 2.0), 0.5)

    else:
        audio_start, audio_end = 0.0, float(total_dur)
        audio_fade_in, audio_fade_out = 0.0, 2.0
        st.markdown(
            '<div style="color:#555;padding:1rem 0;">Ohne Audio werden Fotos gleichmäßig verteilt.</div>',
            unsafe_allow_html=True,
        )


# ── Render-Button ──────────────────────────────────────────────────────────────

st.markdown("---")
ready = bool(titel_img and foto_imgs)

if not ready:
    st.info("Titelkarte und mindestens ein Foto hochladen (oder Projekt laden), um das Reel zu generieren.")

col_btn, col_name, col_status = st.columns([1, 1, 2])
with col_btn:
    render_clicked = st.button("🎬 Reel generieren", disabled=not ready, use_container_width=True)

if render_clicked and ready:
    st.session_state.render_done = False
    st.session_state.video_bytes = None

    # Sortierte Fotos
    foto_map     = {name: img for name, img in foto_imgs}
    sorted_fotos = [(o["name"], foto_map[o["name"]]) for o in st.session_state.foto_order if o["name"] in foto_map]

    progress_bar = st.progress(0, text="Starte …")
    status_text  = st.empty()

    def on_progress(step, total, msg):
        progress_bar.progress(min(step/total, 1.0), text=msg)
        status_text.markdown(f"*{msg}*")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Titelkarte
        titel_path = os.path.join(tmpdir, "titel.png")
        titel_img.save(titel_path)

        # Fotos
        foto_paths = []
        for fname, img in sorted_fotos:
            p = os.path.join(tmpdir, fname)
            img.convert("RGB").save(p)
            foto_paths.append(p)

        # Audio
        audio_path = None
        if audio_bytes_to_use and audio_name_to_use:
            ext = os.path.splitext(audio_name_to_use)[1]
            audio_path = os.path.join(tmpdir, f"audio{ext}")
            with open(audio_path, "wb") as af:
                af.write(audio_bytes_to_use)

        output_path = os.path.join(tmpdir, "reel.mp4")

        try:
            render_reel(
                titel_source=titel_path, foto_sources=foto_paths,
                output_path=output_path, audio_path=audio_path,
                total_duration=total_dur, titel_duration=titel_dur,
                crossfade=crossfade, beats_per_cut=beats_per_cut,
                grade=grade, vignette_strength=vignette_strength,
                audio_start=audio_start, audio_end=audio_end,
                audio_fade_in=audio_fade_in, audio_fade_out=audio_fade_out,
                progress_callback=on_progress,
            )
            with open(output_path, "rb") as vf:
                st.session_state.video_bytes = vf.read()

            st.session_state.render_done = True
            progress_bar.progress(1.0, text="✓ Fertig!")
            status_text.empty()

            # ── Auto-Save zu GitHub ────────────────────────────────────────
            if github_ok:
                save_status = st.empty()
                save_status.info("💾 Speichere Projekt auf GitHub …")
                try:
                    # Titel als Bytes
                    t_buf = io.BytesIO(); titel_img.save(t_buf, format="PNG"); t_buf.seek(0)

                    # Fotos als Bytes (in sortierter Reihenfolge)
                    f_list = []
                    for fname, img in sorted_fotos:
                        f_buf = io.BytesIO(); img.convert("RGB").save(f_buf, format="JPEG", quality=90); f_buf.seek(0)
                        f_list.append((fname, f_buf.read()))

                    settings = {
                        "titel_dur": titel_dur, "total_dur": total_dur,
                        "crossfade": crossfade, "beats_per_cut": beats_per_cut,
                        "kb_min": kb_min, "kb_max": kb_max,
                        "grade": grade, "vignette_strength": vignette_strength,
                        "audio_start": audio_start,
                        "audio_fade_in": audio_fade_in, "audio_fade_out": audio_fade_out,
                    }
                    pname = locals().get("project_name", "") or ""
                    save_project(
                        repo=github_repo, token=github_token,
                        project_name=pname,
                        titel_bytes=t_buf.getvalue(),
                        foto_list=f_list,
                        audio_bytes=audio_bytes_to_use,
                        audio_name=audio_name_to_use,
                        settings=settings,
                    )
                    save_status.success("✓ Projekt auf GitHub gespeichert!")
                except Exception as e:
                    save_status.warning(f"Render ok, aber Speichern fehlgeschlagen: {e}")

        except Exception as e:
            progress_bar.empty()
            st.error(f"Fehler beim Rendern: {e}")


# ── Download & Vorschau ────────────────────────────────────────────────────────

if st.session_state.render_done and st.session_state.video_bytes:
    st.markdown("---")
    st.markdown("### ✅ Reel ist fertig!")
    dl_col, prev_col = st.columns([1, 2])
    with dl_col:
        st.download_button(
            "⬇️ MP4 herunterladen",
            data=st.session_state.video_bytes,
            file_name=st.session_state.video_name,
            mime="video/mp4",
            use_container_width=True,
        )
        size_mb = len(st.session_state.video_bytes) / 1_048_576
        st.caption(f"Dateigröße: {size_mb:.1f} MB")
    with prev_col:
        st.video(st.session_state.video_bytes)
