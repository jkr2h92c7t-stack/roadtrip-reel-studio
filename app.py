"""
Roadtrip Reel Studio — Streamlit Web-App
Hochladen → Sortieren → Reel generieren → Herunterladen
"""

import io
import os
import tempfile
import threading

import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from streamlit_sortables import sort_items

from reel_engine import analyse_beats, render_reel

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
  /* Hintergrund & Basis */
  [data-testid="stAppViewContainer"] {
    background: #0d0d0d;
    color: #f0f0f0;
  }
  [data-testid="stSidebar"] {
    background: #141414;
    border-right: 1px solid #2a2a2a;
  }
  [data-testid="stSidebarContent"] { padding-top: 1.5rem; }

  /* Header */
  .reel-header {
    background: linear-gradient(135deg, #1a1a1a 0%, #222 100%);
    border: 1px solid #333;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .reel-header::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #ff6b35, #f7c59f, #ff6b35);
  }
  .reel-header h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(135deg, #ff6b35, #f7c59f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
  }
  .reel-header p {
    color: #888;
    margin: 0.4rem 0 0;
    font-size: 0.95rem;
  }

  /* Upload-Boxen */
  .upload-card {
    background: #1a1a1a;
    border: 1.5px dashed #333;
    border-radius: 12px;
    padding: 1.2rem;
    transition: border-color 0.2s;
  }
  .upload-card:hover { border-color: #ff6b35; }

  /* Foto-Grid */
  .foto-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 10px;
    margin-top: 0.5rem;
  }
  .foto-thumb {
    border-radius: 8px;
    overflow: hidden;
    aspect-ratio: 9/16;
    background: #222;
    position: relative;
  }
  .foto-thumb img { width: 100%; height: 100%; object-fit: cover; }
  .foto-num {
    position: absolute;
    top: 6px; left: 6px;
    background: rgba(255,107,53,0.9);
    color: white;
    border-radius: 50%;
    width: 22px; height: 22px;
    font-size: 11px;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
  }

  /* Settings-Labels */
  label[data-testid="stWidgetLabel"] {
    color: #ccc !important;
    font-size: 0.85rem;
  }

  /* Buttons */
  .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #ff6b35, #e55a24);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 1.5rem;
    font-weight: 700;
    font-size: 1rem;
    transition: opacity 0.2s, transform 0.1s;
  }
  .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }
  .stButton > button:active { transform: translateY(0); }

  /* Progress */
  .stProgress > div > div { background: linear-gradient(90deg, #ff6b35, #f7c59f); }

  /* Chips */
  .chip {
    display: inline-block;
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: #aaa;
    margin-right: 6px;
  }
  .chip.orange { border-color: #ff6b35; color: #ff6b35; }

  /* Download-Button */
  [data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
    margin-top: 0.5rem;
  }

  /* Divider */
  hr { border-color: #2a2a2a; }

  /* Slider accent */
  [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #ff6b35 !important;
  }
  [data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: #ff6b35 !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Session State initialisieren ───────────────────────────────────────────────

for key, default in [
    ("foto_order", []),
    ("render_done", False),
    ("video_bytes", None),
    ("video_name", "reel.mp4"),
    ("beat_times", None),
    ("bpm", None),
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


# ── Sidebar: Einstellungen ─────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Einstellungen")
    st.markdown("---")

    titel_dur = st.slider(
        "Titelkarten-Dauer (s)", 2.0, 6.0, 3.5, 0.5,
        help="Wie lange die Titelkarte am Anfang angezeigt wird"
    )
    total_dur = st.slider(
        "Gesamtlänge ohne Audio (s)", 10, 30, 20, 1,
        help="Wird ignoriert, wenn ein Musiktrack hochgeladen wurde"
    )
    crossfade = st.slider(
        "Crossfade-Dauer (s)", 0.1, 1.0, 0.35, 0.05,
        help="Länge der Überblend-Animation zwischen Segmenten"
    )
    beats_per_cut = st.slider(
        "Beats pro Schnitt", 1, 4, 2, 1,
        help="Höher = ruhigere, längere Schnitte (nur mit Audio)"
    )

    st.markdown("---")
    st.markdown("### 🎞️ Ken-Burns")
    kb_min = st.slider("Zoom Minimum", 1.00, 1.10, 1.05, 0.01)
    kb_max = st.slider("Zoom Maximum", 1.05, 1.30, 1.18, 0.01)

    st.markdown("---")
    st.markdown(
        '<p style="color:#555;font-size:0.75rem;">1080×1920 · 30 fps · H.264 · max. 30 s</p>',
        unsafe_allow_html=True,
    )


# ── Hauptbereich: drei Spalten ─────────────────────────────────────────────────

col_upload, col_preview, col_audio = st.columns([1.1, 1.6, 1.3], gap="large")


# ── Spalte 1: Upload ───────────────────────────────────────────────────────────

with col_upload:
    st.markdown("#### 🖼️ Titelkarte")
    titel_file = st.file_uploader(
        "titel_upload",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        key="titel_uploader",
    )
    if titel_file:
        img_t = Image.open(titel_file)
        st.image(img_t, use_container_width=True, caption="Titelkarte")
        st.markdown(
            f'<span class="chip orange">✓ {titel_file.name}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 📷 Fotos (5–10)")
    foto_files = st.file_uploader(
        "fotos_upload",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="foto_uploader",
    )
    if foto_files:
        n = len(foto_files)
        color = "orange" if 5 <= n <= 10 else ""
        st.markdown(
            f'<span class="chip {color}">{n} Foto{"s" if n != 1 else ""} hochgeladen</span>',
            unsafe_allow_html=True,
        )
        if n < 5:
            st.warning(f"Mindestens 5 Fotos empfohlen ({5 - n} fehlen noch).")
        if n > 10:
            st.info("Nur die ersten 10 Fotos werden verwendet.")
            foto_files = foto_files[:10]

        # Reihenfolge initialisieren/aktualisieren
        new_names = [f.name for f in foto_files]
        if new_names != [o["name"] for o in st.session_state.foto_order]:
            st.session_state.foto_order = [
                {"name": f.name, "idx": i} for i, f in enumerate(foto_files)
            ]


# ── Spalte 2: Vorschau & Sortierung ───────────────────────────────────────────

with col_preview:
    st.markdown("#### 🔀 Reihenfolge (Drag & Drop)")

    if foto_files and st.session_state.foto_order:
        foto_map = {f.name: f for f in foto_files}

        # Sortierbare Liste (Namen als Items)
        sorted_names = sort_items(
            [o["name"] for o in st.session_state.foto_order],
            direction="vertical",
            key="sortable_fotos",
        )

        # Grid-Vorschau in der sortierten Reihenfolge
        cols_per_row = 3
        foto_sorted = [foto_map[n] for n in sorted_names if n in foto_map]

        rows = [foto_sorted[i : i + cols_per_row] for i in range(0, len(foto_sorted), cols_per_row)]
        for row_idx, row in enumerate(rows):
            grid_cols = st.columns(cols_per_row, gap="small")
            for col_idx, f in enumerate(row):
                abs_idx = row_idx * cols_per_row + col_idx
                with grid_cols[col_idx]:
                    img = Image.open(f)
                    # Thumbnail 9:16
                    thumb = img.copy()
                    iw, ih = thumb.size
                    scale = max(140 / iw, 249 / ih)
                    thumb = thumb.resize((int(iw * scale), int(ih * scale)), Image.LANCZOS)
                    l = (thumb.width - 140) // 2
                    t = (thumb.height - 249) // 2
                    thumb = thumb.crop((l, t, l + 140, t + 249))
                    st.image(thumb, caption=f"#{abs_idx + 1} {f.name[:16]}", use_container_width=True)

        # Sortier-Zustand speichern
        st.session_state.foto_order = [{"name": n, "idx": i} for i, n in enumerate(sorted_names)]

    elif not foto_files:
        st.markdown(
            '<div style="color:#555;text-align:center;padding:4rem 0;">'
            '← Fotos hochladen,<br>dann hier sortieren</div>',
            unsafe_allow_html=True,
        )


# ── Spalte 3: Audio & Beat-Visualisierung ─────────────────────────────────────

with col_audio:
    st.markdown("#### 🎵 Musik (optional)")
    audio_file = st.file_uploader(
        "audio_upload",
        type=["mp3", "wav", "aac", "m4a", "ogg"],
        label_visibility="collapsed",
        key="audio_uploader",
    )

    if audio_file:
        st.markdown(
            f'<span class="chip orange">✓ {audio_file.name}</span>',
            unsafe_allow_html=True,
        )

        # Beat-Analyse (gecacht per Dateiname)
        if (
            st.session_state.bpm is None
            or getattr(st.session_state, "_audio_name", None) != audio_file.name
        ):
            with st.spinner("Analysiere Beats …"):
                with tempfile.NamedTemporaryFile(
                    suffix=os.path.splitext(audio_file.name)[1], delete=False
                ) as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name
                bpm, beat_times = analyse_beats(tmp_path)
                st.session_state.bpm = bpm
                st.session_state.beat_times = beat_times
                st.session_state._audio_name = audio_file.name

        bpm = st.session_state.bpm
        beat_times = st.session_state.beat_times

        st.markdown(
            f'<span class="chip orange">♩ {bpm:.0f} BPM</span>'
            f'<span class="chip">{len(beat_times)} Beats erkannt</span>',
            unsafe_allow_html=True,
        )

        # Beat-Waveform-Plot
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(audio_file.name)[1], delete=False
        ) as tmp2:
            audio_file.seek(0)
            tmp2.write(audio_file.read())
            tmp2_path = tmp2.name

        y, sr = librosa.load(tmp2_path, sr=None, mono=True, duration=30.0)
        times = np.linspace(0, len(y) / sr, num=len(y))

        fig = go.Figure()
        # Waveform
        step = max(1, len(y) // 2000)
        fig.add_trace(go.Scatter(
            x=times[::step], y=y[::step],
            mode="lines",
            line=dict(color="#444", width=0.8),
            name="Waveform",
            showlegend=False,
        ))
        # Beat-Marker
        beat_cut_times = beat_times[beats_per_cut - 1 :: beats_per_cut]
        for bt in beat_cut_times:
            fig.add_vline(x=bt, line_width=1.2, line_color="#ff6b35", opacity=0.7)

        fig.update_layout(
            paper_bgcolor="#1a1a1a",
            plot_bgcolor="#1a1a1a",
            margin=dict(l=8, r=8, t=8, b=30),
            height=160,
            xaxis=dict(
                title="Zeit (s)", color="#666",
                gridcolor="#2a2a2a", tickfont=dict(size=10),
            ),
            yaxis=dict(showticklabels=False, gridcolor="#2a2a2a"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("Orangene Linien = Schnittmarken")

    else:
        st.markdown(
            '<div style="color:#555;padding:1rem 0;">'
            'Ohne Audio werden Fotos gleichmäßig auf die gewählte Gesamtlänge verteilt.'
            '</div>',
            unsafe_allow_html=True,
        )


# ── Render-Button ──────────────────────────────────────────────────────────────

st.markdown("---")

ready = bool(titel_file and foto_files)

if not ready:
    st.info("Titelkarte und mindestens ein Foto hochladen, um das Reel zu generieren.")

col_btn, col_status = st.columns([1, 3])

with col_btn:
    render_clicked = st.button(
        "🎬 Reel generieren",
        disabled=not ready,
        use_container_width=True,
    )

if render_clicked and ready:
    st.session_state.render_done = False
    st.session_state.video_bytes = None

    # Fotos in sortierter Reihenfolge
    foto_map = {f.name: f for f in foto_files}
    sorted_fotos = [
        foto_map[o["name"]]
        for o in st.session_state.foto_order
        if o["name"] in foto_map
    ]

    progress_bar = st.progress(0, text="Starte …")
    status_text  = st.empty()

    def on_progress(step, total, msg):
        pct = min(step / total, 1.0)
        progress_bar.progress(pct, text=msg)
        status_text.markdown(f"*{msg}*")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Titelkarte speichern
        titel_path = os.path.join(tmpdir, "titel.png")
        Image.open(titel_file).save(titel_path)

        # Fotos speichern
        foto_paths = []
        for f in sorted_fotos:
            p = os.path.join(tmpdir, f.name)
            Image.open(f).save(p)
            foto_paths.append(p)

        # Audio speichern (falls vorhanden)
        audio_path = None
        if audio_file:
            ext = os.path.splitext(audio_file.name)[1]
            audio_path = os.path.join(tmpdir, f"audio{ext}")
            audio_file.seek(0)
            with open(audio_path, "wb") as af:
                af.write(audio_file.read())

        # Output-Pfad
        reel_name  = "roadtrip_reel"
        output_path = os.path.join(tmpdir, f"{reel_name}.mp4")

        try:
            render_reel(
                titel_source   = titel_path,
                foto_sources   = foto_paths,
                output_path    = output_path,
                audio_path     = audio_path,
                total_duration = total_dur,
                titel_duration = titel_dur,
                crossfade      = crossfade,
                beats_per_cut  = beats_per_cut,
                progress_callback=on_progress,
            )

            with open(output_path, "rb") as vf:
                st.session_state.video_bytes = vf.read()

            st.session_state.render_done = True
            st.session_state.video_name  = f"{reel_name}.mp4"
            progress_bar.progress(1.0, text="✓ Fertig!")
            status_text.empty()

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
            label="⬇️ MP4 herunterladen",
            data=st.session_state.video_bytes,
            file_name=st.session_state.video_name,
            mime="video/mp4",
            use_container_width=True,
        )
        size_mb = len(st.session_state.video_bytes) / 1_048_576
        st.caption(f"Dateigröße: {size_mb:.1f} MB")

    with prev_col:
        st.video(st.session_state.video_bytes)
