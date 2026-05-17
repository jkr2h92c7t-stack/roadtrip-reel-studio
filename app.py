"""
Roadtrip Reel Studio — Streamlit Web-App
Hochladen → Sortieren → Reel generieren → Herunterladen
Projekte werden in GitHub gespeichert (Ring-Buffer, max. 8 Entwürfe).
"""

import io
import os
import tempfile

import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st
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

    if audio_bytes_to_use:
        cache_key = f"beats_{audio_name_to_use}_{len(audio_bytes_to_use)}"
        if getattr(st.session_state, "_audio_cache_key", None) != cache_key:
            with st.spinner("Analysiere Beats …"):
                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio_name_to_use)[1], delete=False) as tmp:
                    tmp.write(audio_bytes_to_use)
                    tmp_path = tmp.name
                bpm, beat_times = analyse_beats(tmp_path)
                st.session_state.bpm = bpm
                st.session_state.beat_times = beat_times
                st.session_state._audio_cache_key = cache_key

        bpm = st.session_state.bpm
        beat_times = st.session_state.beat_times

        st.markdown(
            f'<span class="chip orange">♩ {bpm:.0f} BPM</span>'
            f'<span class="chip">{len(beat_times)} Beats</span>',
            unsafe_allow_html=True,
        )

        # Beat-Plot
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio_name_to_use)[1], delete=False) as tmp2:
            tmp2.write(audio_bytes_to_use)
            tmp2_path = tmp2.name

        y, sr = librosa.load(tmp2_path, sr=None, mono=True, duration=30.0)
        times = np.linspace(0, len(y)/sr, num=len(y))

        fig = go.Figure()
        step = max(1, len(y)//2000)
        fig.add_trace(go.Scatter(x=times[::step], y=y[::step], mode="lines",
            line=dict(color="#444", width=0.8), showlegend=False))
        for bt in beat_times[beats_per_cut-1::beats_per_cut]:
            fig.add_vline(x=bt, line_width=1.2, line_color="#ff6b35", opacity=0.7)
        fig.update_layout(
            paper_bgcolor="#1a1a1a", plot_bgcolor="#1a1a1a",
            margin=dict(l=8, r=8, t=8, b=30), height=160,
            xaxis=dict(title="Zeit (s)", color="#666", gridcolor="#2a2a2a", tickfont=dict(size=10)),
            yaxis=dict(showticklabels=False, gridcolor="#2a2a2a"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("Orangene Linien = Schnittmarken")
    else:
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
