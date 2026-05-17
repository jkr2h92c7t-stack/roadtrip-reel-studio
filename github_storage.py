"""
GitHub-basierter Projektspeicher.
Speichert bis zu 8 Projekte im GitHub-Repo unter projects/.
Ältestes Projekt wird automatisch gelöscht wenn Slot 9 belegt wird.
"""

import io
import json
import base64
from datetime import datetime

import requests
from PIL import Image

MAX_PROJECTS   = 8
MAX_IMG_PX     = 1200      # Fotos werden auf max. diese Breite/Höhe komprimiert
JPEG_QUALITY   = 82
PROJECTS_PATH  = "projects"


# ── GitHub REST API Hilfsfunktionen ───────────────────────────────────────────

def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _api(method: str, url: str, token: str, **kwargs):
    resp = requests.request(method, url, headers=_headers(token), **kwargs)
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def _get_file(repo: str, token: str, path: str) -> dict | None:
    """Gibt Datei-Metadaten + content zurück oder None wenn nicht vorhanden."""
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    try:
        return _api("GET", url, token)
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            return None
        raise


def _put_file(repo: str, token: str, path: str, content_bytes: bytes, message: str, sha: str | None = None):
    """Erstellt oder aktualisiert eine Datei im Repo."""
    url  = f"https://api.github.com/repos/{repo}/contents/{path}"
    body = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode(),
    }
    if sha:
        body["sha"] = sha
    return _api("PUT", url, token, json=body)


def _delete_file(repo: str, token: str, path: str, sha: str, message: str):
    url  = f"https://api.github.com/repos/{repo}/contents/{path}"
    body = {"message": message, "sha": sha}
    requests.delete(url, headers=_headers(token), json=body).raise_for_status()


def _list_dir(repo: str, token: str, path: str) -> list[dict]:
    """Gibt Verzeichnisinhalt zurück, leere Liste wenn nicht vorhanden."""
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    try:
        result = _api("GET", url, token)
        return result if isinstance(result, list) else []
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            return []
        raise


# ── Bild-Komprimierung ────────────────────────────────────────────────────────

def _compress_image(img_bytes: bytes, max_px: int = MAX_IMG_PX) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue()


# ── Projekt-Operationen ───────────────────────────────────────────────────────

def list_projects(repo: str, token: str) -> list[dict]:
    """
    Gibt sortierte Liste der gespeicherten Projekte zurück (neueste zuerst).
    Jeder Eintrag: {"name": str, "path": str, "timestamp": str, "meta": dict}
    """
    entries = _list_dir(repo, token, PROJECTS_PATH)
    projects = []
    for entry in entries:
        if entry["type"] != "dir":
            continue
        meta_file = _get_file(repo, token, f"{entry['path']}/project.json")
        if not meta_file:
            continue
        meta = json.loads(base64.b64decode(meta_file["content"]).decode())
        projects.append({
            "name":      entry["name"],
            "path":      entry["path"],
            "timestamp": meta.get("timestamp", entry["name"]),
            "meta":      meta,
            "sha_meta":  meta_file["sha"],
        })
    return sorted(projects, key=lambda p: p["timestamp"], reverse=True)


def save_project(
    repo: str,
    token: str,
    project_name: str,
    titel_bytes: bytes,
    foto_list: list[tuple[str, bytes]],  # [(filename, bytes), ...]
    audio_bytes: bytes | None,
    audio_name: str | None,
    settings: dict,
) -> str:
    """
    Speichert ein Projekt im Repo. Löscht ältestes wenn > MAX_PROJECTS.
    Gibt den Projekt-Ordnerpfad zurück.
    """
    # Ring-Buffer: ältestes löschen wenn voll
    existing = list_projects(repo, token)
    if len(existing) >= MAX_PROJECTS:
        oldest = existing[-1]
        _delete_project(repo, token, oldest["path"])

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug    = project_name.strip().replace(" ", "_")[:30] if project_name else "projekt"
    folder  = f"{PROJECTS_PATH}/{ts}_{slug}"
    msg_pfx = f"[reel-studio] {slug} {ts}"

    # Titelkarte (komprimiert)
    titel_compressed = _compress_image(titel_bytes)
    _put_file(repo, token, f"{folder}/titel.jpg", titel_compressed, f"{msg_pfx}: titel")

    # Fotos (komprimiert, nummeriert)
    foto_filenames = []
    for i, (fname, fbytes) in enumerate(foto_list):
        compressed = _compress_image(fbytes)
        stored_name = f"foto_{i+1:02d}.jpg"
        _put_file(repo, token, f"{folder}/fotos/{stored_name}", compressed, f"{msg_pfx}: foto {i+1}")
        foto_filenames.append({"original": fname, "stored": stored_name})

    # Audio (unverändert, optional)
    if audio_bytes and audio_name:
        ext = audio_name.rsplit(".", 1)[-1] if "." in audio_name else "mp3"
        _put_file(repo, token, f"{folder}/audio/track.{ext}", audio_bytes, f"{msg_pfx}: audio")

    # project.json
    meta = {
        "name":           project_name or slug,
        "timestamp":      ts,
        "foto_filenames": foto_filenames,
        "audio_name":     f"track.{ext}" if audio_bytes and audio_name else None,
        "settings":       settings,
    }
    _put_file(
        repo, token,
        f"{folder}/project.json",
        json.dumps(meta, ensure_ascii=False, indent=2).encode(),
        f"{msg_pfx}: project.json",
    )

    return folder


def load_project(repo: str, token: str, project_path: str) -> dict:
    """
    Lädt ein gespeichertes Projekt und gibt alle Assets als Bytes zurück.
    Rückgabe: {
        "meta": dict,
        "titel_bytes": bytes,
        "foto_list": [(filename, bytes), ...],
        "audio_bytes": bytes | None,
        "audio_name": str | None,
    }
    """
    # Metadaten
    meta_file = _get_file(repo, token, f"{project_path}/project.json")
    meta = json.loads(base64.b64decode(meta_file["content"]).decode())

    # Titelkarte
    titel_file = _get_file(repo, token, f"{project_path}/titel.jpg")
    titel_bytes = base64.b64decode(titel_file["content"])

    # Fotos (in gespeicherter Reihenfolge)
    foto_list = []
    for entry in meta.get("foto_filenames", []):
        stored = entry["stored"]
        f = _get_file(repo, token, f"{project_path}/fotos/{stored}")
        if f:
            foto_list.append((entry["original"], base64.b64decode(f["content"])))

    # Audio
    audio_bytes, audio_name = None, None
    if meta.get("audio_name"):
        af = _get_file(repo, token, f"{project_path}/audio/{meta['audio_name']}")
        if af:
            audio_bytes = base64.b64decode(af["content"])
            audio_name  = meta["audio_name"]

    return {
        "meta":        meta,
        "titel_bytes": titel_bytes,
        "foto_list":   foto_list,
        "audio_bytes": audio_bytes,
        "audio_name":  audio_name,
    }


def _delete_project(repo: str, token: str, project_path: str):
    """Löscht alle Dateien eines Projekts rekursiv."""
    entries = _list_dir(repo, token, project_path)
    for entry in entries:
        if entry["type"] == "file":
            _delete_file(repo, token, entry["path"], entry["sha"], f"[reel-studio] cleanup {entry['path']}")
        elif entry["type"] == "dir":
            _delete_project(repo, token, entry["path"])
