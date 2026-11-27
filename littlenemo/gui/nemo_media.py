#!/usr/bin/env python3
"""nemo_media.py - attachment inspection for the chat GUI (no Tk).

Validates an image or video file before the GUI announces it to the
engine (PROTOCOL.md v3, MEDIA): the file must exist, be a regular
file, be non-empty and inside the size bounds, carry a supported
extension, and its leading bytes must agree with that extension.
Images are opened with Pillow (already installed) and checked against
the side and pixel-count bounds before anything decodes the pixels,
so a decompression bomb is refused, not rendered.

Video frames: there is no ffmpeg or OpenCV on the development machine and the
GUI may not download them, so frame extraction uses what exists:
ffmpeg if one is on PATH, else Pillow for animated GIF / WebP. For
MP4 / MOV / WebM / MKV / AVI without ffmpeg, the attachment is still
validated and sent by path with frames=0, and the GUI flags that the
engine has to decode the video itself.

Everything here runs off the Tk thread (the GUI calls it from a
worker and hands results back through a queue).
"""

import os
import shutil
import subprocess

import nemo_proc
import tempfile

import nemo_chat as proto

IMAGE_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".bmp": "image/bmp", ".gif": "image/gif",
}
VIDEO_MIME = {
    ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
    ".webm": "video/webm", ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo", ".gif": "image/gif", ".webp": "image/webp",
}
IMAGE_EXTS = tuple(sorted(IMAGE_MIME))
VIDEO_EXTS = tuple(sorted(VIDEO_MIME))


class AttachmentError(ValueError):
    """A file the GUI refuses to attach; str() is the reason."""


def sniff(head):
    """Container family from a file's first bytes, or None."""
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if head.startswith(b"BM"):
        return "bmp"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return "avi"
    if head[4:8] == b"ftyp":
        return "isobmff"  # mp4 / m4v / mov
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return "ebml"  # webm / mkv
    if head[4:8] in (b"moov", b"mdat", b"wide", b"free"):
        return "isobmff"  # old QuickTime without ftyp
    return None


_EXT_FAMILY = {
    ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".gif": "gif",
    ".bmp": "bmp", ".webp": "webp", ".avi": "avi", ".mp4": "isobmff",
    ".m4v": "isobmff", ".mov": "isobmff", ".webm": "ebml", ".mkv": "ebml",
}


def ffmpeg_path():
    return shutil.which("ffmpeg")


def frame_extractor_note():
    """One line saying how video frames will be obtained here."""
    if ffmpeg_path():
        return "frames via ffmpeg"
    return ("no ffmpeg/OpenCV installed: animated GIF/WebP frames via "
            "Pillow; other video is sent by path for the engine to decode")


def inspect_attachment(path, want):
    """Validate one file for attachment as `want` ('image' or 'video').
    Returns a dict: type, mime, bytes, name, path (absolute), and for
    images w/h, for videos frames (0 when unknown) and maybe w/h.
    Raises AttachmentError with the reason when it is refused."""
    if want not in proto.MEDIA_TYPES:
        raise AttachmentError("attachment type must be image or video")
    if not path:
        raise AttachmentError("no file chosen")
    path = os.path.abspath(path)
    name = os.path.basename(path)
    if len(path) > proto.MAX_PATH_CHARS:
        raise AttachmentError("path is too long")
    if not os.path.exists(path):
        raise AttachmentError(f"{name}: file not found")
    if not os.path.isfile(path):
        raise AttachmentError(f"{name}: not a regular file")
    size = os.path.getsize(path)
    if size <= 0:
        raise AttachmentError(f"{name}: file is empty")
    ext = os.path.splitext(name)[1].lower()
    table = IMAGE_MIME if want == "image" else VIDEO_MIME
    if ext not in table:
        allowed = ", ".join(IMAGE_EXTS if want == "image" else VIDEO_EXTS)
        raise AttachmentError(f"{name}: unsupported {want} type "
                              f"(allowed: {allowed})")
    limit = (proto.MAX_IMAGE_BYTES if want == "image"
             else proto.MAX_VIDEO_BYTES)
    if size > limit:
        raise AttachmentError(f"{name}: {human_bytes(size)} is over the "
                              f"{human_bytes(limit)} {want} limit")
    with open(path, "rb") as f:
        head = f.read(32)
    family = sniff(head)
    if family != _EXT_FAMILY[ext]:
        raise AttachmentError(f"{name}: contents do not match the {ext} "
                              f"extension (looks like {family or 'unknown data'})")
    meta = {"type": want, "mime": table[ext], "bytes": size,
            "name": name, "path": path}
    if want == "image" or family in ("gif", "webp"):
        w, h, frames = _pil_probe(path, name)
        meta["w"], meta["h"] = w, h
        if want == "video":
            if frames < 2:
                raise AttachmentError(f"{name}: a still image, not an "
                                      f"animation; attach it as an image")
            meta["frames"] = frames
    else:
        meta["frames"] = 0  # unknown until decoded (no ffmpeg here)
    return meta


def _pil_probe(path, name):
    """Header-level size check with Pillow, then a verify pass. The
    size bounds are checked before any pixel data is decoded."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is installed
        raise AttachmentError(f"{name}: Pillow is not available ({exc})")
    try:
        with Image.open(path) as im:
            w, h = im.size
            frames = int(getattr(im, "n_frames", 1) or 1)
    except Image.DecompressionBombError:
        raise AttachmentError(f"{name}: image is too large to decode safely")
    except Exception as exc:
        raise AttachmentError(f"{name}: not a readable image ({exc})")
    if w <= 0 or h <= 0:
        raise AttachmentError(f"{name}: image has no pixels")
    if (w > proto.MAX_IMAGE_SIDE or h > proto.MAX_IMAGE_SIDE
            or w * h > proto.MAX_IMAGE_PIXELS):
        raise AttachmentError(f"{name}: {w}x{h} is over the "
                              f"{proto.MAX_IMAGE_SIDE}px / "
                              f"{proto.MAX_IMAGE_PIXELS // 1_000_000} MP bound")
    try:
        with Image.open(path) as im:
            im.verify()
    except Exception as exc:
        raise AttachmentError(f"{name}: image data is corrupt ({exc})")
    return w, h, frames


def extract_frames(meta, out_dir=None, max_frames=8):
    """Sample up to max_frames frames of a video attachment as PNG
    files. Returns (frames_list, note): frames_list is a list of
    {'path', 'ts_ms'}; note says how they were obtained or why there
    are none. Never raises for a decode problem: the attachment is
    then sent without frames and the note says so."""
    max_frames = max(1, min(int(max_frames), proto.MAX_FRAMES))
    if meta.get("type") != "video":
        return [], "not a video"
    out_dir = out_dir or tempfile.mkdtemp(prefix="nemo-frames-")
    src = meta["path"]
    if meta.get("mime") in ("image/gif", "image/webp"):
        return _pil_frames(src, out_dir, max_frames)
    ff = ffmpeg_path()
    if ff:
        return _ffmpeg_frames(ff, src, out_dir, max_frames)
    return [], ("no frame extractor installed (ffmpeg/OpenCV); sent by "
                "path for the engine to decode")


def _pil_frames(src, out_dir, max_frames):
    from PIL import Image

    out = []
    try:
        with Image.open(src) as im:
            n = int(getattr(im, "n_frames", 1) or 1)
            count = min(n, max_frames)
            picks = sorted({round(i * (n - 1) / max(1, count - 1))
                            for i in range(count)})
            ts = 0
            stamps = []
            for i in range(n):
                im.seek(i)
                stamps.append(ts)
                ts += int(im.info.get("duration", 100) or 100)
            for k, i in enumerate(picks):
                im.seek(i)
                p = os.path.join(out_dir, f"frame_{k:02d}.png")
                im.convert("RGB").save(p)
                out.append({"path": p, "ts_ms": stamps[i]})
    except Exception as exc:
        return [], f"frame extraction failed ({exc}); sent without frames"
    return out, f"{len(out)} of {n} frames via Pillow"


def _ffmpeg_frames(ff, src, out_dir, max_frames):
    pattern = os.path.join(out_dir, "frame_%02d.png")
    try:
        subprocess.run(
            [ff, "-v", "error", "-i", src, "-vf",
             "thumbnail,scale='min(1024,iw)':-2", "-frames:v",
             str(max_frames), "-vsync", "vfr", pattern],
            check=True, timeout=120, capture_output=True,
            **nemo_proc.hidden_kwargs())
    except Exception as exc:
        return [], f"ffmpeg could not decode it ({exc}); sent without frames"
    names = sorted(n for n in os.listdir(out_dir) if n.startswith("frame_"))
    return ([{"path": os.path.join(out_dir, n), "ts_ms": 0}
             for n in names[:max_frames]],
            f"{len(names[:max_frames])} frames via ffmpeg")


def thumbnail(meta, frames_list=(), size=56):
    """A small Pillow image for the attachment chip, or None. Called
    on the worker thread; the GUI turns it into a PhotoImage."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None
    src = None
    if meta.get("type") == "image" or meta.get("mime") in ("image/gif",
                                                          "image/webp"):
        src = meta["path"]
    elif frames_list:
        src = frames_list[0]["path"]
    if not src:
        return None
    try:
        with Image.open(src) as im:
            im.seek(0)
            im = im.convert("RGB")
            im.thumbnail((size, size))
            return im.copy()
    except Exception:
        return None


def human_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
