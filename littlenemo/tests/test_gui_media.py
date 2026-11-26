#!/usr/bin/env python3
"""test_gui_media.py - headless tests of attachment validation and the
v3 MEDIA wire format (gui/nemo_media.py, nemo_chat.format_media_lines,
nemo_chat.check_send). Builds its own sample files with Pillow."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gui"))
import nemo_chat  # noqa: E402
import nemo_media  # noqa: E402
from PIL import Image  # noqa: E402

fails = []


def expect(cond, name):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def refused(path, want, needle):
    try:
        nemo_media.inspect_attachment(path, want)
    except nemo_media.AttachmentError as exc:
        return needle in str(exc)
    return False


def main():
    tmp = tempfile.mkdtemp(prefix="nemo-media-")
    png = os.path.join(tmp, "a picture = with spaces.png")
    Image.new("RGB", (64, 48), (200, 10, 10)).save(png)
    gif = os.path.join(tmp, "clip.gif")
    frames = [Image.new("RGB", (32, 32), (i * 40, 0, 0)) for i in range(5)]
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=50)
    still_gif = os.path.join(tmp, "still.gif")
    Image.new("RGB", (8, 8)).save(still_gif)
    mp4 = os.path.join(tmp, "movie.mp4")
    with open(mp4, "wb") as f:
        f.write(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    webm = os.path.join(tmp, "movie.webm")
    with open(webm, "wb") as f:
        f.write(b"\x1a\x45\xdf\xa3" + b"\x00" * 64)
    fake_mp4 = os.path.join(tmp, "really-a-png.mp4")
    with open(png, "rb") as s, open(fake_mp4, "wb") as d:
        d.write(s.read())
    empty = os.path.join(tmp, "empty.png")
    open(empty, "wb").close()
    corrupt = os.path.join(tmp, "corrupt.png")
    with open(png, "rb") as s, open(corrupt, "wb") as d:
        d.write(s.read()[:40])
    txt = os.path.join(tmp, "notes.txt")
    with open(txt, "w") as f:
        f.write("hello")
    big = os.path.join(tmp, "big.png")
    Image.new("RGB", (300, 200)).save(big)

    m = nemo_media.inspect_attachment(png, "image")
    expect(m["type"] == "image" and m["mime"] == "image/png"
           and (m["w"], m["h"]) == (64, 48) and m["bytes"] > 0
           and os.path.isabs(m["path"]), "PNG accepted with size and dims")
    g = nemo_media.inspect_attachment(gif, "video")
    expect(g["type"] == "video" and g["frames"] == 5,
           "animated GIF accepted as video with frame count")
    v = nemo_media.inspect_attachment(mp4, "video")
    expect(v["mime"] == "video/mp4" and v["frames"] == 0,
           "MP4 accepted by signature, frames unknown without ffmpeg")
    expect(nemo_media.inspect_attachment(webm, "video")["mime"] == "video/webm",
           "WebM accepted by EBML signature")
    expect(refused(os.path.join(tmp, "missing.png"), "image", "not found"),
           "missing file refused")
    expect(refused(tmp, "image", "not a regular file"), "directory refused")
    expect(refused(empty, "image", "empty"), "empty file refused")
    expect(refused(txt, "image", "unsupported"), "unsupported extension refused")
    expect(refused(fake_mp4, "video", "do not match"),
           "contents not matching the extension refused")
    expect(refused(png, "video", "unsupported"), "PNG refused as video")
    expect(refused(still_gif, "video", "still image"),
           "single-frame GIF refused as video")
    expect(refused(corrupt, "image", ""), "truncated PNG refused")
    expect(refused(png, "audio", "image or video"), "unknown type refused")

    old_side, old_bytes = nemo_chat.MAX_IMAGE_SIDE, nemo_chat.MAX_IMAGE_BYTES
    try:
        nemo_chat.MAX_IMAGE_SIDE = 256
        expect(refused(big, "image", "bound"), "over-side image refused "
               "before decode")
        nemo_chat.MAX_IMAGE_SIDE = old_side
        nemo_chat.MAX_IMAGE_BYTES = 100
        expect(refused(png, "image", "limit"), "over-size file refused")
    finally:
        nemo_chat.MAX_IMAGE_SIDE, nemo_chat.MAX_IMAGE_BYTES = old_side, old_bytes

    frames_list, note = nemo_media.extract_frames(g, max_frames=3)
    expect(len(frames_list) == 3 and all(os.path.isfile(f["path"])
                                         for f in frames_list)
           and frames_list[0]["ts_ms"] == 0 and frames_list[-1]["ts_ms"] > 0,
           f"GIF frames extracted with timestamps ({note})")
    every, _ = nemo_media.extract_frames(g, max_frames=8)
    expect(len(every) == 5, "every frame kept when fewer than the cap")
    fl, note2 = nemo_media.extract_frames(v)
    if nemo_media.ffmpeg_path() is None:
        expect(fl == [] and "no frame extractor" in note2,
               "MP4 without ffmpeg: no frames, flagged")
    th = nemo_media.thumbnail(m, (), 20)
    expect(th is not None and max(th.size) <= 20, "thumbnail built")

    # wire format
    lines = nemo_chat.format_media_lines(7, dict(g, frames_list=frames_list))
    expect(lines[0].startswith("MEDIA id=7 type=video ") and len(lines) == 4
           and all(l.startswith("FRAME id=7 idx=") for l in lines[1:]),
           "video -> MEDIA + one FRAME per frame")
    pl = nemo_chat.format_media_lines(1, m)[0]
    expect(" " not in pl.split("path=", 1)[1] and "\n" not in pl,
           "path with spaces/'=' stays one token")
    kv = nemo_chat._typed_kv(pl.split()[1:])
    expect(kv["path"] == m["path"] and kv["name"] == os.path.basename(png)
           and kv["bytes"] == m["bytes"] and kv["w"] == 64,
           "MEDIA line round-trips through percent decoding")
    try:
        nemo_chat.format_media_lines(1, dict(m, type="audio"))
        expect(False, "bad type rejected by format_media_lines")
    except ValueError:
        expect(True, "bad type rejected by format_media_lines")

    # replies
    k, p = nemo_chat.parse_line("MEDIA_OK id=3 type=image tokens=256\n")
    expect(k == "media_ok" and p == {"id": 3, "type": "image", "tokens": 256},
           "MEDIA_OK parsed")
    k, p = nemo_chat.parse_line("MEDIA_ERR id=4 reason=file%20not%20found\n")
    expect(k == "media_err" and p["reason"] == "file not found",
           "MEDIA_ERR parsed with decoded reason")

    # send gating
    ready_v2 = {"model": "x", "weights": "raw"}
    ready_v3 = {"media": "image,video"}
    att = [m]
    ok, msg = nemo_chat.check_send("supernemo", False, False, {}, "hi", [])
    expect(not ok and "not available yet" in msg,
           "SuperNemo with no engine: refused, says not available yet")
    ok, msg = nemo_chat.check_send("littlenemo", True, True, ready_v2, "hi", att)
    expect(not ok and "text-only" in msg, "Little Nemo refuses attachments")
    ok, _ = nemo_chat.check_send("littlenemo", True, True, ready_v2, "hi", [])
    expect(ok, "Little Nemo text turn allowed")
    ok, _ = nemo_chat.check_send("supernemo", True, True, ready_v3, "hi",
                                 [m, g])
    expect(ok, "media engine accepts image + video")
    ok, msg = nemo_chat.check_send("supernemo", True, True,
                                   {"media": "image"}, "hi", [g])
    expect(not ok and "video" in msg, "engine without video refuses video")
    ok, msg = nemo_chat.check_send("supernemo", True, True, ready_v3, "hi",
                                   [m] * (nemo_chat.MAX_ATTACHMENTS + 1))
    expect(not ok and "At most" in msg, "attachment count bounded")
    ok, msg = nemo_chat.check_send("supernemo", True, False, ready_v3, "hi", [])
    expect(not ok and "not connected" in msg, "disconnected engine refused")
    ok, msg = nemo_chat.check_send("littlenemo", True, True, ready_v2, "  ", [])
    expect(not ok and msg == "", "empty turn ignored")
    expect(nemo_chat.ready_media(ready_v2) == set(), "v2 READY means text only")

    print("RESULT:", "FAILURES" if fails else "ALL PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
