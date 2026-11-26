#!/usr/bin/env python3
"""test_gui_ui_smoke.py - opens the real chat window against the fake
engine fixture and drives it with the scripted demos (gui/nemo_demo.py):
an image + video turn with live thinking on SuperNemo, the expanded
thinking trace, the stats panel rows, STOP, Little Nemo's text-only
refusal and a text turn, and SuperNemo with no engine ("not available
yet", nothing sent). Skips (exit 0) when Tk or a display is missing."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GUI = os.path.join(HERE, "..", "gui", "nemo_chat.py")
FAKE = os.path.join(HERE, "fake_engine.py")


def run(args, timeout=240):
    p = subprocess.run([sys.executable, GUI] + args, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def main():
    try:
        import tkinter

        r = tkinter.Tk()
        r.destroy()
    except Exception as exc:
        print(f"SKIP: no Tk display ({exc})")
        print("RESULT: SKIPPED")
        return 0
    py = f'"{sys.executable}"'
    fails = []
    cfg = os.path.join(HERE, "no-such-dir", "none.ini")  # no INI at all
    code, out = run(["--config", cfg, "--model", "supernemo",
                     "--supernemo-engine",
                     f'{py} "{FAKE}" --stdio --model supernemo --delay 0.01',
                     "--engine",
                     f'{py} "{FAKE}" --stdio --model littlenemo --delay 0.01',
                     "--demo", "smoke"])
    ok = code == 0 and "DEMO OK" in out
    print(("ok   " if ok else "FAIL ") + "smoke demo: media turn, thinking "
          "line, stats, STOP, Little Nemo refusal + text turn")
    if not ok:
        print(out[-3000:])
        fails.append("smoke")
    code, out = run(["--config", cfg, "--model", "supernemo",
                     "--demo", "unavailable"])
    ok = code == 0 and "DEMO OK" in out
    print(("ok   " if ok else "FAIL ") + "SuperNemo without an engine: "
          "not available yet, nothing sent")
    if not ok:
        print(out[-3000:])
        fails.append("unavailable")
    code, out = run(["--config", cfg, "--model", "supernemo",
                     "--supernemo-engine",
                     f'{py} "{FAKE}" --stdio --model supernemo --delay 0.01',
                     "--engine",
                     f'{py} "{FAKE}" --stdio --model littlenemo --delay 0.01',
                     "--demo", "resize"])
    ok = code == 0 and "DEMO OK" in out
    print(("ok   " if ok else "FAIL ") + "resize: maximize/restore, minsize, "
          "odd sizes, 1080p fullscreen, wrap, no overlap")
    if not ok:
        print(out[-4000:])
        fails.append("resize")
    print("RESULT:", "FAILURES" if fails else "ALL PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
