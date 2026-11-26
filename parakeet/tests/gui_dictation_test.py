"""GUI test for the dictation button: launches NTSE Chat (fake LLM engine),
replays a reference WAV through parakeet-c as the 'mic', screenshots the
composer mid-stream and after DONE, checks the composer text, closes.
Test tooling only (the speech runtime itself is pure C)."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
GUI = os.environ.get("NEMO_GUI_DIR", os.path.join(HERE, "..", "..", "littlenemo", "gui"))
OUT = os.path.join(HERE, "..", "shots")
WAV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("PARAKEET_MODEL", "models"), "samples", "2086-149220-0033.wav")
REF = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.environ.get("PARAKEET_MODEL", "models"), "samples", "ref", "2086-149220-0033.txt")
os.chdir(GUI); sys.path.insert(0, GUI); os.makedirs(OUT, exist_ok=True)
import nemo_ui
from PIL import ImageGrab

orig_init = nemo_ui.ChatApp.__init__
result = {"ok": False, "log": []}

def log(m):
    result["log"].append(m); print(m, flush=True)

def patched(self, *a, **k):
    orig_init(self, *a, **k)
    root = self.root
    t0 = time.monotonic()
    shots = {"partial": False}
    def grab(name):
        root.update()
        x, y = root.winfo_rootx(), root.winfo_rooty()
        w, h = root.winfo_width(), root.winfo_height()
        ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True).save(os.path.join(OUT, name))
        log(f"shot {name}")
    def wait_ready():
        d = getattr(self, "dictation", None)
        if d is None:
            log("FAIL no dictation widget"); root.after(100, self.on_close); return
        if d.proc and d.proc.ready:
            log(f"{time.monotonic()-t0:.2f}s parakeet ready")
            grab("dictation-0-idle.png")
            self.input.delete("1.0", "end"); self.input.insert("1.0", "Nemo, ")
            self.input.mark_set("insert", "end-1c")
            d.start(sim=WAV); root.after(60, watch); return
        if time.monotonic() - t0 > 40:
            log("FAIL parakeet never READY"); root.after(100, self.on_close); return
        root.after(100, wait_ready)
    t_start = [0.0]
    def watch():
        d = self.dictation
        if not t_start[0]:
            t_start[0] = time.monotonic()
        el = time.monotonic() - t_start[0]
        if d.partial and not shots["partial"] and el > 2.5:
            shots["partial"] = True; grab("dictation-1-streaming.png")
            log(f"{el:.2f}s partial: {d.partial!r}")
        if el > 8.2 and d.state == "rec":
            d.stop(); log(f"{el:.2f}s STOP clicked")
        if d.state == "idle" and el > 8.2:
            txt = self.input.get("1.0", "end-1c")
            ref = open(REF, encoding="utf-8").read().strip()
            log(f"{el:.2f}s composer: {txt!r}")
            result["ok"] = txt == "Nemo, " + ref
            log("PASS composer matches reference" if result["ok"] else f"FAIL expected {('Nemo, ' + ref)!r}")
            grab("dictation-2-done.png")
            root.after(300, self.on_close); return
        if el > 40:
            log("FAIL timeout"); root.after(100, self.on_close); return
        root.after(60, watch)
    root.after(800, wait_ready)

nemo_ui.ChatApp.__init__ = patched
import nemo_chat
fake = sys.executable + " ../tests/fake_engine.py --stdio --model littlenemo"
nemo_chat.main(["--model", "littlenemo", "--engine", fake, "--supernemo-engine",
                sys.executable + " ../tests/fake_engine.py --stdio --model supernemo", "--theme", "pixel"])
sys.exit(0 if result["ok"] else 1)
