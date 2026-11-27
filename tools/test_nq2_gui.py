#!/usr/bin/env python3
"""test_nq2_gui.py - the Quantize dialog's logic and window (tools/).

Run from anywhere:  python tools/test_nq2_gui.py
Needs a built nq2-quantize for sections 4 and 5 (beside the script, in the
repository root, on PATH, or in NQ2_QUANTIZE); those sections report SKIP
without it. Everything written goes to a temp folder; nothing is downloaded
and no real weights are read: the test writes a tiny synthetic safetensors.
"""
import hashlib
import importlib.util
import json
import os
import random
import struct
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nq2_quant as nq  # noqa: E402

fails = []


def expect(cond, name):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def write_safetensors(path, tensors):
    """tensors: list of (name, rows, cols); bf16 values from a fixed seed."""
    rnd = random.Random(7)
    header, blobs, off = {}, [], 0
    for name, r, c in tensors:
        vals = [rnd.gauss(0, 0.05) for _ in range(r * c)]
        raw = b"".join(struct.pack("<f", v)[2:] for v in vals)  # bf16 = top 16
        header[name] = {"dtype": "BF16", "shape": [r, c],
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def section_logic(tmp):
    print("-- 1. options, validation, command line")
    o = nq.new_options()
    expect(1 <= o["threads"] <= 12, "default threads within 1..12")
    expect(nq.default_threads() == min(12, os.cpu_count() or 1),
           "default threads = all logical CPUs, capped at 12")
    cmd = nq.build_command("conv", ["a.safetensors"], "out/x", o)
    expect(cmd[:4] == ["conv", "--out", "out/x", "a.safetensors"]
           and cmd[-2:] == ["--threads", str(o["threads"])]
           and len(cmd) == 6, "defaults pass only --threads")
    p = nq.preset_options("CUDA pack (per-tensor, rms, fp16 embeddings, PQ4)", 6)
    cmd = nq.build_command("conv", ["a"], "o", p)
    expect(all(x in cmd for x in ("--books", "per-tensor", "--norm", "rms",
                                  "--embed-fp16", "--pq", "4"))
           and cmd[-2:] == ["--threads", "6"], "CUDA preset command")
    errs, _ = nq.validate(["a"], os.path.join(tmp, "x"), p)
    expect(not errs, "CUDA preset validates")

    def bad(**kw):
        oo = nq.new_options()
        oo.update(kw)
        return nq.validate(["a"], os.path.join(tmp, "x"), oo)[0]
    expect(bad(pq=4), "PQ needs per-tensor books")
    expect(bad(books="per-tensor", pq=4, format=2), "PQ needs format 3")
    expect(bad(books="per-tensor", pq=5, norm="rms"), "PQ must be 4 or 8")
    expect(bad(books="per-tensor", pq=4, block=128), "PQ needs block 256")
    expect(bad(books="per-tensor", pq=4, budgets=True), "PQ and budgets")
    expect(bad(books="per-tensor", format=1), "per-tensor with format 1")
    expect(bad(books="global", format=2), "global with format 2")
    expect(bad(threads=0) and bad(threads=13) and bad(threads="x"),
           "threads outside 1..12 refused")
    expect(bad(entries=300), "entries above 256 refused")
    expect(bad(block=5000), "block above 4096 refused")
    expect(nq.validate([], "", nq.new_options())[0], "no input, no output")
    _, w = nq.validate(["a"], os.path.join(nq.REPO, "pack"), nq.new_options())
    expect(len(w) == 1 or not os.path.isdir(nq.REPO),
           "output inside the repository warns")

    print("-- 2. progress parsing and inputs")
    expect(nq.parse_progress("  tensors 12 / 291 (4.1%), elapsed 9 s, eta 210 s")
           == 4.1, "tensor progress line")
    expect(nq.parse_progress("  pass 2: 5 / 40 chunks (12.5%), elapsed 1 s, "
                             "eta 7 s") == 12.5, "chunk progress line")
    expect(nq.parse_progress("tensor 3/9 model.layers.0.w: 100 blocks") is None
           and nq.parse_progress("round-trip (4.2%) other") is None,
           "other lines are not progress")
    expect(nq.phase_of("k-means (12 threads): 3 s") == "dictionary learned",
           "phase label")
    d = os.path.join(tmp, "in")
    os.makedirs(d, exist_ok=True)
    for n in ("b.safetensors", "a.safetensors", "readme.txt"):
        open(os.path.join(d, n), "wb").close()
    expect([os.path.basename(x) for x in nq.collect_inputs(d)]
           == ["a.safetensors", "b.safetensors"], "folder: shards sorted")
    expect(nq.collect_inputs(os.path.join(d, "a.safetensors"))
           == [os.path.join(d, "a.safetensors")], "single file")
    expect(nq.collect_inputs(os.path.join(d, "none")) == [], "missing path")
    expect(nq.process_flags(False) == (0x08000000 if os.name == "nt" else 0),
           "console hidden by default")
    expect(nq.process_flags(True) == (0x10 if os.name == "nt" else 0),
           "console shown when asked")

    print("-- 3. settings persist outside the repo")
    sp = os.path.join(tmp, "cfg", "nq2-gui.json")
    expect(nq.save_settings({"threads": 5}, sp)
           and nq.load_settings(sp) == {"threads": 5}, "save and load")
    expect(nq.load_settings(os.path.join(tmp, "nope.json")) == {},
           "missing settings file")
    expect(not nq._inside(nq.settings_path(), nq.REPO),
           "settings live outside the repository")
    expect(nq.clean_threads(99) == 12 and nq.clean_threads(0) == 1
           and nq.clean_threads("q") == nq.default_threads(), "clean_threads")


def section_job(tmp):
    print("-- 4. job runner (fake converter)")
    fake = os.path.join(tmp, "fake.py")
    with open(fake, "w") as f:
        f.write("import sys, time\n"
                "print('parallel per-tensor books: 2 tensors', flush=True)\n"
                "print('  tensors 1 / 2 (50.0%), elapsed 0 s, eta 0 s', "
                "flush=True)\n"
                "if 'slow' in sys.argv:\n"
                "    time.sleep(30)\n"
                "if 'fail' in sys.argv:\n"
                "    print('reject: tensor x', file=sys.stderr); sys.exit(3)\n"
                "print('done')\n")

    def run(extra, stop_after=None):
        lines, prog, done = [], [], []
        job = nq.Job([sys.executable, fake] + extra)
        job.start(lines.append, lambda p, ph: prog.append((p, ph)),
                  lambda rc, m: done.append((rc, m)))
        if stop_after:
            t0 = time.time()
            while not prog and time.time() - t0 < 10:
                time.sleep(0.05)
            job.stop()
        t0 = time.time()
        while not done and time.time() - t0 < 20:
            time.sleep(0.05)
        return lines, prog, done
    lines, prog, done = run([])
    expect(done and done[0][0] == 0 and (50.0, None) in prog
           and "done" in lines, "success: lines, progress, rc 0")
    lines, prog, done = run(["fail"])
    expect(done and done[0][0] == 3 and "reject: tensor x" in lines
           and "exit code 3" in done[0][1], "failure: stderr in log, rc 3")
    lines, prog, done = run(["slow"], stop_after=True)
    expect(done and "Stopped" in done[0][1], "Stop ends the run")
    done = []
    nq.Job([os.path.join(tmp, "no-such-exe")]).start(
        lambda t: None, lambda p, ph: None, lambda rc, m: done.append((rc, m)))
    expect(done and done[0][0] == -1 and "Could not start" in done[0][1],
           "missing converter: clean error")


def section_real(tmp):
    exe = nq.find_converter()
    if not exe:
        print("-- 5. SKIP real converter (nq2-quantize not found)")
        return None
    print("-- 5. real converter: hash identity across thread counts")
    src = os.path.join(tmp, "synth.safetensors")
    write_safetensors(src, [
        ("model.embed_tokens.weight", 64, 256),
        ("model.layers.0.mixer.in_proj.weight", 128, 256),
        ("model.layers.0.mixer.out_proj.weight", 256, 128),
        ("model.layers.1.mixer.q_proj.weight", 128, 256),
        ("model.layers.1.mixer.o_proj.weight", 256, 128),
        ("model.layers.1.mlp.up_proj.weight", 256, 256),
    ])
    recipes = {
        "global": nq.preset_options("Default (one global book)"),
        "per-tensor": nq.preset_options("Per-tensor books (rms)"),
        "cuda-pq4": nq.preset_options(
            "CUDA pack (per-tensor, rms, fp16 embeddings, PQ4)"),
    }
    for rname, base in recipes.items():
        sums = {}
        for t in (1, 6, 12):
            o = dict(base)
            o["threads"] = t
            pre = os.path.join(tmp, "out-%s-t%d" % (rname, t))
            cmd = nq.build_command(exe, [src], pre, o)
            done = []
            nq.Job(cmd).start(lambda s: None, lambda p, ph: None,
                              lambda rc, m: done.append(rc))
            t0 = time.time()
            while not done and time.time() - t0 < 300:
                time.sleep(0.1)
            sums[t] = nq.hash_outputs(pre) if done and done[0] == 0 else None
        ok = (all(sums.values()) and len({json.dumps(sums[t], sort_keys=True)
                                          for t in sums}) == 1
              and len(sums[1]) == 3)
        expect(ok, "%s: .nq2/.nq2book/.nq2manifest identical at 1, 6, 12 "
                   "threads" % rname)
    return exe


def section_window(tmp, exe):
    print("-- 6. window")
    try:
        spec = importlib.util.spec_from_file_location(
            "nq2_gui", os.path.join(HERE, "nq2-gui.py"))
        gui = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gui)
        app = gui.Console()
    except Exception as exc:  # no display
        print("SKIP window: %s" % exc)
        return
    app.settings = {}
    gui.messagebox.showwarning = lambda *a, **k: None
    gui.messagebox.askyesno = lambda *a, **k: True
    gui.messagebox.showerror = lambda *a, **k: None
    saved_cfg = nq.settings_path
    nq.settings_path = lambda: os.path.join(tmp, "win-cfg.json")
    try:
        app.update()
        dlg = gui.QuantizeDialog(app)
        dlg.update()
        expect(dlg.v["threads"].get() == str(nq.default_threads()),
               "dialog opens with default threads")
        dlg._start()          # empty form: refused, nothing running
        expect(dlg.job is None, "empty form refused")
        dlg.v["threads"].set("13")
        cmd, err, _ = dlg._plan()
        expect(cmd is None and "Threads" in err, "bad thread count named")
        if exe:
            src = os.path.join(tmp, "synth.safetensors")
            dlg.src.set(src)
            dlg.out_dir.set(tmp)
            dlg.out_name.set("win-out")
            dlg.exe.set(exe)
            dlg.preset.set("CUDA pack (per-tensor, rms, fp16 embeddings, PQ4)")
            dlg._apply_preset()
            dlg.v["threads"].set("12")
            dlg._show_command()
            expect("--threads 12" in dlg.log.get("1.0", "end"),
                   "Show command lists --threads 12")
            dlg._start()
            t0 = time.time()
            while dlg.job is not None and time.time() - t0 < 300:
                dlg.update()
                time.sleep(0.05)
            for _ in range(5):
                dlg.update()
            text = dlg.log.get("1.0", "end")
            expect(dlg.job is None and "Done." in text
                   and os.path.isfile(os.path.join(tmp, "win-out.nq2")),
                   "run from the window: log, completion, pack written")
        dlg._close()
        app._close()
    finally:
        nq.settings_path = saved_cfg


def main():
    with tempfile.TemporaryDirectory(prefix="nq2gui-") as tmp:
        section_logic(tmp)
        section_job(tmp)
        exe = section_real(tmp)
        section_window(tmp, exe)
    print("RESULT:", "FAILURES" if fails else "ALL PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
