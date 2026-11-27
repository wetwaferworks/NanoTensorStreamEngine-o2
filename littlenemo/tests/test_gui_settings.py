#!/usr/bin/env python3
"""test_gui_settings.py - headless test of the chat GUI's runtime
Settings path (resolve_settings / validate_settings /
apply_settings in gui/nemo_chat.py), driven exactly the way the
panel drives it: raw field strings in, INI written, engine
respawned, READY line checked.

The engine here is a small stand-in script that takes the real
engine's spawn arguments and answers with a READY line built
from them, so the test verifies the GUI side of the contract
(resolve, validate, write, build, respawn, parse) end to end.
That the C engine honors the same arguments is covered by the
engine's own suites. Also asserted: unrelated INI content
survives an apply, and a failed validation writes nothing.
"""
import os
import stat
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gui"))
import nemo_chat  # noqa: E402

FAKE_ENGINE = '''#!/usr/bin/env python3
import os
import sys


def main():
    args = sys.argv[1:]
    mode = "max"
    threads = None
    simd = 0
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--mode":
            mode = args[i + 1]
            i += 2
        elif a == "--threads":
            threads = int(args[i + 1])
            i += 2
        elif a == "--simd":
            simd = 1
            i += 1
        elif a in ("--max-new", "--max-pos"):
            i += 2
        else:
            i += 1
    cores = os.cpu_count() or 1
    if threads is None:
        threads = cores
    sys.stdout.write(
        "READY nemo-engine 2 model=fake-nemo weights=raw layers=56 "
        f"cores={cores} threads={threads} mode={mode} simd={simd}\\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if line == "QUIT":
            return
        if line.startswith("PROMPT"):
            sys.stdout.write("TOK ok \\n")
            sys.stdout.write(
                "STATS tokens=1 ms=1.0 tps=1.0 ms_per_tok=1.0 "
                "gbs_avg=1.0 peak_mb=1.0\\n")
            sys.stdout.write("DONE\\n")
            sys.stdout.flush()


main()
'''

INI_TEMPLATE = """; test settings file - comments must survive an apply
[paths]
engine     = {engine}
checkpoint = {ckpt}
tokenizer  = {tok}

[engine]
mode    = max
threads = 0
simd    = false
max_new = 256
max_pos = 4096

[remote]
host =
port =

[notes]
custom_key = keepme
"""


def wait_ready(events, ready_ev, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ready_ev.wait(0.2):
            for kind, payload in events:
                if kind == "ready":
                    return payload
            ready_ev.clear()
    return None


def main():
    fails = []

    def expect(cond, name):
        print(("ok   " if cond else "FAIL ") + name)
        if not cond:
            fails.append(name)

    tmp = tempfile.mkdtemp(prefix="nemo-settings-")
    fake = os.path.join(tmp, "fake_engine.py")
    with open(fake, "w", encoding="utf-8") as f:
        f.write(FAKE_ENGINE)
    os.chmod(fake, os.stat(fake).st_mode | stat.S_IEXEC)
    ckpt = os.path.join(tmp, "ckpt")
    tok = os.path.join(tmp, "tokenizer.json")
    for p in (ckpt, tok):
        with open(p, "w", encoding="utf-8") as f:
            f.write("x")
    ini = os.path.join(tmp, "littlenemo.ini")
    with open(ini, "w", encoding="utf-8") as f:
        f.write(INI_TEMPLATE.format(engine=fake, ckpt=ckpt, tok=tok))
    argv = ["--config", ini]

    # 1. Resolution: the panel initializes from the effective INI.
    values, found = nemo_chat.resolve_settings(argv)
    expect(found == ini, "resolve_settings finds the INI")
    expect(values["mode"] == "max" and values["threads"] == "0"
           and values["simd"] is False
           and values["max_new"] == "256",
           "resolve_settings reads the engine section")
    expect(values["engine"] == fake, "resolve_settings reads the paths")

    # 2. The launcher's own resolution spawns an engine whose
    # READY reports the INI's configuration.
    events = []
    ready_ev = threading.Event()

    def on_event(kind, payload):
        events.append((kind, payload))
        if kind == "ready":
            ready_ev.set()

    cmd, host, port = nemo_chat.load_ini(argv)
    expect(cmd is not None and host is None and port is None,
           "load_ini builds a local command")
    client = nemo_chat.NemoClient(on_event)
    client.connect_process(cmd)
    ready = wait_ready(events, ready_ev)
    expect(ready is not None and ready.get("mode") == "max",
           "initial READY reports mode=max")
    client.quit()

    # 3. Apply through the panel's code path: change mode,
    # threads, simd, max_new; the INI on disk must carry them
    # and unrelated content must survive.
    raw = {"engine": fake, "checkpoint": ckpt, "tokenizer": tok,
           "mode": "minimum", "threads": "3", "simd": True,
           "max_new": "64", "max_pos": "2048", "host": "", "port": ""}
    errors, result = nemo_chat.apply_settings(raw, ini)
    expect(errors == {}, f"apply accepts valid settings ({errors})")
    with open(ini, "r", encoding="utf-8") as f:
        text = f.read()
    expect("custom_key = keepme" in text
           and "; test settings file" in text,
           "apply preserves unrelated sections, keys and comments")
    values2, _ = nemo_chat.resolve_settings(argv)
    expect(values2["mode"] == "minimum" and values2["threads"] == "3"
           and values2["simd"] is True
           and values2["max_new"] == "64"
           and values2["max_pos"] == "2048",
           "INI on disk carries the applied values")

    # 4. The respawned engine's READY reflects the new values.
    events.clear()
    ready_ev.clear()
    client2 = nemo_chat.NemoClient(on_event)
    client2.connect_process(result["cmd"])
    ready2 = wait_ready(events, ready_ev)
    expect(ready2 is not None and ready2.get("mode") == "minimum",
           "respawned READY reports mode=minimum")
    expect(ready2 is not None and ready2.get("threads") == "3",
           "respawned READY reports threads=3")
    expect(ready2 is not None and ready2.get("simd") == "1",
           "respawned READY reports simd=1")
    expect(ready2 is not None and "cores" in ready2,
           "respawned READY reports the machine's cores")
    client2.quit()

    # 5. Validation: bad fields produce per-field errors and the
    # INI is not touched at all (no partial save).
    with open(ini, "rb") as f:
        before = f.read()
    bad = dict(raw, mode="ludicrous", threads="-2", max_new="0",
               max_pos="wide", host="example", port="70000")
    errors3, result3 = nemo_chat.apply_settings(bad, ini)
    expect(result3 is None, "invalid apply returns no result")
    expect(all(k in errors3 for k in
               ("mode", "threads", "max_new", "max_pos", "port")),
           f"invalid apply names every bad field ({sorted(errors3)})")
    with open(ini, "rb") as f:
        expect(f.read() == before,
               "invalid apply leaves the INI byte-identical")

    # 6. update_ini_text mechanics: a missing key joins its
    # section; a missing section is appended whole.
    sample = "[engine]\nmode = max\n\n[remote]\nhost =\n"
    out = nemo_chat.update_ini_text(
        sample, [("engine", "max_new", "99"),
                 ("paths", "engine", "/bin/e")])
    expect("max_new = 99" in out and "[paths]" in out
           and "/bin/e" in out and out.index("[paths]")
           > out.index("[remote]"),
           "update_ini_text inserts keys and appends sections")

    # 7. CUDA engine options: stored under [cuda], passed as the same
    # command-line flags by the panel's Apply and by the launcher's
    # load_ini, and only to an engine whose name says cuda (the CPU
    # engine would reject them).
    cuda_engine = os.path.join(tmp, "nemo-engine-cuda.py")
    with open(cuda_engine, "w", encoding="utf-8") as f:
        f.write(FAKE_ENGINE)
    os.chmod(cuda_engine, os.stat(cuda_engine).st_mode | stat.S_IEXEC)
    cuda_raw = dict(raw, engine=cuda_engine, simd=False,
                    cuda_pack="packs/nemo.nq2", accel="cuda",
                    expand="never", full_path="exact", host_threads="4",
                    tok_overlap="off", lm_head="int8", prefill_chunk="0",
                    dtc_margin="0.25", dtc_entropy="3.5",
                    dtc_prompt_tokens="64", dtc_complexity="5",
                    dtc_longform="off")
    errors, cres = nemo_chat.apply_settings(cuda_raw, ini)
    expect(errors == {}, f"apply accepts CUDA options ({errors})")
    cmd = cres["cmd"]

    def flag(name):
        return cmd[cmd.index(name) + 1] if name in cmd else None

    expect(flag("--full-path") == "exact" and flag("--host-threads") == "4"
           and flag("--tok-overlap") == "0" and flag("--lm-head") == "int8"
           and flag("--prefill-chunk") == "0" and flag("--accel") == "cuda"
           and flag("--expand") == "never"
           and flag("--dtc-margin") == "0.25"
           and flag("--dtc-entropy") == "3.5"
           and flag("--dtc-prompt-tokens") == "64"
           and flag("--dtc-complexity") == "5"
           and flag("--dtc-longform") == "0",
           "apply maps every CUDA option to its engine flag")
    expect(flag("--cuda-pack") == os.path.normpath(
        os.path.join(tmp, "packs", "nemo.nq2")),
           "the pack path resolves against the INI's folder")
    lcmd, _h, _p = nemo_chat.load_ini(argv)
    expect(lcmd == cmd, "load_ini builds the same command as Apply")
    values3, _ = nemo_chat.resolve_settings(argv)
    expect(values3["full_path"] == "exact"
           and values3["host_threads"] == "4"
           and values3["tok_overlap"] == "off"
           and values3["cuda_pack"] == "packs/nemo.nq2",
           "INI on disk carries the CUDA options (they persist)")
    events.clear()
    ready_ev.clear()
    client3 = nemo_chat.NemoClient(on_event)
    client3.connect_process(cmd)
    expect(wait_ready(events, ready_ev) is not None,
           "an engine started with every CUDA flag reaches READY")
    client3.quit()

    # Blank or "default" passes nothing: the engine's own (fastest)
    # defaults apply. The same options on a CPU engine are not passed.
    blank = dict(cuda_raw, **{k: "" for k in (
        "cuda_pack", "accel", "expand", "full_path", "host_threads",
        "tok_overlap", "lm_head", "prefill_chunk", "dtc_margin",
        "dtc_entropy", "dtc_prompt_tokens", "dtc_complexity")},
        dtc_longform="default")
    errors, bres = nemo_chat.apply_settings(blank, ini)
    expect(errors == {} and not any(a.startswith(("--full-path",
           "--host-threads", "--tok-overlap", "--lm-head",
           "--prefill-chunk", "--cuda-pack", "--accel", "--expand",
           "--dtc")) for a in bres["cmd"]),
           "blank CUDA options pass no flags")
    errors, cpu = nemo_chat.apply_settings(dict(cuda_raw, engine=fake), ini)
    expect(errors == {} and "--full-path" not in cpu["cmd"]
           and "--cuda-pack" not in cpu["cmd"],
           "CUDA options are not passed to a non-CUDA engine")

    with open(ini, "rb") as f:
        before = f.read()
    errors, none = nemo_chat.apply_settings(
        dict(cuda_raw, engine=cuda_engine, full_path="turbo",
             host_threads="9", prefill_chunk="600", tok_overlap="maybe",
             lm_head="fp4", accel="tpu", dtc_margin="soft",
             dtc_prompt_tokens="-1"), ini)
    expect(none is None and all(k in errors for k in (
        "full_path", "host_threads", "prefill_chunk", "tok_overlap",
        "lm_head", "accel", "dtc_margin", "dtc_prompt_tokens")),
           f"invalid CUDA options are named ({sorted(errors)})")
    with open(ini, "rb") as f:
        expect(f.read() == before,
               "invalid CUDA options leave the INI byte-identical")

    print("RESULT:", "FAILURES" if fails else "ALL PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
