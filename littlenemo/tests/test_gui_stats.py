#!/usr/bin/env python3
"""test_gui_stats.py - headless tests of the stats panel model, the
thinking-line headline, and v3 parsing (gui/nemo_stats.py,
nemo_chat.parse_line). Uses a fake clock, so timings are exact."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gui"))
import nemo_chat  # noqa: E402
import nemo_stats  # noqa: E402

fails = []
D = nemo_stats.DASH


def expect(cond, name):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def rows_dict(model, now):
    return {sec: dict(items) for sec, items in model.rows(now)}


def main():
    P = nemo_chat.parse_line
    # backward compatibility: v2 lines parse exactly as before
    expect(P("TICK mode=thinking layer=3 next=4 active=1 staged=2 tok=5 "
             "ms_tok=210.0 gbs=84.8\n") ==
           ("tick", {"mode": "thinking", "layer": 3, "next": 4, "active": 1,
                     "staged": 2, "tok": 5, "ms_tok": 210.0, "gbs": 84.8}),
           "v2 TICK unchanged")
    k, p = P("READY nemo-engine 2 model=m weights=raw layers=56 cores=12 "
             "threads=11 mode=max simd=0\n")
    expect(k == "ready" and p["threads"] == "11" and p["layers"] == "56",
           "v2 READY values stay strings")
    expect(P("THINK a\\nb \n") == ("think", "a\nb "), "THINK unescaped")
    expect(P("DONE\n") == ("done", None), "DONE")
    k, p = P("STATS tokens=3 ms=9.0 tps=1.5 ms_per_tok=3.0 gbs_avg=1.0 "
             "peak_mb=5.0 ttft_ms=12.5 prompt_tokens=7 stopped=1\n")
    expect(k == "stats" and p["tokens"] == 3 and p["ttft_ms"] == 12.5
           and p["prompt_tokens"] == 7 and p["stopped"] == 1,
           "STATS with v3 keys typed")
    # robustness: malformed lines are raw, not exceptions
    expect(P("LAYER x mamba\n")[0] == "raw", "malformed LAYER -> raw")
    expect(P("LAYER\n")[0] == "raw", "truncated LAYER -> raw")
    expect(P("X" * (nemo_chat.MAX_LINE + 5))[0] == "raw", "oversize line -> raw")
    expect(P("SOMETHING new=1\n") == ("raw", "SOMETHING new=1"),
           "unknown verb -> raw")
    k, p = P("HW cpu=12th%20Gen%20i5 cores=8 threads=12 ram_mb=16000 "
             "gpu=RTX%203050 vram_mb=4096 accel=cuda\n")
    expect(k == "hw" and p["cpu"] == "12th Gen i5" and p["cores"] == 8
           and p["accel"] == "cuda", "HW decoded and typed")
    k, p = P("DTC layer=4 tensor=layers.4.mlp rows=2688 cols=4096 k=256 "
             "block=8 active=190 lane=cuda state=begin\n")
    expect(k == "dtc" and p["rows"] == 2688 and p["lane"] == "cuda",
           "DTC parsed")
    expect(P("HS pass=2 ordinal=112 hits=10 misses=1 stall_ms=1.5\n")[1]
           ["stall_ms"] == 1.5, "HS parsed")
    expect(P("MEM ram_mb=nan\n")[1]["ram_mb"] == "nan", "nan stays text")

    # headline
    H = nemo_stats.thinking_headline
    expect(H("") == "", "empty headline")
    expect(H("First idea. Second idea is the one we want.") ==
           "Second idea is the one we want.", "latest sentence")
    expect(H("Checking the weights carefully. Then") ==
           "Checking the weights carefully. Then", "fragment joins previous")
    long = "word " * 80
    h = H(long, 40)
    expect(len(h) <= 40 and h.startswith("\u2026"), "long headline trimmed")
    expect(H("a\n\n  b   c") == "a b c", "whitespace collapsed")

    # stats model with a fake clock
    m = nemo_stats.StatsModel(clock=lambda: 0.0)
    m.probe = {"cpu": "Probe CPU", "cores": 8, "threads": 12, "ram_mb": 16000,
               "gpu": "Probe GPU", "vram_mb": 4096, "driver": "1.0",
               "igpu": "Intel UHD"}
    m.on_event("ready", {"model": "x", "layers": "56", "cores": "12",
                         "threads": "11", "mode": "max", "simd": "1"}, 0.0)
    r = rows_dict(m, 0.0)
    expect(r["Throughput"]["Time to first token"] == D, "no TTFT before a turn")
    expect(r["Hardware"]["CPU"] == "Probe CPU  (probe)", "probe CPU labeled")
    expect(r["Hardware"]["Accel lane"].startswith("CPU + AVX2"),
           "lane from READY simd")
    m.turn_begin(now=10.0)
    m.on_event("prefill", {"tokens": 20, "ms": 400.0, "tps": 50.0}, 10.4)
    m.on_event("think", "hmm", 10.5)
    m.on_event("layer", {"idx": 3, "type": "mamba"}, 10.5)
    m.on_event("tick", {"mode": "thinking", "layer": 3, "next": 4,
                        "active": 1, "staged": 2, "tok": 1, "ms_tok": 100.0,
                        "gbs": 2.0}, 10.5)
    for i in range(2, 6):
        m.on_event("tok", "x", 10.5 + 0.25 * (i - 1))
        m.on_event("tick", {"mode": "answering", "tok": i, "ms_tok": 250.0},
                   10.5 + 0.25 * (i - 1))
    m.on_event("expand", {"layer": 3, "tensor": "w.in", "state": "begin",
                          "bytes": 2097152}, 11.0)
    m.on_event("dtc", {"layer": 3, "tensor": "w.out", "rows": 2688,
                       "lane": "cuda", "state": "begin"}, 11.0)
    m.on_event("hs", {"pass": 5, "ordinal": 280, "hits": 9, "misses": 1,
                      "stall_ms": 2.0, "dtc": 4, "dtc_ms": 7.5,
                      "custom_ctr": 3}, 11.0)
    m.on_event("hw", {"gpu": "Engine GPU", "accel": "opencl-igpu"}, 11.0)
    m.on_event("mem", {"ram_mb": 512, "vram_mb": 166}, 11.0)
    m.on_event("stat", {"kv_cache_tokens": 40}, 11.0)
    m.on_event("moe", {"layer": 3, "experts": "1,2,3,4", "k": 4, "of": 512},
               11.0)
    r = rows_dict(m, 11.5)
    expect(r["Throughput"]["Time to first token"] == "500 ms  (gui)",
           "TTFT measured by the GUI")
    expect(r["Throughput"]["Prompt"].startswith("50.0 tok/s"),
           "prompt tok/s from PREFILL")
    expect(r["Throughput"]["Generation"] == "4.00 tok/s  (gui)",
           "live generation tok/s")
    expect(r["Throughput"]["Tokens (think / answer)"] == "1 / 4",
           "thinking vs answer tokens counted separately")
    expect(r["Hardware"]["GPU"] == "Engine GPU", "engine HW overrides probe")
    expect(r["Hardware"]["Accel lane"] == "OpenCL (integrated GPU)",
           "accel lane from engine")
    expect("w.in" in r["Now"]["Expanding"] and "2.00 MB" in r["Now"]["Expanding"],
           "expanding tensor shown")
    expect("w.out" in r["Now"]["In DTC"] and "cuda" in r["Now"]["In DTC"],
           "DTC tensor shown")
    expect(r["Now"]["MoE experts"].startswith("layer 3: 1,2,3,4"), "MoE shown")
    expect("90.0%" in r["Tensor Handshake"]["Prefetch hits / misses"],
           "hit rate computed")
    expect(r["Tensor Handshake"]["custom_ctr"] == "3",
           "unknown HS counters still shown")
    expect(r["Memory"]["RAM in use"] == "512 MB" and
           r["Memory"]["VRAM in use"] == "166 MB", "memory shown")
    expect(r["Other engine stats"]["kv_cache_tokens"] == "40",
           "other STAT keys shown")
    m.on_event("dtc", {"layer": 3, "tensor": "w.out", "state": "end"}, 11.6)
    expect("(finished)" in rows_dict(m, 11.6)["Now"]["In DTC"],
           "DTC end marked finished")
    m.on_event("stats", {"tokens": 5, "ms": 1500.0, "tps": 3.3,
                         "ttft_ms": 480.0, "gen_tps": 4.1}, 12.0)
    m.on_event("done", None, 12.0)
    r = rows_dict(m, 13.0)
    expect(r["Throughput"]["Time to first token"] == "480 ms",
           "engine TTFT preferred after STATS")
    expect(r["Throughput"]["Generation"] == "4.10 tok/s",
           "engine gen tok/s after STATS")
    expect(r["Now"]["Mode"] == "idle", "idle after DONE")
    expect(r["Last turn (STATS)"]["ttft_ms"] == "480", "STATS listed")
    m.on_event("ready", {"model": "y"}, 14.0)
    r = rows_dict(m, 14.0)
    expect(r["Tensor Handshake"]["Pass / ordinal"] == D,
           "a new READY clears engine stats")
    expect(r["Hardware"]["CPU"] == "Probe CPU  (probe)",
           "probe survives an engine restart")

    print("RESULT:", "FAILURES" if fails else "ALL PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
