#!/usr/bin/env python3
"""Summarize the retention battery: teacher-forced tables from ln_retain's
detail files and generation tables from run_battery's results, written
as one JSON summary and the Markdown tables the results doc uses.

    python littlenemo/tests/retention/analyze.py --tf DIR --gen DIR --out summary.json --md tables.md

Direction by JD. Co-implementation by Cortana, built with Grok Bot
by SpaceX.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

FAILED_ROUND = [4, 5, 8, 14, 31, 37, 38, 39, 41, 42, 44, 45, 48, 49]
ARM_NAMES = {"ref": "raw bf16", "ref_h8": "raw, int8 head", "base": "packed, bf16 head", "base_h8": "packed, int8 head (CUDA lane)",
             "X1": "expanded, bf16 head", "X1_h8": "expanded, int8 head (CUDA lane)",
             "X2": "x2 (4 refine tensors), bf16 head", "X2_h8": "x2, int8 head (CUDA lane)"}


def read_detail(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        f = line.split("\t")
        if len(f) < 14:
            continue
        r = {"cfg": f[0], "id": f[1], "set": f[2], "cat": f[3], "nll": float(f[4]), "first_ok": int(f[5]),
             "all_ok": int(f[6]), "pred0": f[7], "nans": int(f[8]), "rank0": int(f[9]), "lp0": float(f[10]),
             "end_lp": float(f[11]), "end_rank": int(f[12]), "end_top": f[13], "top": []}
        for t in f[14:19]:
            a = t.split(":", 2)
            if len(a) == 3:
                r["top"].append((int(a[0]), float(a[1]), a[2]))
        rows.append(r)
    return rows


def read_runs(path: Path):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        f = line.split("\t")
        d = {"name": f[0]}
        for kv in f[1:]:
            if "=" in kv:
                k, v = kv.split("=", 1)
                d[k] = v
        out[f[0]] = d
    return out


def pct(a, b):
    return "%d/%d" % (a, b)


def ece(pairs, bins=5):
    if not pairs:
        return None
    tot, n = 0.0, len(pairs)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, y) for c, y in pairs if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if sel:
            conf = sum(c for c, _ in sel) / len(sel)
            acc = sum(y for _, y in sel) / len(sel)
            tot += len(sel) / n * abs(conf - acc)
    return round(tot, 4)


def tf_tables(tfdir: Path):
    S = {}
    A = read_detail(tfdir / "A-detail.tsv") + read_detail(tfdir / "F-detail.tsv")
    by = defaultdict(lambda: defaultdict(list))
    for r in A:
        by[r["cfg"]][r["set"]].append(r)
    cfgs = [c for c in ["ref", "ref_h8", "base", "base_h8", "X1", "X1_h8", "X2", "X2_h8"] if c in by]
    S["cfgs"] = cfgs
    sets = ["quiz", "para", "multi", "pass", "calib"]
    lane = {}
    for c in cfgs:
        d = {}
        for s in sets:
            rs = [r for r in by[c][s] if not (s == "calib" and r["id"].startswith("u"))]
            if not rs:
                continue
            d[s] = {"n": len(rs), "first_ok": sum(r["first_ok"] for r in rs), "all_ok": sum(r["all_ok"] for r in rs),
                    "nll_per_tok": round(sum(r["nll"] for r in rs) / max(1, sum(r["nans"] for r in rs)), 4),
                    "top5": sum(1 for r in rs if r["rank0"] < 5),
                    "p_stop_after_answer_mean": round(sum(math.exp(r["end_lp"]) for r in rs) / len(rs), 4),
                    "stop_is_argmax_after_answer": sum(1 for r in rs if r["end_rank"] == 0)}
        allr = [r for s in sets for r in by[c][s] if not r["id"].startswith("u")]
        d["all"] = {"n": len(allr), "first_ok": sum(r["first_ok"] for r in allr), "all_ok": sum(r["all_ok"] for r in allr),
                    "nll_per_tok": round(sum(r["nll"] for r in allr) / max(1, sum(r["nans"] for r in allr)), 4),
                    "top5": sum(1 for r in allr if r["rank0"] < 5),
                    "p_stop_after_answer_mean": round(sum(math.exp(r["end_lp"]) for r in allr) / max(1, len(allr)), 4),
                    "p_stop_after_answer_median": round(sorted(math.exp(r["end_lp"]) for r in allr)[len(allr) // 2], 4) if allr else None,
                    "stop_is_argmax_after_answer": sum(1 for r in allr if r["end_rank"] == 0),
                    "ece_top1": ece([(math.exp(r["top"][0][1]), r["first_ok"]) for r in allr if r["top"]]),
                    "mean_top1_conf": round(sum(math.exp(r["top"][0][1]) for r in allr if r["top"]) / max(1, len(allr)), 4)}
        lane[c] = d
    S["matched"] = lane
    # top-k on the round's failed questions, with the packed lane's own wrong answer as a candidate
    fk = {}
    for c in cfgs:
        rows = {r["id"]: r for s in ("quiz", "cand") for r in by[c][s]}
        lst = []
        for q in FAILED_ROUND:
            g = rows.get("q%02d" % q)
            w = rows.get("c%02d" % q)
            if not g:
                continue
            lst.append({"q": q, "gold_rank": g["rank0"], "gold_p": round(math.exp(g["lp0"]), 4),
                        "top1": g["top"][0][2] if g["top"] else "?", "top1_p": round(math.exp(g["top"][0][1]), 4) if g["top"] else None,
                        "gold_nll": round(g["nll"], 3), "wrong_nll": round(w["nll"], 3) if w else None,
                        "gold_beats_wrong": (w is not None and g["nll"] < w["nll"])})
        fk[c] = lst
    S["failed_topk"] = fk
    # abstention, teacher-forced: "I don't know" on made-up entities
    ab = {}
    for c in cfgs:
        us = [r for r in by[c]["calib"] if r["id"].startswith("u")]
        if us:
            ab[c] = {"n": len(us), "idk_first_token_top1": sum(r["first_ok"] for r in us), "idk_exact": sum(r["all_ok"] for r in us),
                     "idk_nll_mean": round(sum(r["nll"] for r in us) / len(us), 3),
                     "p_I_first_mean": round(sum(math.exp(r["lp0"]) for r in us) / len(us), 4)}
    S["abstain_tf"] = ab
    # paraphrase consistency, teacher-forced
    pc = {}
    for c in cfgs:
        g = defaultdict(list)
        for r in by[c]["para"]:
            g[r["id"].rsplit("_", 1)[0]].append(r["first_ok"])
        pc[c] = {"groups": len(g), "all_three_right": sum(1 for v in g.values() if all(v)),
                 "all_three_wrong": sum(1 for v in g.values() if not any(v)),
                 "mixed": sum(1 for v in g.values() if any(v) and not all(v))}
    S["para_tf"] = pc
    S["runs"] = {k: read_runs(tfdir / ("%s.tsv" % k)) for k in ("A", "B", "C", "D1", "D12", "E", "F", "G")}
    # class sensitivity
    B = read_detail(tfdir / "B-detail.tsv")
    cb = defaultdict(list)
    for r in B:
        cb[r["cfg"]].append(r)
    ref_nll = None
    if "ref" in by:
        ref_rows = [r for s in ("quiz", "cand") for r in by["ref"][s]]
        ref_nll = sum(r["nll"] for r in ref_rows if r["set"] == "quiz")
        ref_first = sum(r["first_ok"] for r in ref_rows if r["set"] == "quiz")
    cls = {}
    base_nll = sum(r["nll"] for r in cb.get("base", []) if r["set"] == "quiz")
    for c, rs in cb.items():
        q = [r for r in rs if r["set"] == "quiz"]
        nll = sum(r["nll"] for r in q)
        rec = None
        if ref_nll is not None and base_nll - ref_nll > 0:
            rec = round((base_nll - nll) / (base_nll - ref_nll), 4)
        cls[c] = {"quiz_nll": round(nll, 3), "quiz_first_ok": sum(r["first_ok"] for r in q), "n": len(q),
                  "gap_recovered": rec, "p_stop_after_answer_mean": round(sum(math.exp(r["end_lp"]) for r in q) / max(1, len(q)), 4)}
    if ref_nll is not None:
        cls["ref (from run A)"] = {"quiz_nll": round(ref_nll, 3), "quiz_first_ok": ref_first, "n": 50, "gap_recovered": 1.0}
    S["classes"] = cls
    # long context
    C = read_detail(tfdir / "C-detail.tsv")
    S["long_tf"] = [{"cfg": r["cfg"], "id": r["id"], "first_ok": r["first_ok"], "all_ok": r["all_ok"], "nll": round(r["nll"], 3),
                     "gold_rank": r["rank0"], "top1": r["top"][0][2] if r["top"] else "?"} for r in C]
    # turn end: stop-token probability right after "gold answer + newline" (run E, set term)
    E = read_detail(tfdir / "E-detail.tsv") + [r for r in read_detail(tfdir / "G-detail.tsv") if r["cfg"] != "base"]
    te = {}
    for c in sorted({r["cfg"] for r in E}):
        rs = [r for r in E if r["cfg"] == c]
        ps = sorted(math.exp(r["end_lp"]) for r in rs)
        from collections import Counter
        te[c] = {"n": len(rs), "p_stop_mean": round(sum(ps) / len(ps), 4), "p_stop_median": round(ps[len(ps) // 2], 4),
                 "stop_is_argmax": sum(1 for r in rs if r["end_rank"] == 0),
                 "stop_in_top5": sum(1 for r in rs if r["end_rank"] < 5),
                 "stop_rank_median": sorted(r["end_rank"] for r in rs)[len(rs) // 2],
                 "argmax_tokens": Counter(r["end_top"] for r in rs).most_common(4)}
    S["turn_end_tf"] = te
    d1, d12 = read_runs(tfdir / "D1.tsv"), read_runs(tfdir / "D12.tsv")
    S["determinism_tf"] = {"t1": d1.get("base", {}).get("hash"), "t12": d12.get("base", {}).get("hash")}
    return S


def gen_tables(gendir: Path):
    G = {}
    for p in sorted(gendir.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        arm = d["arm"]
        rows = [r for g in d["groups"].values() for r in g["items"]]
        by = defaultdict(list)
        for r in rows:
            by[r["set"]].append(r)
        a = {"sets": {}}
        for s, rs in by.items():
            a["sets"][s] = {"n": len(rs), "pass": sum(1 for r in rs if r["grade"] == "pass"),
                            "natural_stop": sum(1 for r in rs if r["termination"]["natural_stop"])}
            if s == "instr":
                a["sets"][s]["first_line_ok"] = sum(1 for r in rs if r.get("first_line_ok"))
            if s == "calib":
                a["sets"][s]["unanswerable_abstained"] = sum(1 for r in rs if r["id"].startswith("u") and r.get("abstained"))
                a["sets"][s]["answerable_abstained"] = sum(1 for r in rs if r["id"].startswith("k") and r.get("abstained"))
                a["sets"][s]["answerable_right"] = sum(1 for r in rs if r["id"].startswith("k") and r["grade"] == "pass")
        # stop probability at the step AFTER the first newline of the answer (where a clean turn ends)
        post = []
        for r in rows:
            k = r["termination"].get("eol_step")
            lg = r.get("logits") or []
            if k is not None and k + 1 < len(lg):
                post.append(lg[k + 1])
        if post:
            ps = sorted(math.exp(x["stop_lp"]) for x in post)
            a["after_newline"] = {"n": len(post), "p_stop_mean": round(sum(ps) / len(ps), 4), "p_stop_median": round(ps[len(ps) // 2], 5),
                                  "stop_is_argmax": sum(1 for x in post if x["stop_rank"] == 0),
                                  "stop_rank_median": sorted(x["stop_rank"] for x in post)[len(post) // 2]}
        eol = [r["termination"].get("eol_stop_lp") for r in rows if r["termination"].get("eol_stop_lp") is not None]
        a["eol_p_stop_mean"] = round(sum(math.exp(x) for x in eol) / len(eol), 4) if eol else None
        a["eol_p_stop_median"] = round(sorted(math.exp(x) for x in eol)[len(eol) // 2], 4) if eol else None
        a["eol_stop_rank_median"] = sorted(r["termination"].get("eol_stop_rank") for r in rows if r["termination"].get("eol_stop_rank") is not None)[len(eol) // 2] if eol else None
        a["natural_stop_all"] = pct(sum(1 for r in rows if r["termination"]["natural_stop"]), len(rows))
        # paraphrase consistency
        pg = defaultdict(list)
        for r in by.get("para", []):
            pg[r["id"].rsplit("_", 1)[0]].append(r["grade"] == "pass")
        a["para"] = {"groups": len(pg), "all_three_right": sum(1 for v in pg.values() if all(v)),
                     "all_three_wrong": sum(1 for v in pg.values() if not any(v)),
                     "mixed": sum(1 for v in pg.values() if any(v) and not all(v))}
        # speeds
        sp = {}
        for cap, g in d["groups"].items():
            sp["short (cap %s)" % cap] = dict(g["summary"], wall_s=g["wall_s"], load_s=g["load_s"],
                                              lane=sorted({r["profile"].get("lane") for r in g["items"]} - {None}))
        if "essays" in d:
            e = d["essays"]
            sp["essays"] = dict(e["summary"], wall_s=e["wall_s"], load_s=e["load_s"],
                                lane=sorted({r["profile"].get("lane") for r in e["items"]} - {None}))
            a["essays"] = {"score": e["score"], "items": [{"id": r["id"], "topic": r.get("topic"), "hit": r["required_hit"],
                                                           "n": r["required_n"], "grade": r["grade"],
                                                           "natural_stop": r["termination"]["natural_stop"],
                                                           "gen_tokens": r["profile"].get("gen_tokens")} for r in e["items"]]}
        a["speeds"] = sp
        a["wall_s"] = d.get("wall_s")
        a["texts"] = {r["id"]: r["text"] for r in rows}
        G[arm] = a
    # determinism across host threads
    if "packed" in G and "packed-det-ht1" in G:
        t0, t1 = G["packed"]["texts"], G["packed-det-ht1"]["texts"]
        common = [k for k in t1 if k in t0]
        G["_determinism"] = {"items": len(common), "identical": sum(1 for k in common if t0[k] == t1[k]),
                             "differ": [k for k in common if t0[k] != t1[k]]}
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", type=Path, required=True)
    ap.add_argument("--gen", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    S = {"tf": tf_tables(a.tf), "gen": gen_tables(a.gen)}
    for arm in S["gen"].values():
        if isinstance(arm, dict):
            arm.pop("texts", None)
    a.out.write_text(json.dumps(S, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(S, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
