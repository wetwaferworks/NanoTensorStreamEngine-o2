#!/usr/bin/env python3
"""Build the scrubbed receipts for the retention battery: generation answers with grades and speeds
(no logit dumps, no engine stderr, no local paths), teacher-forced per-item scores (no top-k vectors),
the run tables with their hashes, the analyzer summary and the hand-graded essay notes.

    python littlenemo/tests/retention/make_receipts.py --data DIR --sem DIR --out DIR

Direction by JD. Co-implementation by Cortana, built with Grok Bot by SpaceX.
"""
import argparse
import json
import re
from pathlib import Path


def scrub(s):
    s = re.sub(r"[A-Za-z]:\\[^\s'\"]*", "<path>", s)
    s = re.sub(r"/(?:home|workspace|Users)/[^\s'\"]*", "<path>", s)
    return s


def clean_text(t, limit):
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t[:limit] + (" [...]" if len(t) > limit else "")


def slim_item(r, limit):
    keep = {k: r[k] for k in ("id", "set", "prompt", "gold", "first_line", "grade", "topic", "required_n", "required_hit", "abstained", "first_line_ok") if k in r}
    keep["answer"] = clean_text(r.get("text", ""), limit)
    t = r.get("termination", {})
    keep["termination"] = {k: v for k, v in t.items() if k in ("natural_stop", "gen_steps", "eol_step", "eol_stop_lp", "eol_stop_rank")}
    lg, k = r.get("logits") or [], t.get("eol_step")
    if k is not None and k + 1 < len(lg):
        keep["stop_after_newline"] = {"stop_lp": lg[k + 1]["stop_lp"], "stop_rank": lg[k + 1]["stop_rank"]}
    p = r.get("profile", {})
    keep["profile"] = {x: p.get(x) for x in ("prompt_tokens", "prefill_tps", "gen_tokens", "decode_tps", "ram_peak_mb", "vram_used_peak_mib", "lane", "expand_steps", "dtc_steps") if x in p}
    if r.get("present"):
        keep["facts_found"] = [f if isinstance(f, str) else f.get("fact") for f in r["present"]]
    if r.get("missing"):
        keep["facts_missed"] = [f if isinstance(f, str) else f.get("fact") for f in r["missing"]]
    return keep


def slim_arm(d):
    out = {"arm": d["arm"], "started": d.get("started"), "wall_s": d.get("wall_s"), "topk": d.get("topk"), "groups": {}}
    for cap, g in d["groups"].items():
        out["groups"][cap] = {"command": [scrub(c) for c in g["command"]], "ready": g["ready"], "hw": g["hw"], "max_new": g["max_new"],
                              "load_s": g["load_s"], "launches": g["launches"], "pauses": g["pauses"], "wall_s": g["wall_s"], "gpu": g["gpu"],
                              "summary": g["summary"], "items": [slim_item(r, 400) for r in g["items"]]}
    if "essays" in d:
        e = d["essays"]
        out["essays"] = {"command": [scrub(c) for c in e["command"]], "ready": e["ready"], "hw": e["hw"], "max_new": e["max_new"], "load_s": e["load_s"],
                         "wall_s": e["wall_s"], "gpu": e["gpu"], "summary": e["summary"], "score": e["score"], "items": [slim_item(r, 1800) for r in e["items"]]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--sem", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for p in sorted((a.data / "gen").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        (a.out / ("gen-%s.json" % d["arm"])).write_text(json.dumps(slim_arm(d), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    sn = a.data / "gen-sn.json"
    if sn.exists():
        d = json.loads(sn.read_text(encoding="utf-8"))
        q = d["questions"]
        q = dict(q, command=[scrub(c) for c in q["command"]], items=[slim_item(r, 400) for r in q["items"]])
        (a.out / "gen-supernemo-subset.json").write_text(json.dumps({"started": d.get("started"), "wall_s": d.get("wall_s"), "questions": q}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for p in sorted(a.sem.glob("*.json")):
        (a.out / ("semantic-%s.json" % p.stem)).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    # teacher-forced: run tables (hashes) and per-item scores without the top-k vectors
    for run in ("A", "B", "C", "D1", "D12", "E", "F", "G"):
        for suffix in ("", "-detail"):
            src = a.data / "tf" / ("%s%s.tsv" % (run, suffix))
            if not src.exists() or src.stat().st_size == 0:
                continue
            if suffix == "":
                (a.out / ("tf-%s.tsv" % run)).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            else:
                rows = ["cfg\tid\tset\tcat\tnll\tfirst_ok\tall_ok\tgold_rank\tend_lp\tend_rank"]
                for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
                    f = line.split("\t")
                    if len(f) >= 13:
                        rows.append("\t".join([f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[9], f[11], f[12]]))
                (a.out / ("tf-%s-items.tsv" % run)).write_text("\n".join(rows) + "\n", encoding="utf-8")
    sm = a.data / "summary_new.json"
    if sm.exists():
        (a.out / "summary.json").write_text(sm.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
