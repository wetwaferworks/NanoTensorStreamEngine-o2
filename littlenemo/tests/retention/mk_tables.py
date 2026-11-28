#!/usr/bin/env python3
"""Render the Markdown tables for docs/RETENTION-BATTERY-2026-11-27.md from the analyzer summary,
the SuperNemo subset run and the teacher-forced run logs. Prints one block per table.

Direction by JD. Co-implementation by Cortana, built with Grok Bot by SpaceX.
"""
import json, sys, statistics
from pathlib import Path

D = Path(sys.argv[1])  # directory with summary_new.json, gen-sn.json and ../sem/
S = json.loads((D / "summary_new.json").read_text())
TF, GEN = S["tf"], S["gen"]
NAMES = {"ref": "raw bf16", "ref_h8": "raw bf16, int8 head", "base": "packed", "base_h8": "packed, int8 head",
         "X1": "expanded x1 (6 layers)", "X1_h8": "x1, int8 head", "X2": "expanded x2 (2 layers, 4 tensors)", "X2_h8": "x2, int8 head", "cls_mamba": "packed + original Mamba in/out_proj", "cls_mlp": "packed + original MLP up/down_proj"}
ORDER = ["ref", "ref_h8", "base", "base_h8", "X1", "X1_h8", "X2", "X2_h8"]
out = {}


def tab(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


cf = [c for c in ORDER if c in TF["matched"]]
rows = []
for c in cf:
    m = TF["matched"][c]
    def f(s, k="first_ok"):
        return "%d/%d" % (m[s][k], m[s]["n"]) if s in m else "-"
    rows.append([NAMES[c], f("quiz"), f("quiz", "all_ok"), f("para"), f("multi"), f("pass"), f("calib"), "%d/%d" % (m["all"]["first_ok"], m["all"]["n"]),
                 m["all"]["nll_per_tok"], "%d/%d" % (m["all"]["top5"], m["all"]["n"]), m["all"]["ece_top1"], m["all"]["mean_top1_conf"]])
out["tf_matched"] = tab(["arm", "quiz first token", "quiz whole answer", "paraphrase", "multi-step", "passage", "calibration (answerable)", "all first token", "NLL / token", "gold in top 5", "ECE", "mean top-1 confidence"], rows)

rows = []
for a, b in (("ref", "ref_h8"), ("base", "base_h8"), ("X1", "X1_h8"), ("X2", "X2_h8")):
    if a in TF["matched"] and b in TF["matched"]:
        x, y = TF["matched"][a]["all"], TF["matched"][b]["all"]
        rows.append([NAMES[a], "%d/%d" % (x["first_ok"], x["n"]), x["nll_per_tok"], "%d/%d" % (y["first_ok"], y["n"]), y["nll_per_tok"], round(y["nll_per_tok"] - x["nll_per_tok"], 4)])
out["head"] = tab(["weights", "bf16-class head: first token", "NLL/tok", "int8 per-row head: first token", "NLL/tok", "NLL change"], rows)

rows = []
for c, v in TF["classes"].items():
    rows.append([c, "%d/%d" % (v["quiz_first_ok"], v["n"]), v["quiz_nll"], "-" if v["gap_recovered"] is None else "%.0f%%" % (100 * v["gap_recovered"])])
out["classes"] = tab(["restored to original bf16", "quiz first token", "quiz NLL (50 items)", "share of the packed-to-raw gap closed"], rows)

fk = TF["failed_topk"]
rows = []
for c in [c for c in ("ref", "base", "X1", "X2") if c in fk]:
    f = fk[c]
    rows.append([NAMES[c], "%d/%d" % (sum(1 for x in f if x["gold_rank"] == 0), len(f)), "%d/%d" % (sum(1 for x in f if x["gold_rank"] < 5), len(f)),
                 sorted(x["gold_rank"] for x in f)[len(f) // 2], "%d/%d" % (sum(1 for x in f if x["gold_beats_wrong"]), len(f))])
out["failed_summary"] = tab(["arm", "gold is top-1", "gold in top 5", "median gold rank", "gold more likely than the packed lane's wrong answer"], rows)
rows = []
for i, q in enumerate([x["q"] for x in fk["base"]]):
    r = [q]
    for c in ("ref", "base", "X1", "X2"):
        if c in fk:
            x = fk[c][i]
            r.append("rank %d (p=%.3f), top-1 '%s'" % (x["gold_rank"], x["gold_p"], x["top1"]))
    rows.append(r)
out["failed_detail"] = tab(["quiz item"] + [NAMES[c] for c in ("ref", "base", "X1", "X2") if c in fk], rows)

te = TF.get("turn_end_tf", {})
rows = []
for c in ("ref", "base", "X1", "X2", "cls_mamba", "cls_mlp"):
    if c in te:
        v = te[c]
        rows.append([NAMES[c], v["n"], v["p_stop_mean"], v["p_stop_median"], "%d/%d" % (v["stop_is_argmax"], v["n"]), "%d/%d" % (v["stop_in_top5"], v["n"]), v["stop_rank_median"], "; ".join("%s x%d" % (t.replace("\n", "\\n"), n) for t, n in v["argmax_tokens"][:3])])
out["turn_end_tf"] = tab(["arm", "items", "mean P(stop)", "median P(stop)", "stop is top-1", "stop in top 5", "median stop rank", "what is top-1 instead"], rows)

rows = []
for arm in ("packed", "packed-det-ht1", "expanded-x2"):
    if arm in GEN:
        a = GEN[arm]
        an = a.get("after_newline", {})
        rows.append([arm, a["natural_stop_all"], an.get("n"), an.get("p_stop_mean"), an.get("p_stop_median"), "%d/%d" % (an.get("stop_is_argmax", 0), an.get("n", 0)), an.get("stop_rank_median")])
out["turn_end_gen"] = tab(["generation arm", "answers that ended on their own (of cap-limited runs)", "answers with a newline", "mean P(stop) after the newline", "median", "stop is top-1", "median stop rank"], rows)

rows = []
for arm in ("packed", "packed-det-ht1", "expanded-x2"):
    if arm not in GEN:
        continue
    s = GEN[arm]["sets"]
    g = lambda k, f="pass": ("%d/%d" % (s[k][f], s[k]["n"])) if k in s else "-"
    rows.append([arm, g("quiz"), g("para"), g("multi"), g("pass"), g("calib"), g("instr"), g("instr", "first_line_ok"), g("long")])
out["gen_sets"] = tab(["arm (CUDA, greedy)", "quiz", "paraphrase", "multi-step", "passage", "calibration", "instructions (full)", "instructions (first line only)", "long context"], rows)

rows = []
for arm in ("packed", "expanded-x2"):
    if arm in GEN and "calib" in GEN[arm]["sets"]:
        c = GEN[arm]["sets"]["calib"]
        rows.append([arm, "%d/8" % c["unanswerable_abstained"], "%d/8" % c["answerable_right"], "%d/8" % c["answerable_abstained"]])
out["calib_gen"] = tab(["generation arm", "made-up questions answered \"I don't know\"", "real questions answered right", "real questions wrongly refused"], rows)
rows = []
for c in cf:
    if c in TF.get("abstain_tf", {}):
        v = TF["abstain_tf"][c]
        rows.append([NAMES[c], "%d/%d" % (v["idk_first_token_top1"], v["n"]), v["p_I_first_mean"], v["idk_nll_mean"]])
out["calib_tf"] = tab(["arm", "\"I don't know\" is the top first token", "mean P(first token I)", "mean NLL of the full phrase"], rows)

rows = []
for c in cf:
    if c in TF.get("para_tf", {}):
        v = TF["para_tf"][c]
        rows.append([NAMES[c], v["groups"], v["all_three_right"], v["mixed"], v["all_three_wrong"]])
out["para_tf"] = tab(["arm (teacher-forced first token)", "groups of 3", "all three right", "mixed", "all three wrong"], rows)
rows = []
for arm in ("packed", "expanded-x2"):
    if arm in GEN:
        v = GEN[arm]["para"]
        rows.append([arm, v["groups"], v["all_three_right"], v["mixed"], v["all_three_wrong"]])
out["para_gen"] = tab(["generation arm", "groups of 3", "all three right", "mixed", "all three wrong"], rows)

rows = []
sem = {}
for arm in ("packed", "expanded-x2"):
    p = D.parent / "sem" / ("%s.json" % arm)
    if p.exists():
        sem[arm] = {i["id"]: i for i in json.loads(p.read_text())["items"]}
if sem:
    ids = sorted(next(iter(sem.values())))
    for i in ids:
        r = [i, next(iter(sem.values()))[i]["topic"]]
        for arm in ("packed", "expanded-x2"):
            if arm in GEN and arm in sem:
                kw = [e for e in GEN[arm]["essays"]["items"] if e["id"] == i][0]
                r += ["%d/%d %s" % (kw["hit"], kw["n"], kw["grade"]), sem[arm][i]["semantic"], len(sem[arm][i]["errors"])]
        rows.append(r)
    tot = ["", "**passes**"]
    for arm in ("packed", "expanded-x2"):
        if arm in GEN and arm in sem:
            tot += ["%d/10" % sum(1 for e in GEN[arm]["essays"]["items"] if e["grade"] == "pass"), "%d/10" % sum(1 for v in sem[arm].values() if v["semantic"] == "pass"), sum(len(v["errors"]) for v in sem[arm].values())]
    rows.append(tot)
    out["essays"] = tab(["essay", "topic", "packed: keyword grader", "packed: semantic", "packed: false claims", "x2: keyword grader", "x2: semantic", "x2: false claims"], rows)

rows = []
for r in TF["long_tf"]:
    pass
bycfg = {}
for r in TF["long_tf"]:
    bycfg.setdefault(r["cfg"], []).append(r)
for c in [c for c in ORDER if c in bycfg]:
    v = bycfg[c]
    rows.append([NAMES[c], "%d/4" % sum(x["first_ok"] for x in v), "%d/4" % sum(x["all_ok"] for x in v), round(sum(x["nll"] for x in v) / 4, 3), ", ".join(str(x["gold_rank"]) for x in v), ", ".join("'%s'" % x["top1"] for x in v)])
out["long_tf"] = tab(["arm", "first token", "whole answer", "mean NLL", "gold ranks", "top-1 instead"], rows)

# speeds
rows = []
def med(d):
    return d["median"] if isinstance(d, dict) and "median" in d else d
for arm in ("packed", "packed-det-ht1", "expanded-x2"):
    if arm not in GEN:
        continue
    for k, v in GEN[arm]["speeds"].items():
        pre, dec = v["prefill_tps"], v["decode_tps"]
        rows.append(["Little Nemo " + arm, k, "CUDA" if "cuda" in (v.get("lane") or []) else (v.get("lane") or "?"), "%.1f / %.1f" % (pre["median"], pre["mean"]), "%.1f / %.1f" % (dec["median"], dec["mean"]),
                     "%.1f" % v.get("decode_tps_pooled", dec["mean"]), "%.0f" % v["wall_s"], v.get("vram_card_peak_mib"), v.get("ram_peak_mb"),
                     "%s" % ("meets 10 tok/s floor" if dec["median"] >= 10 else "BELOW 10 tok/s floor") + ("; below 30 tok/s target" if dec["median"] < 30 else "")])
sn = json.loads((D / "gen-sn.json").read_text())
q = sn["questions"]
pre = [i["profile"]["prefill_tps"] for i in q["items"] if i["profile"].get("prefill_tps")]
dec = [i["profile"]["decode_tps"] for i in q["items"] if i["profile"].get("decode_tps")]
rows.append(["SuperNemo (120B MoE) subset, 24 tokens", "short (%d items)" % len(q["items"]), "CUDA (sn-engine)", "%.2f / %.2f" % (statistics.median(pre), statistics.mean(pre)), "%.2f / %.2f" % (statistics.median(dec), statistics.mean(dec)),
             "-", "%.0f" % q["wall_s"], q["gpu"]["vram_used_peak_mib"], max(i["profile"].get("ram_peak_mb", 0) for i in q["items"]), "BELOW 10 tok/s floor; below 30 tok/s target"])
out["speeds_gen"] = tab(["run", "group", "path", "prefill tok/s (median / mean)", "decode tok/s (median / mean)", "decode pooled", "wall s", "card peak MiB", "RAM peak MB", "against the targets"], rows)

tfr = []
tokens = {"A": 5387, "B": 1927, "C": 3449, "D1": 198, "D12": 198, "E": 1602, "F": 5387, "G": 1602}
for k in ("A", "B", "C", "D1", "D12", "E", "F", "G"):
    r = TF["runs"].get(k, {})
    for name, v in r.items():
        if "sec" in v:
            head_only = float(v["sec"]) < 60 and k not in ("D1", "D12")
            tfr.append([k, name, tokens[k], v["sec"], "n/a (head only, layer outputs reused)" if head_only else "%.2f" % (tokens[k] / float(v["sec"])), v.get("hash", "")])
out["speeds_tf"] = tab(["run", "arm", "tokens scored", "seconds", "teacher-forced tok/s (CPU)", "hash"], tfr)


# per-configuration summary: accuracy next to speed (JD picks the best configuration from this)
def tfall(c):
    if c not in TF["matched"]:
        return ("pending", "pending")
    a = TF["matched"][c]["all"]
    return ("%d/%d (%.0f%%)" % (a["first_ok"], a["n"], 100.0 * a["first_ok"] / a["n"]), a["nll_per_tok"])
def tfq(c):
    return "%d/50" % TF["matched"][c]["quiz"]["first_ok"] if c in TF["matched"] else "pending"
def genrow(arm):
    a = GEN[arm]; s = a["sets"]; sp = a["speeds"]["short (cap 48)"]
    sem_p = D.parent / "sem" / ("%s.json" % arm)
    ess = a.get("essays", {}).get("score", {}).get("passes")
    sem_n = sum(1 for i in json.loads(sem_p.read_text())["items"] if i["semantic"] == "pass") if sem_p.exists() else None
    return {"quiz": "%d/50" % s["quiz"]["pass"], "multi": "%d/12" % s["multi"]["pass"], "ess": "%s/10 keyword, %s/10 read by hand" % (ess, sem_n) if ess is not None else "-",
            "stop": a["natural_stop_all"], "dec": "%.1f" % sp["decode_tps"]["median"], "pre": "%.1f" % sp["prefill_tps"]["median"], "vram": sp.get("vram_card_peak_mib")}
rows = []
def cfg_row(label, tfc, gen, where, note=""):
    t = tfall(tfc) if tfc else ("-", "-")
    tq = tfq(tfc) if tfc else "-"
    if gen and gen in GEN:
        g = genrow(gen)
        rows.append([label, where, t[0], t[1], tq, g["quiz"], g["multi"], g["ess"], g["stop"], g["dec"], g["pre"], g["vram"], note])
    else:
        rows.append([label, where, t[0], t[1], tq, "-", "-", "-", "-", "-", "-", "-", note])
cfg_row("raw bf16 (original)", "ref", None, "CPU, teacher-forced only", "no speed on the card: 18 GB, does not fit")
cfg_row("raw bf16, int8 head", "ref_h8", None, "CPU, teacher-forced only", "")
cfg_row("packed, int8 head (the shipping CUDA lane)", "base_h8", "packed", "CUDA 4 GB", "meets the 10 tok/s floor")
cfg_row("packed, bf16 head", "base", None, "CPU only (head 1,120 MiB does not fit)", "no gain over int8 head")
cfg_row("x1 expanded, 6 layers, int8 head", "X1_h8", None, "CPU, teacher-forced only", "does not fit the 4 GB card")
cfg_row("x1 expanded, bf16 head", "X1", None, "CPU, teacher-forced only", "")
cfg_row("x2 expanded, 4 tensors, int8 head", "X2_h8" if "X2_h8" in TF["matched"] else "X2", "expanded-x2", "CUDA 4 GB", "meets the 10 tok/s floor; fits the card" + ("" if "X2_h8" in TF["matched"] else "; TF columns are the bf16-head run (int8-head run pending)"))
if "X2" in TF["matched"] and "X2_h8" in TF["matched"]:
    cfg_row("x2 expanded, bf16 head", "X2", None, "CPU, teacher-forced only", "")
rows.append(["SuperNemo (120B MoE) subset", "CUDA (sn-engine)", "-", "-", "-", "5/5 on 5 questions (earlier round)", "-", "-", "-", "0.17", "0.58", q["gpu"]["vram_used_peak_mib"], "BELOW the 10 tok/s floor"])
out["config_summary"] = tab(["configuration", "where it runs", "first-token hits, all 119 items (teacher-forced)", "NLL per token", "quiz first token (TF)", "quiz pass (generated)", "multi-step (generated)", "essays (generated)", "answers that stop on their own", "decode tok/s (median)", "prefill tok/s (median)", "card peak MiB", "against the 10 tok/s floor"], rows)

for k, v in out.items():
    print("<!--%s-->\n%s\n" % (k, v))
(D / "tables.json").write_text(json.dumps(out, indent=1))
