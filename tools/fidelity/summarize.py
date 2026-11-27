"""Collect fidelity-lab results into one Markdown table.

Reads data/fidelity/runs/NAME.tsv (fid-sigma summaries) and
NAME.quant.tsv (fid-quant rates) for every NAME given, or for every
.tsv run present, and prints a table: rate (scope and complete bits per
weight), top-1 agreement and mean KL against the raw model per gamma
and overall, and cross-entropy against the true tokens.
"""
import os, re, sys

RUNS = os.path.join("data", "fidelity", "runs")

def summary(name):
    p = os.path.join(RUNS, name + ".tsv")
    if not os.path.exists(p):
        return None
    out = {"g": {}}
    for line in open(p, encoding="utf-8", errors="replace"):
        m = re.match(r"# summary gamma (\S+): positions (\d+) top1 (\S+) top5 (\S+) kl_mean (\S+) kl_max (\S+) ce_ref (\S+) ce_q (\S+).* acc_ref (\S+) acc_q (\S+)", line)
        if m:
            g = m.group(1)
            out["g"][g] = dict(n=int(m.group(2)), top1=float(m.group(3)), top5=float(m.group(4)), kl=float(m.group(5)),
                               klmax=float(m.group(6)), ce_ref=float(m.group(7)), ce_q=float(m.group(8)),
                               acc_ref=float(m.group(9)), acc_q=float(m.group(10)))
        m = re.match(r"# logits digest (\S+)", line)
        if m:
            out["digest"] = m.group(1)
    if not out["g"]:
        return None
    n = sum(v["n"] for v in out["g"].values())
    out["n"] = n
    for k in ("top1", "top5", "kl", "ce_ref", "ce_q", "acc_ref", "acc_q"):
        out[k] = sum(v[k] * v["n"] for v in out["g"].values()) / n
    return out

def rate(name):
    p = os.path.join(RUNS, name + ".quant.tsv")
    if not os.path.exists(p):
        return None, None
    scope = full = None
    for line in open(p, encoding="utf-8", errors="replace"):
        m = re.search(r"scope bpw (\S+)", line)
        if m:
            scope = float(m.group(1))
        m = re.search(r"complete bpw with them at 16: (\S+)", line)
        if m:
            full = float(m.group(1))
    return scope, full

def main():
    names = sys.argv[1:]
    if not names:
        names = sorted(f[:-4] for f in os.listdir(RUNS) if f.endswith(".tsv") and not f.endswith(".quant.tsv"))
    gams = None
    rows = []
    for nm in names:
        s = summary(nm)
        if not s:
            continue
        if gams is None:
            gams = list(s["g"].keys())
        sc, fu = rate(nm)
        rows.append((nm, sc, fu, s))
    if not rows:
        print("no results")
        return
    hdr = "| run | scope bpw | complete bpw | positions | top-1 | top-5 | KL mean | " + " | ".join(
        f"top-1 / KL at gamma {g}" for g in gams) + " | CE quant (raw) |"
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    for nm, sc, fu, s in rows:
        cells = [nm, f"{sc:.4f}" if sc else "-", f"{fu:.4f}" if fu else "-", str(s["n"]),
                 f"{s['top1']:.4f}", f"{s['top5']:.4f}", f"{s['kl']:.4f}"]
        for g in gams:
            v = s["g"].get(g)
            cells.append(f"{v['top1']:.4f} / {v['kl']:.4f}" if v else "-")
        cells.append(f"{s['ce_q']:.4f} ({s['ce_ref']:.4f})")
        print("| " + " | ".join(cells) + " |")

if __name__ == "__main__":
    main()
