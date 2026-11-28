"""Checks the retention battery and graders without a model: every gold
answer passes its own grader, abstentions pass on unanswerable items and
fail on answerable ones, and the instruction rules accept a model-perfect
reply and reject one with trailing text.

Direction by JD. Co-implementation by Cortana, built with Grok Bot
by SpaceX.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grade_retention import grade_item  # noqa: E402

GOOD = {"i01": "YES", "i02": "Apple\nBanana\nCherry", "i03": "Blue", "i04": '{"city": "Paris"}', "i05": "1, 2, 3, 4, 5",
        "i06": "The quick brown fox jumps over the lazy dog.", "i07": "paris", "i08": "Certainly! Red.",
        "i09": "1 2 3", "i10": "Bees make honey!"}


def main():
    b = json.loads((HERE / "battery.json").read_text(encoding="utf-8"))
    bad = 0
    for it in b["items"]:
        if it["set"] == "cand":
            continue
        if it["set"] == "instr":
            g = grade_item(it, GOOD[it["id"]])
            g2 = grade_item(it, GOOD[it["id"]] + "\n\nSure, here is more text.")
            # i08 only asks how to start, so trailing text is allowed there
            want2 = "pass" if it["id"] == "i08" else "fail"
            ok = g["grade"] == "pass" and g2["grade"] == want2 and g2["first_line_ok"]
        elif it["set"] == "calib" and it.get("abstain"):
            ok = grade_item(it, "I don't know")["grade"] == "pass" and grade_item(it, "Paris")["grade"] == "fail"
        elif it["set"] == "calib":
            ok = grade_item(it, it["expect"])["grade"] == "pass" and grade_item(it, "I don't know")["grade"] == "fail"
        else:
            ok = grade_item(it, it["expect"])["grade"] == "pass"
            if it["expect"].lower() not in ("no",) and it["expect"] in it["prompt"].split("Question:")[-1]:
                print("note: gold appears in the question of", it["id"])
        if not ok:
            bad += 1
            print("FAIL", it["id"])
    print("retention grader checks:", "PASS" if bad == 0 else "%d FAIL" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
