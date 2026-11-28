#!/usr/bin/env python3
"""Build the NanoTensor retention battery: battery.json (every item, for
the generation runner and graders) and probe.tsv (the teacher-forced
items for ln_retain). Deterministic: the long-context haystacks come
from a fixed seed, so rebuilding gives byte-identical files.

    python littlenemo/tests/retention/build_battery.py

Direction by JD. Co-implementation by Cortana, built with Grok Bot
by SpaceX.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUIZ = HERE.parent / "quiz"
JUST = " Answer with just the answer."

items = []


def add(id_, set_, cat, prompt, expect, accept=None, wrong=None, tf=True, **kw):
    it = {"id": id_, "set": set_, "category": cat, "prompt": prompt, "expect": expect}
    if accept:
        it["accept"] = accept
    if wrong:
        it["wrong"] = wrong
    it["tf"] = tf
    it.update(kw)
    items.append(it)


# 1. the 50-question round, same prompts and gold answers (matched raw vs packed)
q50 = json.loads((QUIZ / "questions50.json").read_text(encoding="utf-8"))["questions"]
for q in q50:
    add("q%02d" % q["id"], "quiz", q["category"], q["prompt"], str(q["expect"]),
        q.get("accept"), q.get("wrong"), case_sensitive=q.get("case_sensitive", False))

# 2. the packed lane's own wrong answers on the round, scored as candidates
#    next to the gold answer (is the right answer still competitive?)
cand = {4: "7", 5: "42", 8: "49", 14: "Toronto", 31: "1789", 37: "44.39", 38: "60",
        39: "60", 41: "365", 42: "1", 44: "Earth", 45: "Mars"}
for qid, w in cand.items():
    q = next(x for x in q50 if x["id"] == qid)
    add("c%02d" % qid, "cand", q["category"], q["prompt"], w, gen=False, gold_of="q%02d" % qid)

# 3. paraphrase stability: the same fact asked three ways
para = [
    ("ottawa", "Ottawa", ["What is the capital of Canada?", "Which city is the capital of Canada?",
                          "Canada's capital city is which city?"]),
    ("jupiter", "Jupiter", ["Which planet is the largest in our solar system?",
                            "Name the biggest planet in the solar system.",
                            "Of all the planets orbiting the Sun, which one is the largest?"]),
    ("mercury", "Mercury", ["Which planet orbits closest to the Sun?", "Name the planet nearest to the Sun.",
                            "What is the innermost planet of the solar system?"]),
    ("leap", "366", ["A leap year has how many days?", "How many days does a leap year contain?",
                     "Count the days in a leap year. How many are there?"]),
    ("paris", "Paris", ["What is the capital of France?", "Which city is the capital of France?",
                        "France's capital city is which city?"]),
    ("moon", "1969", ["In which year did humans first land on the Moon?",
                      "What year was the first crewed Moon landing?",
                      "Apollo 11 landed on the Moon in what year?"]),
    ("gold", "Au", ["What is the chemical symbol for gold?", "Which element symbol stands for gold?",
                    "Gold is written with what chemical symbol?"]),
    ("insect", "6", ["How many legs does an insect have?", "An insect has how many legs?",
                     "Count the legs on a typical adult insect. How many are there?"]),
    ("romeo", "Shakespeare", ["Who wrote Romeo and Juliet?", "Romeo and Juliet was written by whom?",
                              "Name the author of the play Romeo and Juliet."]),
    ("decl", "1776", ["In what year was the United States Declaration of Independence adopted?",
                      "The Declaration of Independence was adopted in which year?",
                      "What year did the American colonies adopt the Declaration of Independence?"]),
    ("water", "H2O", ["What is the chemical formula for water?", "Water has what chemical formula?",
                      "Write the chemical formula of water."]),
    ("hour", "3600", ["How many seconds are in one hour?", "One hour contains how many seconds?",
                      "If you count every second in an hour, how many seconds is that?"]),
    ("hexagon", "6", ["How many sides does a hexagon have?", "A hexagon has how many sides?",
                      "Count the sides of a hexagon. How many are there?"]),
]
pacc = {"6": ["six"], "1969": [], "Shakespeare": ["William Shakespeare"], "H2O": ["H₂O"], "3600": ["3,600"],
        "366": [], "Au": []}
for key, gold, forms in para:
    for k, f in enumerate(forms):
        add("p_%s_%d" % (key, k + 1), "para", "paraphrase", f + JUST, gold, pacc.get(gold),
            group=key, case_sensitive=(gold == "Au"))

# 4. multi-step arithmetic and word problems
multi = [
    ("Maya buys 4 pens at 3 dollars each and pays with a 20 dollar bill. How many dollars of change does she get?", "8"),
    ("A train travels at 60 kilometers per hour for 3 hours, then at 40 kilometers per hour for 2 hours. How many kilometers does it travel in total?", "260"),
    ("Tom has 3 boxes with 12 apples in each box. He gives away 10 apples. How many apples does he have left?", "26"),
    ("What is (8 + 4) times 5, minus 7?", "53"),
    ("A rectangle is 7 meters long and 5 meters wide. What is its perimeter in meters?", "24"),
    ("Sara reads 15 pages a day. How many pages does she read in 2 weeks?", "210"),
    ("A shirt costs 40 dollars and is discounted by 25 percent. What is the sale price in dollars?", "30"),
    ("A class has 28 students. Half of them play soccer, and 4 of the soccer players also play chess. How many soccer players do not play chess?", "10"),
    ("What is 3 squared plus 4 squared?", "25"),
    ("Ben is twice as old as Ana. Ana is 9 years old. How old will Ben be in 5 years?", "23"),
    ("What is 1000 minus 3 times 125?", "625"),
    ("A baker makes 6 trays of 8 muffins and sells 30 of them. How many muffins are left?", "18"),
]
for k, (p, a) in enumerate(multi):
    add("m%02d" % (k + 1), "multi", "multistep", p + JUST, a)

# 5. novel passages: facts that cannot be memorized, so the answer must be reasoned from the text
passages = [
    ("In the village of Orlen, the mill opens at 6 AM. The bakery opens two hours after the mill, and the market opens one hour after the bakery.", "At what time does the market open?", "9 AM", ["9:00 AM", "9 a.m.", "9am", "9:00"]),
    ("Kira has three pets: a cat named Bolt, a dog named Pebble, and a parrot named Juno. The dog is the oldest, and the parrot is younger than the cat.", "Which pet is the youngest? Give its name.", "Juno", None),
    ("The Tarn Bridge is 340 meters long. The Vell Bridge is 120 meters shorter than the Tarn Bridge.", "How long is the Vell Bridge, in meters?", "220", None),
    ("In the game of Skiv, a red token is worth 5 points, a blue token is worth 3 points, and a green token is worth 1 point. Lio has 2 red tokens and 3 blue tokens.", "How many points does Lio have?", "19", None),
    ("The Fennick library is closed every Tuesday and open every other day. Mara wants to visit on the day after Monday.", "Will the library be open when Mara visits? Answer yes or no.", "No", None),
    ("Dr. Ilsen discovered the mineral quarrite in 1931. Twelve years later, her student Pavo first described its blue form.", "In what year was the blue form first described?", "1943", None),
    ("Five runners finished a race. Ada finished before Bea. Cal finished after Bea but before Dov. Eli finished last.", "Who finished fourth?", "Dov", None),
    ("The planet Vorra has two moons. Ise orbits Vorra in 9 days, and Telo takes three times as long as Ise.", "How many days does Telo take to orbit Vorra?", "27", None),
    ("A jar holds only marbles: 4 are red, 6 are green, and the rest are white. There are 15 marbles in all.", "How many marbles are white?", "5", ["five"]),
    ("The city of Quell is north of Dara. Dara is north of Mirn.", "Which of the three cities is farthest south?", "Mirn", None),
]
for k, (ps, q, a, acc) in enumerate(passages):
    add("r%02d" % (k + 1), "pass", "passage", "Read the passage and answer the question.\nPassage: %s\nQuestion: %s%s" % (ps, q, JUST),
        a, acc, wrong=None)

# 6. calibration and abstention: made-up entities must get "I don't know";
#    real questions under the same instruction must still be answered
IDK = " Answer with just the answer. If the question has no real answer or you do not know it, reply exactly: I don't know."
unk = [
    "What is the capital city of the country of Zantoria?",
    "In what year did the explorer Hendrik Valmoor reach the summit of Mount Ostrel?",
    "What is the chemical symbol for the element vorinium?",
    "Who won the 1987 Nobel Prize in Interplanetary Gardening?",
    "How many moons does the planet Krellis have?",
    "What is the phone number of the Blue Heron Cafe in the town of Pellford?",
    "What was the population of the city of Ardenmoor in the year 1600?",
    "What is the title of the third novel by the author Imogen Tarsk?",
]
for k, p in enumerate(unk):
    add("u%02d" % (k + 1), "calib", "unanswerable", p + IDK, "I don't know", abstain=True)
known = [
    ("What is the capital of Italy?", "Rome", None), ("What is the chemical symbol for oxygen?", "O", None),
    ("How many continents are there on Earth?", "7", ["seven"]),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", ["Leonardo", "da Vinci"]),
    ("What is the capital of Japan?", "Tokyo", None), ("How many sides does a triangle have?", "3", ["three"]),
    ("What gas do plants absorb from the air for photosynthesis?", "Carbon dioxide", ["CO2", "carbon dioxide"]),
    ("What is the largest ocean on Earth?", "Pacific", ["Pacific Ocean"]),
]
for k, (p, a, acc) in enumerate(known):
    add("k%02d" % (k + 1), "calib", "answerable", p + IDK, a, acc, abstain=False, case_sensitive=(a == "O"))

# 7. instruction following (generation only, checked by rule)
instr = [
    ("Reply with the word yes in all capital letters and nothing else.", {"exact": "YES"}),
    ("List three fruits, one per line, with nothing else.", {"lines": 3}),
    ("Answer in exactly one word: what color is a clear daytime sky?", {"words": 1, "any": ["blue"]}),
    ("Respond with only a JSON object that has the key \"city\" with the value \"Paris\".", {"json": {"city": "Paris"}}),
    ("Count from 1 to 5, separated by commas, and nothing else.", {"regex": r"^\s*1\s*,\s*2\s*,\s*3\s*,\s*4\s*,\s*5\.?\s*$"}),
    ("Repeat this sentence exactly: The quick brown fox jumps over the lazy dog.", {"exact": "The quick brown fox jumps over the lazy dog."}),
    ("What is the capital of France? Reply in lowercase letters only.", {"exact": "paris", "lower": True}),
    ("Start your reply with the word Certainly, then name one primary color.", {"starts": "Certainly", "any": ["red", "blue", "yellow"]}),
    ("Write the numbers 3, 1 and 2 in increasing order, separated by spaces, and nothing else.", {"regex": r"^\s*1 2 3\.?\s*$"}),
    ("Give a one-sentence answer that ends with an exclamation mark: what do bees make?", {"ends": "!", "any": ["honey"], "lines": 1}),
]
for k, (p, chk) in enumerate(instr):
    add("i%02d" % (k + 1), "instr", "instruction", p, "", tf=False, check=chk)

# 8. long-context recall: a needle in a deterministic haystack
rng = random.Random(20261127)
names = ["Aster", "Birch", "Cobalt", "Dune", "Ember", "Fable", "Garnet", "Harbor", "Iris", "Juniper", "Kestrel",
         "Linden", "Marlow", "Nettle", "Onyx", "Pippin", "Quarry", "Rowan", "Sable", "Thistle"]
things = ["ferry", "tram", "bakery", "lighthouse", "orchard", "museum", "workshop", "garden", "library", "station"]
verbs = ["opens at", "closes at", "was rebuilt in", "employs", "serves", "holds"]


def filler(n):
    out = []
    for i in range(n):
        nm, th, vb = rng.choice(names), rng.choice(things), rng.choice(verbs)
        if vb in ("opens at", "closes at"):
            val = "%d %s" % (rng.randint(1, 12), rng.choice(["AM", "PM"]))
        elif vb == "was rebuilt in":
            val = str(rng.randint(1850, 2010))
        else:
            val = "%d %s" % (rng.randint(3, 90), rng.choice(["people", "visitors", "workers"]))
        out.append("The %s %s %s %s." % (nm, th, vb, val))
    return out


longs = [("l01", 60, 0.1, "blue", "4817"), ("l02", 60, 0.5, "blue", "2093"), ("l03", 60, 0.9, "blue", "7354"),
         ("l04", 120, 0.5, "red", "6128")]
for id_, n, depth, door, code in longs:
    sents = filler(n)
    at = int(depth * n)
    sents.insert(at, "The code for the %s door is %s." % (door, code))
    if id_ == "l04":
        sents.insert(int(0.2 * n), "The code for the green door is 3391.")
    hay = " ".join(sents)
    add(id_, "long", "long-context", "Read these notes, then answer the question.\nNotes: %s\nQuestion: What is the code for the %s door?%s"
        % (hay, door, JUST), code, wrong=(["3391"] if id_ == "l04" else None), depth=depth)

# 9. turn end: the quiz gold answer followed by its newline, teacher-forced. The probe
#    reads the stop-token probability right after that newline, where a model that
#    ends its turn cleanly would stop. (Measuring right after the answer word instead
#    is the wrong place: the raw model also writes a newline first.)
for q in q50:
    add("t%02d" % q["id"], "term", "turn-end", q["prompt"], str(q["expect"]) + "\\n", gen=False)

meta = {
    "name": "NanoTensor retention battery",
    "created": "2026-11-27",
    "notes": "Matched items for raw bf16, packed, expanded and higher-precision-head arms. tf items are teacher-forced by ln_retain (gold answer scored after the engine's first-turn chat template, thinking off); every item with gen true is also generated greedily by the CUDA engine. No model, pack or machine path is stored here.",
    "conditions": {"decoding": "greedy", "chat_template": True, "thinking": False, "fresh_conversation_per_item": True},
}
for it in items:
    it.setdefault("gen", True)
(HERE / "battery.json").write_text(json.dumps({**meta, "items": items}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
with open(HERE / "probe.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("# NanoTensor retention battery, 2026-11-27, built by build_battery.py. id, set, category, prompt, teacher-forced answer.\n")
    for it in items:
        if it["tf"]:
            f.write("\t".join([it["id"], it["set"], it["category"], it["prompt"].replace("\n", "\\n"), it["expect"].replace("\n", "\\n")]) + "\n")
print(len(items), "items;", sum(i["tf"] for i in items), "teacher-forced;", sum(i["gen"] for i in items), "generated")
