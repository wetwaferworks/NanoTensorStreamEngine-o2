# SuperNemo quiz receipts, 2026-11-27

Started 2026-11-27 13:21:41 -0500 (ET). Whole run 1106 s. Machine-readable copy: `receipts.json`.

## Conditions

- engine: sn-engine built from the scratch branch with the raw k/v loader fix (commit 5ce3b8d)
- lane: CUDA, --gpu-mb 3000 (273 tensors, 2830 MB resident), routed experts on the CPU, --threads 12, DTC auto, RAM expert cache 4096 MB, no VRAM expert cache, speculative decoding off
- pack: Nemotron 3.5 Super VL NanoTensor pack (124.45B weights, per-tensor books, RMS frame, PQ d4, tensors of 1,048,576 weights or fewer raw)
- machine: RTX 3050 Ti Laptop 4 GB, i5-12450H (12 threads), 16 GB RAM, Windows
- decoding: greedy, chat on, thinking off, RESET between items (sn-engine acknowledges RESET with DONE)
- max_new: 16 for questions, 320 for essays; one long-lived engine per part so the expert cache stays warm
- priority: --below-normal and BelowNormal runner
- items: questions 5, 15, 28, 31, 44 and essays 3, 7 from the same files as Little Nemo
- other load: a LithoKraft window open for most of the run; the GPU-quantizer runner was checked before and after every item and never ran a GPU or heavy step (no pauses, no overlap)

## Questions: 5/5

Command (paths are placeholders):

    <scratch>\build\quiz\bin\sn-engine.exe --max-new 16 --pack <models>\nemotron-3.5-super-vl-nanotensor\nemotron-3.5-super-vl --shapes <models>\nemotron-3.5-super-vl-nanotensor\nemotron-3.5-super-vl.shapes.tsv --config <models>\nemotron-3.5-super-vl-120b-a12b-bf16\original\config.json --tokenizer <models>\nemotron-3.5-super-vl-120b-a12b-bf16\original\tokenizer.json --threads 12 --below-normal --gpu-mb 3000

Engine READY: `READY nemo-engine 3 model=SuperNemo weights=packed layers=88 cores=12 threads=12 mode=custom simd=0 accel=cuda chat=1 think=0 moe=512 media=image`

Engine load 5.0 s, part wall time 244 s, 1 engine launch(es), 0 pause(s) for other jobs. Pooled prefill 0.917 tok/s over 158 tokens; pooled decode 0.209 tok/s over 11 tokens. Engine RAM peak 708.0 MB, whole-card VRAM peak 3685 MiB, mean GPU load 8.8%.

### Q5 (math): PASS

- Prompt: What is the square root of 144? Answer with just the answer.
- Gold: `12` (also accepted: `twelve`)
- First answer line (graded): `12`
- Timing: prompt 33 tok, prefill 35311 ms (0.935 tok/s); generated 2 tok, decode 0.2731 tok/s; turn 45496 ms; wall 45.496 s; RAM peak 692.7 MB; card VRAM peak 3685 MiB

Full answer:

```text
12
```

### Q15 (capital): PASS

- Prompt: What is the capital of Australia? Answer with just the answer.
- Gold: `Canberra`; listed wrong answers: `Sydney`, `Melbourne`
- First answer line (graded): `Canberra`
- Timing: prompt 29 tok, prefill 32794 ms (0.884 tok/s); generated 2 tok, decode 0.2932 tok/s; turn 42920 ms; wall 42.92 s; RAM peak 706.5 MB; card VRAM peak 3677 MiB

Full answer:

```text
Canberra
```

### Q28 (chem-symbol): PASS

- Prompt: What is the chemical symbol for helium? Answer with just the answer.
- Gold: `He`; listed wrong answers: `H`, `Hl`, `Hm`
- First answer line (graded): `He`
- Timing: prompt 30 tok, prefill 33752 ms (0.889 tok/s); generated 1 tok, decode 0.258 tok/s; turn 41084 ms; wall 41.084 s; RAM peak 706.7 MB; card VRAM peak 3677 MiB

Full answer:

```text
He
```

### Q31 (date): PASS

- Prompt: In what year was the United States Declaration of Independence adopted? Answer with just the answer.
- Gold: `1776`
- First answer line (graded): `1776`
- Timing: prompt 34 tok, prefill 37256 ms (0.913 tok/s); generated 4 tok, decode 0.3325 tok/s; turn 52609 ms; wall 52.609 s; RAM peak 708.0 MB; card VRAM peak 3677 MiB

Full answer:

```text
1776
```

### Q44 (science): PASS

- Prompt: What is the largest planet in our solar system? Answer with just the answer.
- Gold: `Jupiter`; listed wrong answers: `Earth`, `Saturn`, `Neptune`, `Uranus`
- First answer line (graded): `Jupiter`
- Timing: prompt 32 tok, prefill 33279 ms (0.962 tok/s); generated 2 tok, decode 0.3193 tok/s; turn 42801 ms; wall 42.802 s; RAM peak 708.0 MB; card VRAM peak 3677 MiB

Full answer:

```text
Jupiter
```

## Essays: 2/2

Command (paths are placeholders):

    <scratch>\build\quiz\bin\sn-engine.exe --max-new 320 --pack <models>\nemotron-3.5-super-vl-nanotensor\nemotron-3.5-super-vl --shapes <models>\nemotron-3.5-super-vl-nanotensor\nemotron-3.5-super-vl.shapes.tsv --config <models>\nemotron-3.5-super-vl-120b-a12b-bf16\original\config.json --tokenizer <models>\nemotron-3.5-super-vl-120b-a12b-bf16\original\tokenizer.json --threads 12 --below-normal --gpu-mb 3000

Engine READY: `READY nemo-engine 3 model=SuperNemo weights=packed layers=88 cores=12 threads=12 mode=custom simd=0 accel=cuda chat=1 think=0 moe=512 media=image`

Engine load 5.1 s, part wall time 861 s, 1 engine launch(es), 0 pause(s) for other jobs. Pooled prefill 0.981 tok/s over 72 tokens; pooled decode 0.351 tok/s over 273 tokens. Engine RAM peak 709.8 MB, whole-card VRAM peak 3822 MiB, mean GPU load 24.1%.

### Essay 3 (seasons): PASS, 4/5 facts

- Prompt: Write a short factual essay of about 150 words. Explain what causes the seasons on Earth.
- Found: Earth's axis is tilted (matched `tilt`); tilt is about 23.5 degrees (matched `23.5 degrees`); Earth orbits the Sun (matched `orbit`); sunlight angle or directness changes (matched `angle`)
- Missed: hemispheres have opposite seasons
- Forbidden claims found: none (checked: seasons are caused by distance from the Sun)
- Timing: prompt 37 tok, prefill 37325 ms (0.991 tok/s); generated 119 tok, decode 0.3543 tok/s; turn 376918 ms; wall 376.919 s; RAM peak 695.1 MB; card VRAM peak 3677 MiB

Full essay:

```text
The seasons on Earth are caused by the tilt of the planet's axis, not by the distance between the Earth and the Sun. Earth's axis is tilted at an angle of about 23.5 degrees relative to its orbital plane. As Earth orbits the Sun, this tilt causes different parts of the planet to receive varying amounts of sunlight. When the Northern Hemisium is tilted toward the Sun, it experiences summer, while the Southern Hemisium experiences winter. As Earth continues its orbit, the tilt changes, leading to spring and autumn. This axial tilt is the primary cause of seasonal changes.
```

### Essay 7 (dna): PASS, 7/8 facts

- Prompt: Write a short factual essay of about 150 words. Explain the structure of DNA.
- Found: double helix (matched `double helix`); adenine (matched `adenine`); thymine (matched `thymine`); guanine (matched `guanine`); cytosine (matched `cytosine`); A pairs with T and G with C (matched `adenine pairs with thymine`); sugar-phosphate backbone (matched `phosphate`)
- Missed: nucleotides
- Forbidden claims found: none (checked: adenine pairs with cytosine or guanine; triple helix)
- Timing: prompt 35 tok, prefill 36096 ms (0.97 tok/s); generated 154 tok, decode 0.3552 tok/s; turn 473291 ms; wall 473.291 s; RAM peak 709.8 MB; card VRAM peak 3822 MiB

Full essay:

```text
DNA, or deoxyriucleic acid, is a molecule that carries the genetic instructions used in the growth, development, and functioning of all known living organisms. It is composed of two strands coiled around each other to form a double helix. Each strand is made of a sugar-phosphate backbone, with nitrogen bases attached to the sugar. The four types of nitrogen bases are adenine, cytosine, guanine, and thymine. These bases pair up in a specific manner: adenine pairs with thymine, and cytosine pairs with guanine. The sequence of these bases along the DNA strand encodes biological information. The structure of DNA was discovered by James Watson and Francis Crick in 1953, revealing how genetic information is stored and passed on.
```

