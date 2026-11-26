# Precision Gate calibration: derivation record, 2026-11-25

Produced in scratch (pgate crew, scratch phase) from the calibration
dumps of the two scratch vehicles, per DTC-PRECISION-GATE-PLAN
section 5 and the design note section 5. Nothing here has touched
the live tree.

## Inputs

- Pack: the frozen Sigma pack, results/sigma-full-b256-v2run.nq2 +
  .nq2book (format v1, block 256, entries 256, 14,976,764 blocks,
  book 65,536 floats). The scratch copy in pgate/test/ is
  byte-identical (md5 9c4b3a7ff35d513b842b5e5fb8ca5e4f stream,
  4a5806a62b238514d292225ea99e159a book).
- Config (both vehicles): --ids 1 --tokens 32 --steps 1
  --guidance 1 --score-temp 1 --seed 42 on the packed model.
- Dense vehicle: pgate/build/base-engine, dumps
  calib/run/base-layers.bin (3 LDMP passes: encode n=1, two
  denoise evaluations n=32) and base-logits.bin (2 LOG1
  evaluations of 32 positions).
- DTC vehicle: pgate/build/dtc-engine --dtc-calib, dumps
  calib/run/dtc-layers.bin and dtc-logits.bin, same shapes and
  sizes; its run log shows a complete run, ticks in agreement
  (fetch 181, expand 182 of 182, consume 181).
- Analyzer: pgate/calib/analyze.py. Full human summary in
  calib/run/analyze-summary.txt; gate lines in
  calib/run/gate-lines.txt.

## Derivation (exactly as the analyzer computes it)

- Per-layer divergence: for the last denoise pass (n=32), the
  relative L2 between DTC and dense hidden states after each of
  the 26 layers. The noise profile increments are
  delta[l] = sqrt(max(div[l]^2 - div[l-1]^2, 0)).
- Drift budget = 0.5 * div[25] (final-layer divergence).
- dmax per position = max over the vocab of |logit_DTC -
  logit_dense| across both dumped evaluations (64 positions).
  tie_epsilon = 4 * max(dmax): a tie can flip only when the
  margin is below 2*dmax; the factor 4 covers unseen prompts on
  the same pack.
- gate-gen (legacy key) = FNV-1a 64 (seed
  14695981039346656037, prime 1099511628211) over the geometry
  words (version, block, entries as u32 LE; nblocks and total
  book floats as u64 LE) and then the stored book bytes (book
  floats region, offset 12 in the v1 book): 987cac0cda1f2f01.

## Headline numbers

- tie_epsilon = 3.507614136e+01 (from dmax max 8.769035e+00,
  p99 8.722591e+00).
- Drift budget = 1.276953220e+00 (final-layer divergence
  2.553906e+00).
- Per-layer divergence range over the profile pass: 7.89e-02
  (layer 0) rising to a peak of 8.478992e+00 at layer 5, then
  decaying to 2.553906e+00 at layer 25. Largest increments:
  layer 4 (7.066219562e+00) and layer 5 (4.685421012e+00);
  median increment 0.
- Dense margin percentiles (1/5/25/50): 0.0125 / 0.0228 /
  0.0786 / 0.2305.
- Argmax agreement between the two vehicles: 8 of 64 positions.

## Outputs

- Gate lines: calib/run/gate-lines.txt (gate 1, gate-gen,
  gate-eps, gate-budget, 26 gate-layer lines).
- Legacy sidecar for the frozen pack: placed beside the scratch
  copy as pgate/test/sigma-full-b256-v2run.nq2gate (the engine
  derives this exact path by replacing the stream's .nq2 suffix
  with .nq2gate). Content identical to gate-lines.txt.

## Observation the next crew must not lose

The measured DTC-vs-dense divergence on this pack is at the
1e-1 to 1e0 scale (argmax agreement 8/64), NOT the ~1e-6
association-noise scale the plan's framing assumed. Consequences
for the gate as calibrated here: tie_epsilon (35.08) exceeds
every margin percentile measured, so the margin gate would
escalate essentially every final evaluation; and the drift
projection crosses the budget (1.277) as soon as layers 4 and 5
enter any lookahead window, so Auto would finalize EXPAND for
nearly everything downstream of layer 3. As calibrated, Auto
would behave like Think Harder with gate overhead. Whether the
divergence is a property of the pack/model or of the scratch
DTC vehicle's consume path is a question this record raises and
the scratch tests (impl-engine Fast/Auto against Evidence) bear
on; it is not answered here.
