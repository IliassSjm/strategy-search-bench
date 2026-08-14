# FINDINGS — hypotheses, assumptions, results

> The canonical record of what we claimed, what we assumed, and what the runs
> actually showed. Update this file after every experiment; the paper gets
> written from it. Companion files: `PAPER_PLAN.md` (research plan, venues),
> `NOTES.md` (source article), `LOG.md` (personal journal), `runs/` (raw data).
>
> All results below are on bench **grammar v0.1.1** (momentum n≥1, 1-bar purge).
> Runs made before the 2026-08-11 patch live in `runs_pre_patch/` and are NOT
> comparable — different grammar, different search space.

## TL;DR (final for bench v0.1.1 — full matrix complete 2026-08-14: 2 worlds × 3 operators × 20 seeds = 120 runs)

Three results, in order of importance:

1. **Nothing beats random search where it counts.** On out-of-sample (paid)
   performance, no operator differs from any other in either world — paired
   Wilcoxon p ≥ .156 on the planted-edge world, p ≥ .334 on the null world.
   On the null world all three manufacture the same ≈ +0.63 of believed Sharpe
   (each p = 2×10⁻⁶ vs zero) and pay ≈ 0; on the planted-edge world all three
   recover 37–72% of the +1.10 net ceiling with no separable differences.
2. **Evolution optimizes the illusion.** GP's one statistically solid
   distinction is that it reliably inflates the *believed* score above the
   other operators' — in BOTH worlds (vs random p = .004 gbm / .044 ar1;
   vs llm p = .044 / .006) — while never separably improving what the data
   actually pays (all paired paid p ≥ .156). Selection pressure optimizes the
   metric; when the metric is noise, it optimizes noise. The paper's sharpest line.
3. **The 7B LLM is an expensive random search.** Indistinguishable from random
   on every paid metric in both worlds, at 2.1–2.6× wall-clock, 35–38%
   duplicate proposals, and it is the only operator that emits invalid
   candidates (134 rejects across its 40 runs).

**Meta-lesson for the discussion section:** the analysis itself manufactured
two exciting findings en route ("LLM pays best" at n=5; "LLM inflates less"
at n=20-by-columns) and both died under pre-planned paired tests. Selection
bias attacks the researcher exactly as it attacks the strategies.

## 1. Research questions

- **RQ1 (power):** at a matched budget of scored candidates, does LLM mutation
  recover a *planted* edge better than GP or random mutation over the same grammar?
- **RQ2 (false discovery):** in worlds with provably no edge, how much apparent
  out-of-sample Sharpe does each operator manufacture?
- **RQ3 (rails):** do the overfitting controls (read-once holdout; later
  PBO/DSR and null-max) correctly separate real discoveries from manufactured ones?

## 2. Hypothesis board

| # | Hypothesis | Status | Evidence |
|---|---|---|---|
| H1 | Every search operator manufactures apparent edge on edgeless markets | **Confirmed (final, n=20)** | E1: med believed +0.626 / +0.626 / +0.625 on GBM, each p = 2×10⁻⁶ vs zero; med paid −0.19…−0.31, none ≠ 0 (p ≥ .117) |
| H2 | LLM mutation outperforms dumb operators (the field's implicit bet) | **Refuted (final; 7B, budget 60, both worlds)**: no pairing separable on paid | Paired Wilcoxon holdout — ar1: gp-rnd p=.156, gp-llm p=.648, llm-rnd p=.968; gbm: p=.981 / .674 / .334 |
| H3 | The read-once holdout catches the inflation | **Confirmed (final)** | gbm: all 3 arms deflate to ≈0 (paid p ≥ .117 vs zero); ar1: paid stays positive (each op p ≤ .015 vs zero) — the rail discriminates worlds correctly |
| H4 | Evolution ≈ random on noise, > random when a real gradient exists | **Refuted → replaced by F12**: gp raises the *believed* score everywhere (p ≤ .044, both worlds) but the *paid* score nowhere (p ≥ .156); gp's champion is literally random's warm-up tree on 3/20 gbm and 5/20 ar1 seeds | Paired matrix runs/; F12 |
| H5 | The believed-minus-paid gap is ≈ constant across worlds (inflation stacks on top of real edge) | **Holds within noise (final)**: gbm gaps +0.81 / +0.96 / +0.91, ar1 +0.53 / +0.50 / +0.34; all paired gap differences n.s. (p ≥ .278) | Note the level shift: gaps are *larger* on the null world — inflation is the whole score there |
| H6 | LLM proposals collapse to repeated structures (cf. arXiv:2606.05408) | **Confirmed for 7B, robust (final)** | 40 runs: 1,434 dups + 134 rejects; dup share llm 35% (gbm) / 38% (ar1) vs gp 30/31% vs random 23/23% |
| H7 | The LLM's plausibility prior acts as a regularizer (smaller inflation, better generalization) | **Refuted (final)** — both versions died: "pays best" (F7, n=5 luck; llm-rnd paid p=.968/.334) and "inflates less" (gap diffs p=.701 ar1, p=.349 gbm) | Autopsy in F7/F11; the kill-test discipline worked twice |

## 3. Design assumptions (Methods section, with rationale)

1. **Same grammar for every operator** — strategies are JSON expression trees
   over 6 leaves (const_long, trend, momentum, meanrev_z, rsi_gate, breakout)
   and 3 combinators (wsum, vol_gate, switch_z); depth ≤ 3, ≤ 9 nodes;
   **5,587,818 distinct structures** (exact enumeration). Why: makes "same
   search space" literally true. Threat: constrains the LLM's expressive
   advantage; an expressive-Python mode is a planned robustness check.
2. **Structure from the proposer, numbers from the optimizer** — parameter
   ranges are fixed per node type; scipy differential_evolution fits values
   per split. Proposers never set numbers (inherited from Maciejewski 2026).
3. **Fitness** = median out-of-sample Sharpe over 24 random 75/25 assignments
   of whole calendar quarters; signals computed on the full series, scored on
   masked bars; **1-bar purge** at each test-block entry. Disclosed residual:
   indicator windows span quarter boundaries (inherent to non-chronological CV;
   a walk-forward answers the deployment question and is out of scope).
4. **Budget fairness** = 60 *scored* candidates per run; duplicates and rejects
   are logged overhead and do not consume budget.
5. **Holdout** = final 350 bars, excluded from search and splits, read once per
   run after a champion refit (DE budget ×3), 1-bar purge at entry.
6. **Worlds** (all ~2,500 business days, seeded, 5 seeds = 5 histories;
   identical seeds across operators → paired comparison):
   - `gbm`: zero-drift GBM, ann. vol 30% — strict null, no timing edge exists.
   - `ar1`: AR(1) returns, phi = 0.10 — planted edge; oracle = sign(yesterday):
     **gross ceiling +1.48, net-of-costs ceiling +1.10** (empirical, 5 seeds).
     Compare recoveries to the NET number.
   - `bootstrap` (not yet run): stationary block bootstrap of demeaned real
     returns — *approximately* edgeless only (sub-block autocorrelation survives).
7. **Costs** = 5 bp per unit turnover; no slippage/impact/borrow. No claim of
   tradability anywhere.
8. **Operators**: `random` = fresh uniform tree each call (the floor);
   `gp` = tournament-3 selection, pop 24, subtree crossover + mutation
   (p = 0.35), plain hand-written implementation; `llm` = qwen2.5-coder:7b via
   local Ollama, sees top-25 scored history best-first, exploration turn every
   5th call, 2 repair retries then reject. Deviation from Maciejewski's
   whole-history prompt: we cap at 25 (context economy) — ablation candidate.
9. **Fitting is imperfect by design** (DE iter 6 × pop 8 during search): e.g.
   on ar1 a local optimum at momentum n≈22 (+0.71) competes with the true n=1
   (+0.94). Identical fitting for all operators, so comparisons stay fair;
   raise --de-iter for final paper runs.
10. **Determinism**: worlds, splits, DE, and operators all seeded; reruns reproduce.
11. Sharpe = mean/std × √252, ddof = 1, no risk-free adjustment.

## 4. Experiments

### E1 — null world (gbm), 2026-08-11 → 08-14 (complete at n = 20)

Setup: `--world gbm --budget 60 --splits 24`, M2 Pro. Seeds 0–4 first
(2026-08-11), seeds 5–19 added 2026-08-13/14 with `--workers` throttling.
Caveat: gbm arms ran concurrently at various points → **gbm wall-clock is not
quotable** (use E2's, where the llm arm ran solo).

**Final result (n = 20, the numbers the paper quotes):**

| op | med CV believed | med holdout paid | med gap | holdout IQR | rejects | dups | dup share |
|---|---|---|---|---|---|---|---|
| random | +0.626 | −0.186 | +0.806 | [−0.43, +0.11] | 0 | 357 | 23% |
| gp | +0.626 | −0.307 | +0.960 | [−0.60, +0.17] | 0 | 514 | 30% |
| llm | +0.625 | −0.310 | +0.908 | [−0.79, +0.20] | 74 | 676 | 35% |

Statistics (paired across the same 20 worlds): believed score vs zero —
**p = 2×10⁻⁶ for every operator** (the fool's gold, now with certainty);
holdout vs zero — p = .117 / .277 / .165, i.e. **nobody paid anything**.
Paired holdout: gp-rnd p=.981, gp-llm p=.674, llm-rnd p=.334 — flat null.
Paired *believed*: **gp beats random p=.004 (14/20 seeds) and beats llm
p=.044 (14/20)** — evolution reliably inflates the in-sample score on pure
noise while paying the same nothing (→ F12). The three medians agreeing to
±0.001 (+0.625…+0.626) is a coincidence of medians, not of distributions.
gp's champion is byte-identical to random's on 3/20 seeds.

Below: the original seeds 0–4 detail, kept as provenance.

| op | seed | CV believed | holdout paid | buy&hold holdout | rej | dup | champion |
|---|---|---|---|---|---|---|---|
| random | 0 | +0.418 | −0.288 | −0.483 | 0 | 19 | vol_gate(trend, trend) |
| random | 1 | +0.629 | −0.309 | −0.597 | 0 | 24 | momentum |
| random | 2 | +1.049 | −0.389 | −0.241 | 0 | 19 | wsum(momentum, switch_z(breakout, …)) |
| random | 3 | +0.856 | +0.412 | −0.221 | 0 | 14 | switch_z(rsi_gate, rsi_gate) |
| random | 4 | +0.372 | +0.000 | +0.738 | 0 | 15 | wsum(rsi_gate, rsi_gate) |
| gp | 0 | +0.500 | +1.291 | −0.483 | 0 | 21 | vol_gate(rsi_gate, wsum(…)) |
| gp | 1 | +0.629 | −0.309 | −0.597 | 0 | 23 | momentum |
| gp | 2 | +0.989 | −0.881 | −0.241 | 0 | 19 | wsum(wsum(meanrev_z, rsi_gate), …) |
| gp | 3 | +0.894 | −1.424 | −0.221 | 0 | 33 | vol_gate(const_long, wsum(breakout, breakout)) |
| gp | 4 | +0.589 | −0.304 | +0.738 | 0 | 45 | vol_gate(const_long, vol_gate(…)) |
| llm | 0 | +0.192 | +1.329 | −0.483 | 2 | 36 | rsi_gate |
| llm | 1 | +0.795 | +0.470 | −0.597 | 3 | 56 | vol_gate(switch_z(trend, rsi_gate), …) |
| llm | 2 | +0.795 | −0.802 | −0.241 | 4 | 42 | wsum(switch_z(trend, meanrev_z), …) |
| llm | 3 | +0.612 | −1.275 | −0.221 | 5 | 32 | wsum(switch_z(trend, breakout), …) |
| llm | 4 | +0.327 | −0.641 | +0.738 | 2 | 48 | vol_gate(const_long, switch_z(…)) |

(Seeds 0–4 medians were random +0.629/−0.288 · gp +0.629/−0.309 ·
llm +0.612/−0.641 — the n=20 numbers above supersede them; the early llm
holdout median of −0.641 was the same small-sample noise that produced F7
on ar1, in the opposite direction.)

### E2 — planted-edge world (ar1, phi=0.10, net ceiling ≈ +1.10), 2026-08-12/13

Status: **complete at n = 20 seeds per operator** (60 runs). Seeds 5–19 ran
with `--workers 3` (thermal management; results unaffected — workers only
changes scheduling).

**Final result (n = 20, the numbers the paper quotes):**

| op | med CV believed | med holdout paid | med gap | recovery of +1.10 | holdout IQR | rejects | dups | med sec |
|---|---|---|---|---|---|---|---|---|
| random | +0.905 | +0.411 | +0.526 | 37% | [+0.09, +1.18] | 0 | 357 | 817 |
| gp | +0.980 | +0.787 | +0.500 | 72% | [+0.29, +1.38] | 0 | 550 | 737 |
| llm | +0.939 | +0.445 | +0.343 | 40% | [+0.06, +1.00] | 60 | 758 | 2,110 |

Paired statistics (same 20 worlds for everyone): every operator beats every
other on exactly **10/20 seeds**; Wilcoxon signed-rank on holdout — gp vs
random p=.156, gp vs llm p=.648, llm vs random p=.968; on inflation gap —
llm vs random p=.701, llm vs gp p=.729. **No pairing is separable** on what
the data pays. The believed score is another story: gp inflates it above
random (p=.044) and above llm (p=.006) here too — the same signature as on
gbm (→ F12). GP's
attractive median comes from the upper half of its distribution, not from
consistent wins; on 5/20 seeds its champion is *literally the tree random
found in the shared warm-up* (budget 60 doesn't always let evolution escape
its initialization). Reading: all three operators partially recover the
planted edge; none earns its seat over random at this budget; the
differences that remain are cost and behavior (llm: 37% dups, 2.6× time).

Below: the original seeds 0–4 detail, kept as provenance for the F7 autopsy.

| op | seed | CV believed | holdout paid | buy&hold holdout | rej | dup | champion |
|---|---|---|---|---|---|---|---|
| random | 0 | +0.690 | −0.682 | −0.536 | 0 | 19 | vol_gate(trend, trend) |
| random | 1 | +0.784 | +0.495 | −0.666 | 0 | 24 | switch_z(switch_z(breakout, meanrev_z), …) |
| random | 2 | +0.885 | +0.116 | −0.257 | 0 | 19 | wsum(trend, momentum) |
| random | 3 | +1.288 | +0.396 | −0.248 | 0 | 14 | wsum(meanrev_z, const_long) |
| random | 4 | +0.727 | −0.578 | +0.818 | 0 | 15 | vol_gate(vol_gate(breakout, breakout), momentum) |
| gp | 0 | +0.690 | −0.682 | −0.536 | 0 | 31 | vol_gate(trend, trend) |
| gp | 1 | +1.037 | +1.336 | −0.666 | 0 | 35 | vol_gate(trend, breakout) |
| gp | 2 | +0.995 | +0.257 | −0.257 | 0 | 19 | wsum(const_long, const_long) |
| gp | 3 | +1.369 | +0.379 | −0.248 | 0 | 43 | wsum(meanrev_z, switch_z(meanrev_z, const_long)) |
| gp | 4 | +0.676 | +0.057 | +0.818 | 0 | 44 | wsum(meanrev_z, breakout) |
| llm | 0 | +0.446 | +0.095 | −0.536 | 4 | 31 | wsum(rsi_gate, meanrev_z) |
| llm | 1 | +1.019 | +1.284 | −0.666 | 2 | 66 | wsum(trend, meanrev_z) |
| llm | 2 | +0.899 | −0.027 | −0.257 | 5 | 70 | switch_z(wsum(wsum(trend, breakout), wsum(momentum, meanrev_z)), rsi_gate) |
| llm | 3 | +1.187 | +0.715 | −0.248 | 3 | 10 | wsum(trend, meanrev_z) |
| llm | 4 | +0.692 | +0.532 | +0.818 | 4 | 32 | wsum(trend, meanrev_z) |

Medians at n=5 were: random +0.784/+0.116 · gp +0.995/+0.257 · llm
+0.899/+0.532 — the source of the now-refuted F7. What n=20 revealed: seeds
0–4 were simply an unlucky draw for random and a lucky one for the llm.
Still true at any n: holdouts flip positive when real signal exists (the
bench detects power), and no champion in any arm is the true lag-1 momentum
structure — the edge gets captured through proxies.

## 5. Cross-experiment findings

- **F1 (RQ2 answered, final): the fool's-gold effect is real, measured, and
  certain.** ≈ +0.63 believed Sharpe from provably empty markets — every
  operator, 20/20 world-seeds each, one-sample Wilcoxon p = 2×10⁻⁶ (the
  minimum attainable at n=20: not one seed produced a non-positive believed
  score). The holdout pays none of it back (p ≥ .117 vs zero).
- **F2 (RQ3, holdout rail, final): works.** Deflates everything on gbm (paid
  ≈ 0 for all arms); passes real edge through on ar1 (paid > 0, p ≤ .015 per
  arm). The rail discriminates worlds correctly in both directions.
- **F3 (RQ2, final): on null worlds, a 7B LLM = slow random search.**
  Identical believed (+0.625 vs +0.626) and paid (−0.31 vs −0.19, paired
  p=.334) scores, at 2.1× wall-clock with 35% dups and 74 rejects. The
  field's implicit bet fails *where there is nothing to find*.
- **F4 (superseded by F10/F12): "evolution > random when a real gradient
  exists" did not survive n=20.** The n=5 hint (+0.995/+0.257 vs
  +0.784/+0.116) collapsed to paired p=.156 on paid; what evolution
  *reliably* does is raise the believed score (F12), not the paid one.
- **F5 (final): the inflation gap is statistically flat across operators in
  both worlds** — gbm +0.81/+0.96/+0.91, ar1 +0.53/+0.50/+0.34; every paired
  gap difference n.s. (p ≥ .278). The n=5 story "the LLM inflates less on the
  signal world" died with H7. What survives is the world-level shift: gaps are
  larger where there is nothing to find, because on the null world the gap IS
  the entire believed score.
- **F6 (final): LLM structural collapse confirmed for the 7B** —
  duplicate-proposal share llm 35/38% > gp 30/31% > random 23/23% (both
  worlds), matching arXiv:2606.05408's mutation-collapse prediction; plus 134
  invalid replies (rejects) that only the llm arm produces. Reject taxonomy +
  per-iteration unique-skeleton curves: TODO from JSONLs.
- **F7 (AUTOPSY — refuted by kill-test (a) at n=20): "the LLM pays best" was
  a 5-seed artifact.** At n=20 the llm's holdout median (+0.445) sits next to
  random's (+0.411), and the paired test is as null as it gets (p=.968).
  The follow-up claim ("the LLM inflates less", med gap +0.34 vs +0.53) also
  failed its paired test (p=.70). Keep this section in the paper as the
  worked example of why the discipline exists: the finding was exciting,
  mechanistically plausible (F8 supported it!), and wrong.
- **F8 (JSONL audit, 2026-08-12): the star tree was available to everyone;
  what differed is what *else* each searcher scored.**
  - *Smoking gun (ar1 s3):* GP scored the mirror star tree `wsum(meanrev_z,
    trend)` at CV +1.154 — and then crowned a flashier tree (+1.369) that paid
    only +0.379 on holdout. The LLM's champion on the same world WAS the star
    tree (+1.187 → paid +0.715). Same discovery, different fate: GP's wider,
    wilder candidate pool contained an imposter with a higher believed score;
    the LLM's narrow pool did not.
  - *Random also drew it* (ar1 s0, mirror order) and fit it to only −0.072 —
    the structure is not auto-win; DE fitting luck and selection dynamics matter.
  - *Search breadth, measured* (distinct depth-1 families among 60 scored):
    random ≈ 55, gp ≈ 43–45, llm ≈ 32–37, both worlds. The narrow-beam story
    is real. But narrowness alone can't explain F7: on gbm the LLM was equally
    narrow yet its inflation gap was *no smaller* (+0.91 at n=20, vs random's
    +0.81 — n.s., p=.349, but certainly not compressed). Discrimination:
    "fewer effective lottery tickets" predicts smaller gaps in both worlds
    (false); "narrow beam *aimed at true structure*" predicts smaller gap only
    where the aim coincides with reality (matches). Taste, not just thrift.
- **F10 (the headline, final at 2×3×20): at budget 60, nothing beats random
  where it counts — including the LLM the field bet on and classical GP.**
  On paid performance every pairing is null in both worlds (ar1: all 10/20
  wins, p ≥ .156; gbm: p ≥ .334); inflation-gap differences are null too
  (p ≥ .278). All three arms recover 37–72% of the net ceiling on ar1, so the
  task is solvable — the operators just don't differ in solving it. The story
  now lives on the escalation axes: budget (do operators separate at
  120/240?), model strength (Groq 70B-class), and world difficulty.
- **F12 (the positive discovery: evolution optimizes the illusion).** The one
  paired difference that survives everywhere is on the *believed* score: GP
  inflates it above random (gbm p=.004, 14/20; ar1 p=.044) and above the llm
  (gbm p=.044; ar1 p=.006) — while its paid score never separates from
  anyone's (p ≥ .156 in all four world-pairings). On gbm this is selection
  pressure doing exactly its job on exactly the wrong target: the fitness
  signal is pure noise, and tournament selection climbs it anyway,
  manufacturing an extra +0.05 of believed-but-fake Sharpe over blind sampling
  (median paired difference; small, but present on 14/20 seeds — consistency,
  not size, is what the test detects). Corollary for practitioners: *the better your search optimizer,
  the better your backtest — and that is precisely why your backtest cannot
  be trusted to rank optimizers.* This is the paper's most quotable result,
  and it required the random arm to see: without the placebo, GP's higher
  believed scores would read as skill.
- **F11 (meta-finding, for the discussion section): the analysis pipeline
  itself manufactured two false positives** (F7 and the inflation-gap claim),
  both killed by pre-planned paired tests. Multiple-testing bias operates at
  the researcher level exactly as it does at the strategy level — the paper's
  thesis, demonstrated on its own authors.
- **F9 (audit artifact — bench bug found, disclosed): duplicate detection
  misses JSON-shape variants.** A leaf can be written with or without
  `"children": []`; both validate and evaluate identically but hash
  differently, so the LLM occasionally got the *same* structure scored twice
  (e.g. gbm s0 i=4/i=36) — semantic duplicates consumed its scored budget.
  Direction of bias: **against the LLM**, so F7 is if anything understated.
  Decision: do NOT patch mid-campaign — seeds 0–19 stay on v0.1.1 for internal
  comparability; normalization fix (hash on children-normalized trees) goes
  into v0.2 before the final paper runs, with everything rerun.

## 6. Current limitations (state honestly in the paper)

- **One budget (60), one grammar size, one small LLM (7B).** The null result
  is scoped to this cell: "a 7B LLM at budget 60 doesn't beat random" is what
  we can claim; budget curves and a 70B-class arm are the obvious escalations
  and could reverse it. The n=20 paired design has power to detect ~0.05
  believed-score shifts (it did, F12) — so the paid-score nulls are not
  power-starved at the effect sizes that would matter economically.
- **The llm arm is the only non-deterministic one.** Run seeds fix worlds,
  splits, and DE — but not the LLM's sampling (temperature 1.0, no seed
  passed to the endpoint). Rerunning an llm run reproduces its world and
  scoring exactly, not its proposal stream. random/gp reruns are bit-exact.
- **Wall-clock caveats:** gbm arms ran concurrently at times (gbm timings not
  quotable; ar1's are clean — llm 2.6× random). One outlier: llm_gbm_s8 took
  26,256 s (~7.3 h vs ~30 min median) during an Ollama stall; the
  request-retry/backoff path kept the run alive and its *results* are normal —
  medians are used for all timing claims partly for this reason.
- Synthetic worlds only; `bootstrap` and the real-data field test pending.
- Sharpe-only objective; PBO / Deflated Sharpe layer not yet implemented (v0.2).
- One 350-bar holdout per seed — noisy per run (spread gp_gbm +1.29 to −1.42
  across seeds is luck, not skill); medians-across-seeds carry every claim.
- F9 dedupe bug (JSON-shape variants hash differently) biases *against* the
  llm arm; unfixed in v0.1.1 by design (mid-campaign comparability), fixed in
  v0.2 with a full rerun.

## 7. Next experiments (priority order)

**Status 2026-08-14: the v0.1.1 synthetic data collection is COMPLETE**
(2 worlds × 3 operators × 20 seeds = 120 runs). Everything below is either
escalation or v0.2. Paper drafting is unlocked — the core story (F1, F10,
F12, F11) no longer depends on pending runs.

1. ~~ar1 seeds 5–19~~ / ~~gbm seeds 5–19~~ / ~~JSONL audit~~ **Done →
   F8/F9/F10/F12.**
2. **Groq free-tier 70B arm** (env-only change: `LLM_BASE_URL=
   https://api.groq.com/openai/v1 LLM_API_KEY=gsk_… LLM_MODEL=<groq id>`):
   does model strength move the LLM off the random line? Cheapest
   biggest-upside experiment left; same seeds 0–19, both worlds.
3. **Budget curves on ar1** (120/240, subset of seeds): does GP/LLM separate
   from random when evolution gets room to compound? The budget-60 null's
   most likely failure mode.
4. **v0.2 bench before final paper runs**: children-normalized tree hashing
   (F9), PBO + Deflated Sharpe on every champion (López de Prado), reject
   taxonomy + unique-skeleton diversity curves from JSONLs — then rerun the
   full matrix on v0.2.
5. `bootstrap` world (approximate null with real distributional texture),
   then the ~50-ticker real-data field test — only after the synthetic story
   is locked.
6. Ablations parked for the paper's robustness section: history shown to the
   LLM (top-25 vs whole), exploration-turn cadence, expressive-Python mode vs
   fixed grammar.

## 8. Reproducibility

- Commands (verbatim):
  `python run_experiment.py --operator {random|gp|llm} --world {gbm|ar1} --budget 60 --seeds 0 … 19`
  (llm env: `LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5-coder:7b`);
  report: `python make_report.py runs/`. Seeds 5–19 ran with `--workers 3…6`
  (thermal management; workers only changes split scheduling, not results —
  verified: same seed, same numbers, any worker count).
- Environment: MacBook Pro M2 Pro, Python 3.11.14, pytest 8.3.3 (17 tests green),
  Ollama 0.32.6, qwen2.5-coder:7b (4.7 GB). Data: synthetic, generated from
  seeds — no downloads involved. The llm arm is reproducible in
  distribution only (see §6: endpoint sampling is unseeded).
- Bench version: grammar v0.1.1 (post-audit patch). TODO: record the git commit
  hash of each experiment here once the repo is initialized.
