# PAPER SKELETON — write the paper from this file

> Companion to `FINDINGS.md` (all numbers), `PAPER_PLAN.md` (background),
> `NOTES.md` (source article). Every stat below is copied from FINDINGS v5
> (final, 2026-08-14, bench v0.1.1, 120 runs). Write in English, ACM sigconf
> two-column, **8 pages max including figures and references**.
>
> How to use: work top to bottom of §7's schedule, not of the paper. Results
> first (the numbers are done), methods second, intro last-but-one, abstract
> polish last. Each section below lists: goal, contents, exact numbers, and
> 1–3 drafted sentences you can keep, cut, or rewrite. The drafted sentences
> are suggestions; the paper must be yours.

## 0. Fixed decisions

| Decision | Value |
|---|---|
| Target | **ICAIF 2026 workshop track** (Milan, workshops Nov 14–15) |
| Deadline reality | Main track closed Aug 9. Workshop CFPs are appearing now; paper deadlines land ~mid-September (notifications Oct 14). Watch icaif2026.org for the workshop list and pick the best-fitting one the day it posts. |
| Fallback (same week you'd submit) | arXiv q-fin.CP + SSRN. Costs nothing, timestamps the result. |
| Full-length follow-up | GECCO 2027 (~Feb deadline) with bench v0.2: PBO/DSR layer, fixed dedupe hashing, budget curves, Groq 70B arm. |
| Format | ACM sigconf LaTeX, anonymous option per workshop rules. 8 pages hard cap. |
| Scope discipline | v0.1.1 results only. No result that lives in a run you have not done. The Groq arm and budget curves are "future work" unless they finish before the deadline. |

**Title candidates** (pick one, keep the runner-up as a section head):

1. *Nothing Beats Random: Matched-Budget Controls for LLM Mutation in
   Trading-Strategy Search* ← recommended
2. *Evolution Optimizes the Illusion: A Ground-Truth Benchmark for
   Strategy-Search Operators*
3. *Does the LLM Earn Its Seat? Placebo Controls for Evolutionary
   Trading-Strategy Discovery*

## 1. Abstract (drafted — ~200 words, trim to the workshop's limit)

> Recent systems use large language models as mutation operators to evolve
> trading strategies and report profitable discoveries. None answers a basic
> control question: at the same budget, does the LLM beat random search? We
> build a benchmark that isolates the proposal operator. Three operators
> (random sampling, tree GP, a 7B LLM) search one grammar of 5.6M strategy
> structures; one optimizer fits all parameters; one purged cross-validation
> scheme scores every candidate; a walled-off holdout is read once per run.
> The markets have known ground truth: zero-drift GBM, where no timing edge
> exists, and AR(1) returns with a planted edge of known size. We run 2
> worlds × 3 operators × 20 paired seeds at 60 scored candidates each. Every
> operator manufactures ≈ +0.63 of believed Sharpe on the edgeless world
> (p = 2×10⁻⁶) and pays ≈ 0 out of sample. No operator beats any other on
> paid performance in either world (paired Wilcoxon p ≥ .156). One robust
> difference exists: GP inflates believed scores above both alternatives in
> both worlds (p ≤ .044) while paying no more. Selection optimizes the
> metric; on noise it optimizes noise. The 7B LLM matches random search at
> 2.1–2.6× the wall clock with 35–38% duplicate proposals. We release the
> benchmark and argue that operator-versus-random controls in ground-truth
> worlds should be a reporting requirement for this literature.

## 2. Section-by-section skeleton (page budget in brackets)

### §1 Introduction [1.0 p]

Goal: the missing-control hook, then contributions. No literature detail yet.

- Open on the field's implicit bet: LLM-guided evolution finds trading
  strategies (AlgoEvolve, MadEvolve, QuantaAlpha, FunSearch lineage). All
  report believed backtest scores; none reports the placebo arm.
- The trap is specific to trading: the objective is noisy and gameable, so
  *any* selection loop manufactures score. A control must separate "search
  found structure" from "search exploited evaluation noise."
- Contributions list (4 bullets):
  1. an open benchmark where the proposal operator is the only moving part
     and world truth is known (null + planted edge);
  2. a paired-seed, pre-planned kill-test protocol at matched scored-budget;
  3. a null: at budget 60, neither GP nor a 7B LLM beats random search on
     out-of-sample performance in any world (F10);
  4. a positive: evolution inflates *believed* scores without improving paid
     ones (F12), and the same selection bias produced two false positives in
     our own analysis before paired tests killed them (F11).
- Drafted sentence (thesis): "A better search operator gives you a better
  backtest; on markets with no edge it gives you nothing else, and the
  backtest cannot tell you which case you are in."

### §2 Related work [0.75 p]

Goal: one paragraph per cluster + the gap table (Table 1).

- LLM-evolution trading systems: AlgoEvolve (arXiv:2606.26173), MadEvolve
  (arXiv:2605.23007), QuantaAlpha (arXiv:2602.07085), Maciejewski's
  practitioner write-up (the seed of this bench). What each reports; what
  each omits (see Table 1).
- Program-search lineage: FunSearch and successors. Key contrast: their
  objectives are exactly verifiable (a bin-packing score is true); a
  backtest score is an estimate that search itself corrupts.
- Backtest-overfitting literature: Bailey & López de Prado (PBO, Deflated
  Sharpe), CSCV; purged CV. We import the discipline; v0.2 imports the
  statistics.
- LLM mutation collapse: arXiv:2606.05408 predicts degenerate proposal
  diversity; our duplicate-share ordering (llm 35–38% > gp 30–31% >
  random 23%) confirms it in this domain.
- **ACTION before submitting: re-read the three arXiv systems and verify
  every ✗ in Table 1 against the papers' own text. The table is the paper's
  most attackable object.** (Current entries reflect our earlier read.)

### §3 The benchmark [1.75 p]

Goal: a reviewer can reimplement from this section. Map FINDINGS §3 items
1–11 into prose. Fig 1 = pipeline schematic.

- Grammar: 6 leaves + 3 combinators, depth ≤ 3, ≤ 9 nodes, 5,587,818
  structures (exact enumeration). Proposers emit structure only.
- Structure/values split: differential evolution fits parameters per split;
  identical fitting for every operator (imperfect on purpose, shared, fair).
- Scoring: median OOS Sharpe over 24 random 75/25 whole-quarter splits,
  1-bar purge at test-block entries. Disclose the residual (indicator
  windows cross quarter boundaries; chronological walk-forward is future
  work).
- Budget fairness: 60 *scored* candidates; duplicates and rejects are logged
  overhead. Paired seeds: same worlds, same splits, same DE randomness for
  every operator.
- Holdout: final 350 bars, read once after champion refit.
- Worlds: gbm (strict null), ar1 phi=0.10 (oracle gross +1.48, **net-of-5bp
  ceiling +1.10**; recoveries compare to net). bootstrap deferred.
- Operators: random / GP (tournament-3, pop 24, subtree ops) / LLM
  (qwen2.5-coder:7b, top-25 history, exploration every 5th, 2 repairs).
- Drafted sentence (design credo): "Same grammar, same optimizer, same
  exam, same budget; only the inventor changes."

### §4 Results [2.0 p]

Goal: three subsections mirroring the TL;DR. Fig 2 (gbm strips), Fig 3
(ar1 strips + ceiling line), Fig 4 (paired believed dots), Tables 2–3.

- **4.1 Fool's gold on the null world.** Believed +0.626/+0.626/+0.625,
  each p = 2×10⁻⁶ (20/20 positive seeds per arm); paid −0.19…−0.31, none
  ≠ 0 (p ≥ .117). Holdout rail works (F2).
- **4.2 Nothing beats random where it counts.** Paired holdout Wilcoxon:
  ar1 p = .156/.648/.968, gbm p = .981/.674/.334; every ar1 pairing wins
  exactly 10/20 seeds. Recovery of the +1.10 net ceiling: 37% / 72% / 40%
  in median; medians differ, paired tests do not (state both, explain why
  paired is the right lens). GP's champion is byte-identical to random's on
  3/20 (gbm) and 5/20 (ar1) seeds.
- **4.3 Evolution optimizes the illusion (the positive result).** GP raises
  *believed* scores everywhere: vs random p = .004 (gbm, 14/20) and .044
  (ar1); vs llm p = .044 / .006. Median paired lift +0.05. Same worlds, no
  paid improvement. On gbm the believed signal is noise by construction, so
  selection provably climbed noise.
- **4.4 (or fold into 4.3) Cost of the LLM seat.** 2.1–2.6× wall-clock,
  35–38% duplicates, 134 invalid replies across 40 runs; gaps statistically
  flat across operators (p ≥ .278), larger on the null world (+0.81…+0.96)
  than the edge world (+0.34…+0.53).
- Drafted sentence: "The only thing our strongest searcher reliably
  optimized was the score we searched on."

### §5 How we fooled ourselves (discussion) [1.0 p]

Goal: F7/H7 autopsy as a first-class result, not a confession buried in a
footnote. This section is the paper's personality.

- Timeline: at n=5, "the LLM pays best" (+0.532 vs +0.116 medians) looked
  real and mechanistically explicable (F8's narrow-beam story fit it). At
  n=20 with pre-planned paired tests: p = .968. The follow-up "the LLM
  inflates less" died the same way (p = .70).
- The F8 mechanism study stays interesting: search breadth random ≈ 55 >
  gp ≈ 43–45 > llm ≈ 32–37 depth-1 families per 60 candidates; the star
  tree was available to all arms and fate diverged on what *else* each arm
  scored.
- The symmetry claim: multiple-testing bias attacks the researcher exactly
  as it attacks the strategies; the same discipline (pre-planned paired
  tests, controls, holdouts) kills both.
- Drafted sentence: "We ran the experiment on strategies and caught
  ourselves in the control group."

### §6 Limitations and scope [0.5 p]

Straight from FINDINGS §6, compressed: one budget / one grammar / one 7B
model (the null is scoped, not universal); llm arm reproducible in
distribution only (unseeded sampling); synthetic worlds only; Sharpe-only
objective; known dedupe bug biased against the llm (F9, disclosed, fixed in
v0.2); single noisy holdout per seed, medians carry claims.

### §7 Conclusion + what would change our mind [0.25 p]

- Restate: controls first, scale after. The bench is open; adding an
  operator is one class.
- Falsifiers we commit to testing: budget curves (120/240), a 70B-class
  arm, real-data worlds. If any separates from random on paid performance,
  the null is scoped to small models and budgets and we will say so.

### Back matter

- **Reproducibility statement:** seeds, commands, environment from FINDINGS
  §8; repo link (init the git repo first).
- **Disclosure (drafted, edit to taste):** "Benchmark code, experiment
  orchestration, and statistical analysis were developed with an AI
  assistant (Claude, Anthropic); all experiments ran on the author's
  hardware; the author set the research questions, pre-planned the tests,
  reviewed the code, and is responsible for all claims. The 7B model under
  test is unrelated to the assistant used for development."
- References: ~15–20 entries. Cite the three systems, FunSearch, López de
  Prado / Bailey PBO+DSR, purged CV, the mutation-collapse paper, Wilcoxon
  methodology if venue expects it.

## 3. Figures and tables

| Item | Status | Spec |
|---|---|---|
| Fig 1 pipeline schematic | **to make** | One row: propose → fit → purged CV → observe loop → refit → one holdout read. Annotate "structure only" and "budget = scored" |
| Fig 2 fool's gold, gbm | exists (`runs/fools_gold_gbm.png`) | Regenerate at n=20 with final styling; believed cloud above zero, paid cloud straddling it |
| Fig 3 recovery, ar1 | exists (`runs/fools_gold_ar1.png`) | Add horizontal net-ceiling line at +1.10 and label recoveries |
| Fig 4 paired believed dots | **to make** (I can build from summaries) | Per-seed gp−random believed difference, gbm: 20 dots, 14 above zero, median +0.05 marked. Makes F12 visible in one glance |
| Table 1 gap table | **to draft + verify** | Rows: AlgoEvolve, MadEvolve, QuantaAlpha, practitioner write-up, FunSearch, this work. Columns: random arm / GP arm / matched budget / ground-truth worlds / read-once holdout / PBO-DSR |
| Table 2 main results | numbers final | Both worlds × 3 ops: med believed, med paid, med gap, one-sample p, paired p matrix (or split into 2a/2b) |
| Table 3 overhead | numbers final | rejects, dups, dup share, med wall-clock (ar1 only; gbm timing not quotable) |

## 4. Canonical numbers (copy from here, never re-derive)

gbm n=20: believed rnd +0.626 / gp +0.626 / llm +0.625 (each p=1.9×10⁻⁶ vs 0);
paid −0.186 / −0.307 / −0.310 (p=.117/.277/.165 vs 0); gaps +0.806/+0.960/+0.908.
Paired paid: gp-rnd .981, gp-llm .674, llm-rnd .334. Paired believed: gp-rnd
**.004** (14/20), gp-llm **.044** (14/20), llm-rnd .475. Gap diffs: ≥ .349.

ar1 n=20 (net ceiling +1.10): believed +0.905/+0.980/+0.939; paid
+0.411/+0.787/+0.445 (recovery 37/72/40%; each >0: p=.015/.002/.011);
gaps +0.526/+0.500/+0.343. Paired paid: gp-rnd .156, gp-llm .648, llm-rnd
.968 (every pairing 10/20 wins). Paired believed: gp-rnd **.044**, gp-llm
**.006**. Gap diffs: ≥ .278.

Overhead (20 runs/arm/world): dups rnd 357+357, gp 514+550, llm 676+758;
rejects llm 74 (gbm) + 60 (ar1), others 0. Wall-clock (ar1, clean): med
817 / 737 / 2,110 s → llm 2.6× rnd; gbm ratio 2.1× (contaminated, don't
quote absolute gbm times). Champion identity gp==rnd: 3/20 gbm, 5/20 ar1.
Anomaly: llm_gbm_s8 26,256 s (Ollama stall; results normal).

## 5. Submission checklist

1. [ ] `git init`, first commit, public repo (before the paper cites it)
2. [ ] Your name in LICENSE; LOG.md up to date (authorship trail)
3. [ ] Re-read the 3 arXiv systems; verify every cell of Table 1
4. [ ] Regenerate Figs 2–3 at n=20; build Figs 1 and 4
5. [ ] Pick the ICAIF workshop the day the list posts; note its deadline
6. [ ] Red-team pass: every claim traced to a number in §4 above
7. [ ] arXiv/SSRN preprint the same week regardless of workshop outcome

## 6. Writing schedule (deadline assumed ~Sept 15)

| Week | Deliverable |
|---|---|
| by Aug 21 | §4 Results drafted + Figs 2–4 final (numbers already locked) |
| by Aug 28 | §3 Benchmark + §2 Related work (includes the 3-paper re-read) |
| by Sept 4 | §1 Intro + §5 discussion + §6–7 |
| by Sept 11 | Abstract polish, ACM format, red-team pass, submit early |
