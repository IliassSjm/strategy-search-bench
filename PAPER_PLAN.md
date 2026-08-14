# Paper Plan — "Does the LLM Earn Its Seat?"

> Research proposal for your own paper, built on the gap Vincent's article (and the whole field) left open.
> Status: proposal · 2026-08-11 · lives in `Quant/strategy_evolution/`

## TL;DR

**Yes, you can write your own paper, and no, it's not too close — IF you answer the question they all skipped instead of rebuilding their system.** Vincent's own Future Work section calls it "the most important control still missing": *is the LLM actually doing anything a dumb mutation operator couldn't?* I checked the three closest papers (AlgoEvolve, MadEvolve, QuantaAlpha) — **none of them runs that control either.** Every baseline they use is either non-evolutionary (LSTM, Random Forest) or itself LLM-based. Meanwhile, program-synthesis research already runs exactly this comparison (GP vs LLM, matched budgets) and finance already has the overfitting statistics (Deflated Sharpe, PBO). Nobody has joined the three. That junction is your paper.

**The paper in one line:** a controlled testbed where ground truth is *known* — markets with no edge, markets with a planted edge — plus real data, measuring whether LLM mutation finds more true edge and manufactures less fake edge than genetic-programming and random mutation over the *same* strategy grammar at the *same* candidate budget, scored with proper multiple-testing statistics (PBO, Deflated Sharpe).

**Why it's resume-perfect for QR/QT intern–grad:** either outcome is a headline (LLM wins → first controlled evidence in trading; LLM ties random → "expensive random search," a spicy negative result), it needs no PhD, no paid data, laptop-class compute — and it demonstrates the exact skill quant interviews probe: skepticism about backtests under multiple testing.

**Est. cost:** ~10 weeks part-time (~8–10 h/wk), ~$50–150 API budget, free data. MVP proof-of-concept in one weekend (§7).

---

## 1. "Too close to them?" — the verdict

**Not too close, with two rules.**

1. **Don't do their follow-up for them.** Vincent announced his next step: run his system across a broad universe of tickers. If your paper is "his repo, more tickers, longer runs," that's his sequel with your name on it. Skip that lane entirely.
2. **Do answer a question he explicitly punted.** His Future Work literally hands out the research agenda: the LLM-vs-dumb-operator control ("the most important control still missing"), better nulls (block bootstrap), full multiple-testing accounting (PBO / Deflated Sharpe), transfer across universes. Answering one of those with your own harness, citing him, is exactly how research is supposed to work.

Practical notes:

- His repo is **MIT-licensed**, 5 commits, 0 stars — you may legally fork and build on it. For the resume, build your **own harness** anyway (§5): your paper's contribution *is* the evaluation, so the evaluation must be yours.
- **Email him.** Niche author, 412 followers, clearly wants this control run ("this is the most important control still missing" is an invitation). A short note — "I'm running the ablation you flagged; want to see the design?" — costs nothing, may get you feedback, a citation, a LinkedIn amplifier, and a networking contact at the AI/HFT crossroads. Worst case: silence.
- Cite AlgoEvolve and MadEvolve prominently; your gap table (§3) is *built* from their own stated limitations, which is the polite way to position against people.

## 2. What their article got right (the parts worth keeping)

Five ideas worth stealing as your foundations — all credit-cited:

1. **Structure from the model, numbers from the optimizer.** The LLM writes code + parameter *ranges*, never values; `differential_evolution` fits them. Cleanest division of labor in this literature.
2. **The null-max bar.** Rerun the *entire search* on sign-flipped returns and use the best score it wrings from noise as the bar. It tests the search process, not one strategy — most people only test the strategy.
3. **Constrained causal grammar.** Strategies can only call a fixed library of backward-looking indicators — every candidate is look-ahead-safe *by construction*, and the grammar makes operators comparable (key for your ablation).
4. **The walled-off holdout, read once** — and the honest read of it: he watched CV climb 1.47→1.63 while holdout moved 0.71→0.76 and called it what it was (the search fitting the folds). He also published that he *lost to buy-and-hold*. That honesty is the house style your paper should copy.
5. **Swappable mutation operator.** His mutation model is an environment variable. That design choice is precisely what makes the ablation cheap — the loop doesn't change, only the proposer does.

## 3. The gap map (what's taken vs. open)

| | Vincent's system | AlgoEvolve | MadEvolve | QuantaAlpha |
|---|---|---|---|---|
| LLM's role | mutation operator | semantic mutation + meta-evolved prompts | 5-model ensemble mutation | multi-agent trajectory evolution |
| Data | 1 stock (NVDA), daily close | intraday 5-min equities (NUMIN) | BTC minute bars | CSI 300 cross-sectional |
| Overfitting controls | quarter CV + null-max + holdout | walk-forward, cost penalty | chrono splits, impact costs, degradation analysis | complexity + redundancy checks |
| **GP / random-mutation baseline, same grammar** | ❌ (flagged as top missing control) | ❌ (confirmed in paper) | ❌ | ❌ (all baselines LLM-based) |
| **Matched compute/candidate budgets** | ❌ | ❌ | ❌ | ❌ |
| **PBO / Deflated Sharpe reporting** | ❌ (listed as future work) | ❌ | ❌ | ❌ |
| **Ground-truth (synthetic/planted-edge) worlds** | ❌ | ❌ | ❌ | ❌ |

Two adjacent literatures already solved their halves:

- **Program synthesis** runs the fair fight: "GP and LLMs for Program Synthesis: No Clear Winners" (arXiv 2508.03966) compares PushGP vs GPT-4o on PSB2 with explicit fairness rules — equal information, equal budgets. Nobody has ported those rules to trading. Bonus ammunition: "Mutation Without Variation" (arXiv 2606.05408) shows LLM mutation chains collapse to repeated structures (in 87% of chains, >93% of mutations revisit known forms) — a measured reason to *doubt* that LLM mutation explores better.
- **Finance statistics** has the multiple-testing toolkit: Probability of Backtest Overfitting and Deflated Sharpe Ratio (Bailey & López de Prado). Nobody has applied it to LLM-driven searches, which are multiple-testing *machines*.

Your paper = trading systems × fair-comparison methodology × overfitting statistics. Each corner exists; the junction is empty. That's the "new, edging" you asked for.

## 4. The recommended paper

**Working titles** (pick a vibe):

- *Fool's Gold: What LLM-Driven Trading-Strategy Search Actually Finds*
- *Does the LLM Earn Its Seat? A Controlled Study of Mutation Operators in Trading-Strategy Evolution*
- *Edge or Artifact? Ground-Truth Benchmarks for LLM Strategy Search*

**Research questions:**

- **RQ1 (power):** At a matched budget of scored candidates, does LLM mutation recover a *planted* edge more often than GP or random mutation over the same grammar?
- **RQ2 (false discovery):** In worlds with provably *no* edge, how much apparent out-of-sample Sharpe does each operator manufacture?
- **RQ3 (do the safety rails work):** Do null-max bars, PBO, and Deflated Sharpe correctly separate the planted-edge discoveries from the manufactured ones?

**Design — three worlds × four operators:**

Worlds:

1. **No-edge nulls** — GBM series and stationary block-bootstrap resamples of real returns (preserves vol clustering + fat tails; fixes Vincent's own complaint that sign-flipping is too weak a null). Any "edge" found here is fool's gold by construction.
2. **Planted-edge synthetics** — inject a weak, known signal (e.g., AR/momentum term or regime-switching drift calibrated to Sharpe ≈ 0.5–0.8) into null series. Ground truth known → you can score *recovery rate*, which no real-data study can.
3. **Real-data field test** — 50–100 liquid US names + ETFs, pre-2026 for search (quarter CV + null-max), 2026 as read-once holdout. Report the *distribution* of outcomes across tickers, never a champion.

Operators (same grammar, same `differential_evolution` fitting, same CV — only the proposer changes):

1. Strong hosted LLM (Claude / GPT-class)
2. Small local LLM (qwen2.5-coder:7b via Ollama — tests whether model quality matters)
3. Classic tree-GP (DEAP: subtree mutation + crossover over the indicator grammar)
4. Random grammar sampler (the floor)

Fairness rules ported from the program-synthesis comparison: identical information (scored history format), identical budget (N scored candidates, e.g. 300/run), ≥20 seeds per cell, GP hyperparameters tuned in good faith.

**Metrics that make it a paper, not a blog post:**

- Planted-edge recovery rate + budget-to-recovery curves
- Distribution of max OOS Sharpe on no-edge worlds per operator ("fool's gold curve")
- PBO and Deflated Sharpe for every champion; CV-gain vs holdout-gain slope (Vincent's divergence tell, made quantitative)
- Structural diversity over time (unique strategy skeletons per 100 candidates — connects to the Mutation Without Variation findings)
- Cost: wall-clock + $ per unit of *validated* Sharpe (the practitioner's number nobody reports)

**Why either result wins:** LLM clearly beats GP at matched budget → first controlled evidence in trading that semantic mutation earns its cost. LLM ≈ GP or random → the field's premise is an expensive placebo, and you have the receipts. Negative results with clean design are *more* citable here, because three published systems implicitly bet the other way.

## 5. Build it as a NEW repo (not in this one)

- New standalone repo (e.g. `strategy-search-bench`) — clean resume artifact with its own README, results tables, and a "add your operator" interface (benchmark framing invites others to submit operators → citations).
- This `strategy_evolution/` folder inside the Quant job-scraper repo stays as your notes/scratch space; the Quant repo remains the scraper's.
- Write the ~1,500-line harness yourself (grammar, backtester, CV, nulls, operators). Vincent's repo is reference material and a related-work citation, not your codebase. Your interviews will include "walk me through your backtester" — it must be yours to walk through.

## 6. Backup / extension angles (each is a standalone paper if the main one stalls)

1. **"LLM evolution rediscovers volatility targeting."** Take evolved champions and benchmark against exposure-matched vol-targeting, SMA timing, and drawdown filters. If evolved "edge" ≈ risk-control baselines, the loop is relearning 1990s risk management. Cheap, spicy, very defensible. (~4 wks)
2. **Frozen-structure transfer.** Evolve structures on universe A, freeze, evaluate on disjoint universe B and a later era. Separates transferable principle from per-ticker curve fitting. (~6 wks, heavier compute)
3. **Null engineering.** Turn the null-max bar into real inference: distribution of null-world maxima via block bootstrap, report P(S_null ≥ S_observed). Methods-paper flavored; folds naturally into the main paper as §RQ3 if you keep it. (~3 wks)

## 7. Timeline (part-time, ~8–10 h/wk)

| Weeks | Milestone |
|---|---|
| 1–2 | Grammar + backtester + quarter-CV harness; block-bootstrap + planted-edge generators |
| 3 | Random sampler + DEAP tree-GP operators running end-to-end |
| 4 | LLM operators (Ollama local + one hosted); prompt = Vincent-style fixed system message |
| 5–6 | Synthetic-world runs (nulls + planted edge), 20 seeds/cell; first fool's-gold curves |
| 7–8 | Real-data field test (50–100 tickers); PBO/DSR layer |
| 9 | Ablations + robustness (budget sensitivity, grammar sensitivity) |
| 10 | Write 8-page paper + README leaderboard; arXiv + SSRN + LinkedIn post |

**Weekend-1 MVP (do this first, ~2 days):** random sampler vs local LLM on pure-noise worlds only, one budget, 5 seeds. Output: one chart — "apparent OOS Sharpe found in provably edgeless markets, by operator." That chart alone tells you if the full build is worth it, and is already a strong LinkedIn post.

**Compute reality check:** the expensive part is DE fitting × CV splits (CPU), not LLM calls. Mitigations: 25 splits during search / 100 only for finalists, cache indicator arrays, `multiprocessing` across candidates, and a cheap big-CPU cloud box (~€40) for the final runs if the laptop chokes.

## 8. Resume & interview payoff

CV line when done:

> *Built an open-source benchmark testing whether LLM mutation beats genetic-programming baselines in trading-strategy search under matched budgets; measured false-edge manufacture on ground-truth null markets using Deflated Sharpe / PBO. Paper on arXiv; harness reused by others.*

Interview talking points it buys you (these are *the* QR interview themes): selection bias under multiple testing, why CV Sharpe rises while holdout stalls, why sign-flip nulls are weak vs block bootstrap, DSR/PBO mechanics, and "here's the experiment I designed to falsify my own hypothesis." For intern/grad roles, "I built a testbed that catches fake edge" beats "my bot beat the market" in front of every serious interviewer — claimed alpha from a student is a red flag; demonstrated skepticism is the green one.

Distribution plan: arXiv (q-fin.CP, cross-list cs.NE) + SSRN → LinkedIn thread tagging Vincent + AlgoEvolve authors → repo README as the living leaderboard.

**Venues** (paper works without any of them; these are upside):

- ICAIF 2026 main track closed Aug 9, 2026 (missed by 2 days) — but **watch its workshop CFPs** (typically September deadlines), Milan, Nov 14–17
- **GECCO 2027** (evolutionary-computation crowd; LLM-vs-GP is their home turf; deadline usually ~Feb)
- ICAIF 2027 main track as the full-paper target

## 9. Risks / honesty box

- **GP fairness is the attack surface.** A lazy GP baseline invalidates the paper. Tune it in good faith, document the tuning, cite the No Clear Winners fairness rules as your protocol.
- **One holdout year is noisy** — many tickers and many synthetic seeds are your replication, not 2026 alone.
- **yfinance survivorship bias** — say it out loud in limitations; lean on ETFs + still-listed large caps; claims are about *search behavior*, not tradability.
- **Never claim tradability.** Costs are flat per-trade; no slippage/impact. The paper measures search operators, not P&L.
- **Scope creep kills this.** Freeze the three RQs; everything else goes to §6.

## References

- Vincent M., [Why LLMs Can't Trade — and How to Use Them in Trading](https://www.linkedin.com/pulse/why-llms-cant-trade-how-use-them-trading-vincent-maciejewski-2p9rf/) + [repo (MIT)](https://github.com/vincent212/trading_strategy_evolution_agent)
- [AlgoEvolve — arXiv:2606.26173](https://arxiv.org/abs/2606.26173) · [MadEvolve — arXiv:2605.23007](https://arxiv.org/abs/2605.23007) · [QuantaAlpha — arXiv:2602.07085](https://arxiv.org/html/2602.07085v2)
- [GP and LLMs for Program Synthesis: No Clear Winners — arXiv:2508.03966](https://arxiv.org/pdf/2508.03966) · [Mutation Without Variation — arXiv:2606.05408](https://arxiv.org/html/2606.05408)
- Bailey & López de Prado — [The Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) · [The Probability of Backtest Overfitting](https://www.researchgate.net/publication/318600389_The_probability_of_backtest_overfitting)
- Romera-Paredes et al., [FunSearch, Nature 2023](https://www.nature.com/articles/s41586-023-06924-6) · Bergmeir, Hyndman & Koo (2018) on CV validity for time series
- [ICAIF 2026 CFP](https://icaif2026.org/call-for-papers.html) (main deadline passed Aug 9; watch workshops)
