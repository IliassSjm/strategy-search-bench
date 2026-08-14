# strategy-search-bench

**Does an LLM earn its seat as a mutation operator in trading-strategy search —
or is it an expensive random-number generator?**

This bench compares proposal operators (random search, tree-GP, an LLM) over the
**same** strategy grammar, with the **same** parameter-fitting and cross-validation,
at a **matched budget of scored candidates** — in worlds where the ground truth is
*known*: markets with provably no edge, and markets with a planted edge of known
size. Published LLM-evolution trading systems (AlgoEvolve, MadEvolve, QuantaAlpha,
and the write-up that inspired this bench) all skip this control.

## TL;DR quickstart (the weekend experiment)

```bash
pip install -r requirements.txt
pytest                                   # 16 tests, ~3s — look-ahead safety included

# Fool's gold: what does each operator "find" where NOTHING exists?
python run_experiment.py --operator random --world gbm --budget 60 --seeds 0 1 2 3 4
python run_experiment.py --operator gp     --world gbm --budget 60 --seeds 0 1 2 3 4

# LLM operator: start Ollama first (or see "Model options" below)
ollama pull qwen2.5-coder:7b
export LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5-coder:7b
python run_experiment.py --operator llm    --world gbm --budget 60 --seeds 0 1 2 3 4

python make_report.py runs/              # table + fools_gold_gbm.png
```

Model options for `--operator llm` (env vars only, no code changes):

- **Local, free (no account):** Ollama as above. Bigger free arms if you have
  the RAM, e.g. `qwen2.5-coder:14b`.
- **Hosted big models, free tier:** any OpenAI-compatible endpoint. Groq:
  `export LLM_BASE_URL=https://api.groq.com/openai/v1 LLM_API_KEY=gsk_...
  LLM_MODEL=<id from Groq console>`. `LLM_API_KEY` is sent as a Bearer token
  when set.
- **Anthropic API (paid credits):** unset `LLM_BASE_URL`, set
  `ANTHROPIC_API_KEY` and `LLM_MODEL`.

Runtime scales with `--budget x --splits x --de-iter x --de-pop`; `--workers`
(default: cores−2) parallelizes across CV splits. Start smaller (`--budget 20
--seeds 0 1`) to gauge your machine.

## Design

```
operator.propose(history) ──> tree (STRUCTURE only, shared grammar)
        │                          │
        │                          ▼
        │        differential_evolution fits parameter VALUES per split
        │                          │
        │                          ▼
        │        median out-of-sample Sharpe over random quarter splits
        │                          │
        └────── observe(record) ◄──┘        (budget = N scored candidates)

after the loop: champion refit on the search region, then ONE read of the
walled-off final ~350 bars (the holdout).
```

- **Grammar** (`ssbench/grammar.py`): expression trees over causal indicator
  leaves (`trend`, `momentum`, `meanrev_z`, `rsi_gate`, `breakout`, `const_long`)
  and nonlinear combinators (`wsum`, `vol_gate`, `switch_z`). Parameter *ranges*
  are fixed per node type; proposers never set values. Depth ≤ 3, ≤ 9 nodes.
- **Worlds** (`ssbench/worlds.py`):
  - `gbm` — zero-drift GBM: no timing edge exists, by construction.
  - `bootstrap` — stationary block bootstrap of a demeaned real return series
    (vol clustering and fat tails survive; drift and cross-block
    predictability do not). A realistic but only *approximately* edgeless
    null — sub-block autocorrelation survives; `gbm` is the strict one. Needs
    `scripts/fetch_data.py SPY` then `--source-csv data/SPY.csv`.
  - `ar1` — planted edge: AR(1) returns with known `--phi`;
    `worlds.oracle_ar1_sharpe()` gives the recoverable ceiling per series.
- **Operators** (`ssbench/operators.py`): `random` (fresh sample each call — the
  floor), `gp` (tournament selection + subtree crossover/mutation, dependency-free),
  `llm` (scored history in, one JSON tree out; Ollama / OpenAI-compatible /
  Anthropic via env vars; invalid replies get 2 repair attempts then count as
  rejects), `mockllm` (offline stand-in exercising the same parse/repair path).
- **Fairness contract** (`ssbench/runner.py`): a run ends after N *scored*
  candidates for every operator; duplicates and rejects are logged overhead.
  Splits, worlds, DE seeds all deterministic per run seed.

## What the headline figure means

`make_report.py` plots, per operator: champion **CV fitness** (what the search
believed) vs **holdout Sharpe** (what the walled-off data paid), across seeds.
On no-edge worlds the CV cloud floats above zero and the holdout cloud straddles
it — the vertical gap is manufactured edge, measured. On `ar1` worlds the
question flips: which operator closes the gap to the oracle ceiling with the
fewest candidates?

## Roadmap to the paper

1. **MVP (this repo, one weekend):** random vs gp vs llm on `gbm`, 5 seeds,
   budget 60 — the fool's-gold figure.
2. Planted-edge recovery curves on `ar1` (budget on x, fraction-of-oracle on y).
3. `bootstrap` worlds; real-data field test across ~50 tickers.
4. PBO + Deflated Sharpe layer on every champion; reject/duplicate overhead and
   wall-clock/$ per validated Sharpe as first-class metrics.
5. Ablations: history shown to the LLM (whole vs top-k), exploration turns,
   grammar size; structural-diversity-over-time per operator.

## Known limitations

- **Runs are only comparable under one grammar.** Any change to
  `grammar.NODE_SPECS` (types, ranges, caps) changes the search space —
  rerun every arm you intend to compare. Old runs in `runs/` don't mix with
  new ones.
- Test scoring applies a 1-bar purge (first bar of each test block and of the
  holdout is dropped — its position was formed on train/search data), but
  rolling indicator windows still span quarter boundaries. That's inherent to
  scoring a structure across non-chronological quarters; a chronological
  walk-forward answers the deployment question and belongs in the paper as a
  secondary diagnostic.
- Costs are a flat per-side fee on turnover; no slippage, impact, or borrow.
- One asset, daily closes only. Fitness is Sharpe-only for now (PBO /
  Deflated Sharpe are roadmap item 4).

Design notes and related work: see `PAPER_PLAN.md` and `NOTES.md` in this
folder. Not investment advice; nothing here claims tradability.
