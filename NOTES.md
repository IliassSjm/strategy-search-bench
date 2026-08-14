# LLM Strategy Evolution — test notes

> Source: [Why LLMs Can't Trade — and How to Use Them in Trading](https://www.linkedin.com/pulse/why-llms-cant-trade-how-use-them-trading-vincent-maciejewski-2p9rf/) (Vincent M., Aug 10, 2026)
> Repo: https://github.com/vincent212/trading_strategy_evolution_agent

## TL;DR — the one thing to remember

An LLM can write trading strategies but **cannot tell a real edge from a pattern that only looks good in historical data**. So never let it judge or trade. Use it only as a **search engine** — proposing and mutating strategy *structures* — while a framework it never controls does the judging: an optimizer fits the numbers, out-of-sample validation decides what survives.

> "Don't ask the LLM to trade. Ask it to search."

**The receipt:** in the author's NVDA run, the champion the search scored best (CV Sharpe 1.63) was overfit — on the walled-off 2026 data it lost to plain buy-and-hold. The LLM would have shipped it. The validation framework caught it.

## The division of labor (the design worth copying)

1. **LLM = mutation operator only.** Proposes strategy code + parameter *ranges* (`param_space()`), never values. Never sees market data, never told whether its strategy worked.
2. **`differential_evolution` fits the numbers.** The model decides which knobs exist; the optimizer sets them.
3. **Quarter-level cross-validation scores it.** Fit on 75% of calendar quarters, score on the withheld 25%, 100 random splits → median OOS Sharpe.
4. **Null-max bar.** Rerun the same search on sign-flipped returns; the best "edge" it wrings out of pure noise is the bar a real candidate must clear.
5. **One untouched holdout (2026), read exactly once.** The gap to watch: when CV keeps rising but the holdout stops moving, the search has started fitting the folds, not the market.

## Run it

```bash
brew install ollama
ollama serve                      # leave running
ollama pull qwen2.5-coder:7b      # ~4.7 GB

pip install -r requirements.txt
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=qwen2.5-coder:7b
python run.py --ticker NVDA --start 2017-01-01 --iterations 300
```

- Progress: `runs/<TICKER>_progress.log` → result in `runs/run_<TICKER>_*.json`
- Wall-clock cost is mostly CPU (fitting each candidate across 100 splits), not LLM calls
- Stronger model instead of the 7B: set `ANTHROPIC_API_KEY`, or `LLM_PROVIDER=subagent` for a Claude Code subagent (near-zero rejected candidates vs ~50% for the 7B)
- Return-max instead of Sharpe-max: `--objective return --min-sharpe 0.8` (NVDA run: ~93% avg exposure vs ~29% — the objective shapes the book)

## Caveats the author flags himself

- One stock (NVDA), one holdout year, ~50-iteration searches — illustrative, not proof
- It did **not** beat buy-and-hold on 2026; the edge it finds is risk control, not return
- Tool library = classical technical indicators only — crowded, mostly arbitraged; the library is the ceiling
- Biggest open question (his words): does the LLM beat random/genetic mutation over the same grammar at all?

## Test ideas for this folder

- [ ] Clone the repo, reproduce the NVDA run with the local 7B
- [ ] Try a second ticker (does anything carry over, or is it NVDA-only?)
- [ ] Watch the CV-vs-holdout gap live in the progress log — the overfitting moment is the interesting part
