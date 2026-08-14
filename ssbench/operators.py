"""Mutation/proposal operators. All share one interface:

    propose(history) -> tree          history: list of {"tree", "fitness"} dicts,
    observe(record)  -> None          called after each candidate is scored.

Every operator emits trees in the SAME grammar (ssbench.grammar); parameter
values are never chosen here — the evaluator fits them. Budget fairness is
enforced by the runner: a run ends after N *scored* candidates regardless of
operator; rejected/duplicate proposals are logged as overhead.
"""

import copy
import json
import os
import re
import time

import numpy as np
import requests

from . import grammar


class ProposalError(RuntimeError):
    """Operator failed to produce a valid tree (counts as a reject)."""


# ---------------------------------------------------------------------------
# Random search: fresh sample every time. The floor every other operator
# must beat.
# ---------------------------------------------------------------------------

class RandomOp:
    name = "random"

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def propose(self, history):
        return grammar.random_tree(self.rng)

    def observe(self, record):
        pass


# ---------------------------------------------------------------------------
# Tree GP: tournament selection + subtree crossover + subtree mutation.
# Deliberately plain (no DEAP dependency) so reviewers can read all of it.
# ---------------------------------------------------------------------------

class GPOp:
    name = "gp"

    def __init__(self, seed: int, pop_size: int = 24, init_size: int = 12,
                 p_mutate: float = 0.35, tournament: int = 3):
        self.rng = np.random.default_rng(seed)
        self.pop = []                     # list of {"tree", "fitness"}
        self.pop_size = pop_size
        self.init_size = init_size
        self.p_mutate = p_mutate
        self.tournament = tournament

    def _tournament_pick(self):
        idx = self.rng.integers(len(self.pop), size=min(self.tournament, len(self.pop)))
        best = max(idx, key=lambda i: self.pop[i]["fitness"])
        return self.pop[best]["tree"]

    def _random_subtree_path(self, tree):
        """Uniformly pick a node; return its path (list of child indices)."""
        paths = []

        def walk(node, path):
            paths.append(path)
            for i, c in enumerate(node.get("children", [])):
                walk(c, path + [i])

        walk(tree, [])
        return paths[self.rng.integers(len(paths))]

    @staticmethod
    def _get(tree, path):
        node = tree
        for i in path:
            node = node["children"][i]
        return node

    @staticmethod
    def _set(tree, path, subtree):
        if not path:
            return subtree
        node = tree
        for i in path[:-1]:
            node = node["children"][i]
        node["children"][path[-1]] = subtree
        return tree

    def propose(self, history):
        if len(self.pop) < self.init_size:
            return grammar.random_tree(self.rng)
        a = copy.deepcopy(self._tournament_pick())
        if self.rng.random() < self.p_mutate or len(self.pop) < 2:
            # subtree mutation: replace a random node with a fresh random tree
            path = self._random_subtree_path(a)
            child = self._set(a, path, grammar.random_tree(self.rng, depth=len(path)))
        else:
            # subtree crossover: graft a random subtree of b into a
            b = self._tournament_pick()
            pa = self._random_subtree_path(a)
            pb = self._random_subtree_path(b)
            child = self._set(a, pa, copy.deepcopy(self._get(b, pb)))
        try:
            grammar.validate(child)
        except grammar.InvalidTree:
            child = grammar.random_tree(self.rng)   # size cap blown: resample
        return child

    def observe(self, record):
        self.pop.append({"tree": record["tree"], "fitness": record["fitness"]})
        if len(self.pop) > self.pop_size:
            worst = min(range(len(self.pop)), key=lambda i: self.pop[i]["fitness"])
            self.pop.pop(worst)


# ---------------------------------------------------------------------------
# LLM operator: the model sees the scored history and returns ONE JSON tree.
# Endpoint config (same conventions as Vincent's repo):
#   LLM_BASE_URL (default http://localhost:11434/v1)  + LLM_MODEL   -> OpenAI-compatible
#   ANTHROPIC_API_KEY (if LLM_BASE_URL unset)                       -> Anthropic API
# ---------------------------------------------------------------------------

GRAMMAR_SPEC = """You evolve trading-strategy STRUCTURES as JSON expression trees.

Node types (children count):
LEAVES: "const_long"(0) always long | "trend"(0) SMA fast>slow | "momentum"(0) sign of n-bar return | "meanrev_z"(0) fade the z-score | "rsi_gate"(0) long oversold, short overbought | "breakout"(0) rolling high/low breaks
COMBINATORS: "wsum"(2) weighted sum of children | "vol_gate"(2) child A in calm vol regime, child B in wild | "switch_z"(2) child A when z<0 else child B

Format: {"type": "...", "children": [...]}  — leaves take "children": [].
Max depth 3, max 9 nodes. Do NOT include parameter values anywhere: every
node's windows/weights/thresholds are fitted by a separate optimizer.
Nonlinear structure (vol_gate / switch_z) usually beats flat sums.

Reply with ONLY the JSON tree. No prose, no code fences."""


def _fitness_line(rec):
    return f'score={rec["fitness"]:+.3f}  {grammar.describe(rec["tree"])}  {grammar.canonical(rec["tree"])}'


class LLMOp:
    name = "llm"

    def __init__(self, seed: int, explore_every: int = 5, max_repairs: int = 2,
                 history_cap: int = 25, timeout: int = 120):
        self.calls = 0
        self.explore_every = explore_every
        self.max_repairs = max_repairs
        self.history_cap = history_cap
        self.timeout = timeout
        self.base_url = os.environ.get("LLM_BASE_URL")
        self.model = os.environ.get("LLM_MODEL", "qwen2.5-coder:7b")
        self.api_key = os.environ.get("LLM_API_KEY")      # for authed
        # OpenAI-compatible endpoints (Groq, etc.); Ollama needs none
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.base_url and not self.anthropic_key:
            self.base_url = "http://localhost:11434/v1"

    # -- transport ----------------------------------------------------------
    def _complete(self, system: str, user: str) -> str:
        if self.base_url:
            headers = ({"Authorization": f"Bearer {self.api_key}"}
                       if self.api_key else {})
            r = requests.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json={"model": self.model, "temperature": 1.0,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}]},
                timeout=self.timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.anthropic_key,
                     "anthropic-version": "2023-06-01"},
            json={"model": os.environ.get("LLM_MODEL", "claude-sonnet-4-5"),
                  "max_tokens": 1000, "system": system,
                  "messages": [{"role": "user", "content": user}]},
            timeout=self.timeout)
        r.raise_for_status()
        return r.json()["content"][0]["text"]

    # -- prompt -------------------------------------------------------------
    def _user_message(self, history) -> str:
        self.calls += 1
        ranked = sorted(history, key=lambda r: r["fitness"], reverse=True)
        shown = ranked[: self.history_cap]
        lines = "\n".join(_fitness_line(r) for r in shown) or "(no strategies scored yet)"
        if self.explore_every and self.calls % self.explore_every == 0:
            task = ("EXPLORATION TURN: ignore the scores. Propose a structure "
                    "UNLIKE anything above.")
        else:
            task = ("Propose ONE new tree that should BEAT the best score above: "
                    "keep mechanisms that scored well, drop what scored poorly, "
                    "recombine NONLINEARLY. Do not repeat a listed structure.")
        return f"Strategies tried so far (best first):\n{lines}\n\n{task}"

    # -- parsing ------------------------------------------------------------
    @staticmethod
    def _extract_tree(text: str):
        text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE)
        start = text.find("{")
        if start < 0:
            raise grammar.InvalidTree("no JSON object in reply")
        depth = 0
        for i, ch in enumerate(text[start:], start):
            depth += ch == "{"
            depth -= ch == "}"
            if depth == 0:
                tree = json.loads(text[start:i + 1])
                grammar.validate(tree)
                return tree
        raise grammar.InvalidTree("unbalanced JSON object")

    def propose(self, history):
        user = self._user_message(history)
        last_err = None
        for attempt in range(1 + self.max_repairs):
            try:
                text = self._complete(GRAMMAR_SPEC, user)
            except requests.RequestException as e:
                # Endpoint down / 429 / timeout: back off and burn an attempt
                # instead of crashing the run. Persistent outages surface as
                # rejects and eventually the runner's overhead cap.
                last_err = e
                time.sleep(min(2.0 * 2 ** attempt, 30.0))
                continue
            try:
                return self._extract_tree(text)
            except (grammar.InvalidTree, json.JSONDecodeError, ValueError) as e:
                last_err = e
                user = (f"Your previous reply was invalid: {e}\n"
                        f"Reply with ONLY a valid JSON tree per the rules.")
        raise ProposalError(f"LLM failed after {self.max_repairs + 1} tries: {last_err}")

    def observe(self, record):
        pass


# ---------------------------------------------------------------------------
# Mock LLM: offline stand-in with the same parse/repair path (for tests and
# for exercising the pipeline without a model).
# ---------------------------------------------------------------------------

class MockLLMOp(LLMOp):
    name = "mockllm"

    def __init__(self, seed: int, **kw):
        super().__init__(seed, **kw)
        self.rng = np.random.default_rng(seed)

    def _complete(self, system, user):
        roll = self.rng.random()
        if roll < 0.15:
            return "Here is my strategy: I think trend following is best."
        if roll < 0.30:
            return '{"type": "made_up_node", "children": []}'
        tree = grammar.random_tree(self.rng)
        return f"```json\n{grammar.canonical(tree)}\n```"


OPERATORS = {"random": RandomOp, "gp": GPOp, "llm": LLMOp, "mockllm": MockLLMOp}


def make_operator(name: str, seed: int):
    try:
        return OPERATORS[name](seed)
    except KeyError:
        raise ValueError(f"unknown operator {name!r} "
                         f"(choose from {sorted(OPERATORS)})") from None
