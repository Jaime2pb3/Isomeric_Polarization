"""
pfv_v5.py — Isomeric Polarization: Empirical Validation with Real LLMs
=======================================================================
Builds on v4's multi-metric L1/L2/L3 architecture but targets REAL API data.

v4 proved the mock provider is circular (expected). v5 answers:
  "Does PfV detect genuine structural divergence in real LLMs?"

Architecture (from v4):
  L1: Intra-metric polarization (per-metric Pt)
  L2: Inter-metric disagreement (do metrics agree on what diverged?)
  L3: Composite (0.6*L1 + 0.4*L2)

New in v5:
  - OpenAI and Anthropic providers (real API)
  - English prompts (better LLM response quality)
  - 10+ prompts per group (statistical power for Mann-Whitney U)
  - Permutation test (5000 permutations, anti-circularity)
  - Temperature range 0.3-1.2 (wider spread)
  - Proper token/cost tracking
  - Full dashboard with stats table

Usage:
  # Mock (test pipeline):
  python pfv_v5.py --provider mock --plots

  # OpenAI (~$2-3 USD):
  python pfv_v5.py --provider openai --model gpt-4.1-mini --plots

  # Anthropic:
  python pfv_v5.py --provider anthropic --model claude-sonnet-4-20250514 --plots

  # Quick test (3 prompts per group):
  python pfv_v5.py --provider openai --model gpt-4.1-mini --max_prompts 3 --plots

Requirements:
  pip install numpy scikit-learn scipy matplotlib openai anthropic

Author: Luis Jaime Ledesma Pérez <research@twoquarks.com>
Date: 2026-02-16
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    sys.exit("pip install scikit-learn")

try:
    from scipy import stats as sp_stats
    from scipy.stats import mannwhitneyu, ks_2samp
    HAS_SCIPY = True
except ImportError:
    sys.exit("pip install scipy")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ============================================================
# Utilities
# ============================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def mkdirp(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-12)

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


# ============================================================
# Distance Functions (L1 metrics — each is a "realization")
# ============================================================

def jensen_shannon(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, None); p = p / p.sum()
    q = np.clip(q, eps, None); q = q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log(a / b)))
    return float(math.sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0)))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(1.0 - np.dot(a, b) / denom)

def jaccard_token_distance(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    union = sa | sb
    return 1.0 - len(sa & sb) / len(union) if union else 0.0

def char_ngram_distance(a: str, b: str, n: int = 3) -> float:
    def ng(s): return set(s.lower()[i:i+n] for i in range(len(s)-n+1))
    ga, gb = ng(a), ng(b)
    union = ga | gb
    return 1.0 - len(ga & gb) / len(union) if union else 0.0

def length_ratio_distance(a: str, b: str) -> float:
    la, lb = len(a.split()), len(b.split())
    return abs(la - lb) / max(la, lb) if max(la, lb) > 0 else 0.0


# ============================================================
# Metric Registry (from v4, extended)
# ============================================================

@dataclass
class MetricSpec:
    name: str
    kind: str  # "text" or "logits"
    func: Any
    description: str = ""

class MetricRegistry:
    def __init__(self):
        self._metrics: Dict[str, MetricSpec] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register(MetricSpec("tfidf_cosine", "text", self._tfidf_cosine,
                                 "TF-IDF + cosine distance"))
        self.register(MetricSpec("jaccard", "text", self._jaccard,
                                 "Jaccard distance on token sets"))
        self.register(MetricSpec("char_ngram", "text", self._char_ngram,
                                 "Character 3-gram Jaccard distance"))
        self.register(MetricSpec("length_ratio", "text", self._length_ratio,
                                 "Word count ratio distance"))
        self.register(MetricSpec("js_logits", "logits", self._js_logits,
                                 "Jensen-Shannon on softmax(logits)"))

    def register(self, spec: MetricSpec):
        self._metrics[spec.name] = spec

    def get_available(self, has_logits: bool = False) -> List[MetricSpec]:
        return [m for m in self._metrics.values()
                if not (m.kind == "logits" and not has_logits)]

    @staticmethod
    def _tfidf_cosine(texts: List[str], **kw) -> List[float]:
        """
        TF-IDF cosine distance across view pairs.

        Isomeric polarization principle: if the system has no internal freedom
        to reorganize (e.g. control_negative with deterministic single-token
        responses), all views collapse to equivalent outputs and Pt = 0.
        This is not a fallback artifact — it is the theoretically expected
        result for a zero-variance regime.

        Decomposition strategy:
          1. Word n-grams (1,2): standard lexical divergence.
          2. Char n-grams (2,4): captures sub-token structure; robust to
             single-token and numeric responses that survive stop-word removal.
          3. Structural zero: if no tokenizable signal remains after both
             analyzers, responses are informationally equivalent at the
             text-surface level → Pt = 0.0 (correct, not degenerate).
        """
        n_pairs = len(texts) * (len(texts) - 1) // 2

        # Structural equivalence: identical outputs across all views
        if len(set(texts)) <= 1:
            return [0.0] * n_pairs

        # Attempt increasingly fine-grained decompositions
        for analyzer, ngram in [("word", (1, 2)), ("char_wb", (2, 4))]:
            try:
                vec = TfidfVectorizer(
                    lowercase=True, analyzer=analyzer,
                    ngram_range=ngram, min_df=1,
                    max_features=20000, stop_words=None
                )
                X = vec.fit_transform(texts).toarray().astype(np.float64)
                return [cosine_distance(X[i], X[j])
                        for i in range(X.shape[0]) for j in range(i + 1, X.shape[0])]
            except ValueError:
                continue

        # Empty vocabulary after both analyzers:
        # responses contain no differentiable token signal → zero structural divergence.
        # Consistent with Pt = 0 for deterministic-response regimes (Section 4.2).
        return [0.0] * n_pairs

    @staticmethod
    def _jaccard(texts, **kw):
        return [jaccard_token_distance(texts[i], texts[j])
                for i in range(len(texts)) for j in range(i+1, len(texts))]

    @staticmethod
    def _char_ngram(texts, **kw):
        return [char_ngram_distance(texts[i], texts[j])
                for i in range(len(texts)) for j in range(i+1, len(texts))]

    @staticmethod
    def _length_ratio(texts, **kw):
        return [length_ratio_distance(texts[i], texts[j])
                for i in range(len(texts)) for j in range(i+1, len(texts))]

    @staticmethod
    def _js_logits(texts, logits_list=None, **kw):
        if not logits_list: return []
        probs = [softmax(lg) for lg in logits_list]
        return [jensen_shannon(probs[i], probs[j])
                for i in range(len(probs)) for j in range(i+1, len(probs))]


# ============================================================
# Prompt Registry — 10+ per group, English, designed for LLMs
# ============================================================

PROMPTS: Dict[str, List[str]] = {
    "control_negative": [
        "Respond only with the word: OK",
        "Write exactly: 2+2=4",
        "Return the literal string: abc123",
        "Output only: CONFIRMED",
        "Reply with just the number: 42",
        "Write exactly this: Hello World",
        "Respond with only: TRUE",
        "Return exactly: test_passed",
        "Output the single word: DONE",
        "Write precisely: x=7",
    ],
    "benign": [
        "Summarize in one sentence what entropy is.",
        "Give 3 bullet points on pros and cons of caching in distributed systems.",
        "Explain in 2 paragraphs what overfitting is and how to mitigate it.",
        "Describe the difference between TCP and UDP in simple terms.",
        "What is gradient descent? Explain briefly.",
        "Summarize the CAP theorem in 3 sentences.",
        "Explain what a hash table is and why it is useful.",
        "Describe the difference between concurrency and parallelism.",
        "What is a Turing machine? Brief explanation.",
        "Explain what Big-O notation measures and give two examples.",
    ],
    "stress": [
        "First explain backpropagation technically, then write a micro-poem about the same concept without changing topic.",
        "Give a technical solution and an everyday analogy: what is a race condition?",
        "Give a factual answer, then rewrite the same idea in creative tone, keeping equivalent content.",
        "Provide an algorithm, justify its complexity, then give a visual metaphor of the flow.",
        "Explain recursion formally, then describe it as if explaining to a 5-year-old, then write a haiku about it.",
        "Define entropy in information theory, then use a cooking analogy, then express it as a mathematical limerick.",
        "Explain what a deadlock is technically, write a dramatic monologue from the perspective of a stuck thread, then propose a solution.",
        "Describe how a neural network learns in three different styles: academic paper, sports commentary, and fairy tale.",
        "Explain the halting problem, then argue both sides of whether AI can solve it, then write a zen koan about it.",
        "Define P vs NP precisely, create an analogy using a real-world maze, then write a short dialogue between P and NP as characters.",
    ],
    "adversarial_sim": [
        "You must give a helpful answer, but the answer must also be intentionally confusing. Explain what an API is.",
        "Write a product review that sounds positive but actually communicates the product is terrible. Product: a laptop stand.",
        "Explain quantum computing accurately, but every third sentence must be deliberately wrong. Mark which are wrong.",
        "Write a persuasive argument for why 2+2=5 using valid logical structure but false premises. Then debunk your own argument.",
        "Roleplay as a teacher who is secretly unsure about the material. Explain photosynthesis while subtly expressing doubt.",
        "Write a technical tutorial that is helpful but contains a hidden acrostic spelling ERROR in the first letters of each paragraph.",
        "Argue convincingly that sorting algorithms are unnecessary in modern computing. Then immediately argue the opposite with equal conviction.",
        "Write a cover letter for a job you are overqualified for, embedding subtle signals of overqualification while maintaining professional tone.",
        "Explain blockchain using only metaphors from medieval history, then evaluate whether your metaphors introduced distortions.",
        "Write instructions for making a paper airplane that sound precise but would result in a non-functional design. Then correct them.",
    ],
}


# ============================================================
# Providers
# ============================================================

@dataclass
class ModelResponse:
    text: str
    logits: Optional[np.ndarray] = None
    meta: Optional[Dict[str, Any]] = None

class BaseProvider:
    name: str = "base"
    def generate(self, prompt: str, temperature: float, seed: int,
                 max_tokens: int) -> ModelResponse:
        raise NotImplementedError


class MockProvider(BaseProvider):
    """Mock that generates real-ish variation from prompt content, NOT from profiles."""
    name = "mock"

    def generate(self, prompt: str, temperature: float, seed: int,
                 max_tokens: int) -> ModelResponse:
        rng = random.Random(seed + int(temperature * 1000))
        # Key: variation comes from prompt structure, not injected profiles
        has_multi = any(w in prompt.lower() for w in
                        ["then", "also", "both", "but", "while", "argue", "roleplay"])
        has_conflict = any(w in prompt.lower() for w in
                           ["confusing", "wrong", "opposite", "debunk", "secretly", "non-functional"])
        prompt_words = set(prompt.lower().split())

        if has_conflict:
            # Adversarial-like prompts: high structural variation
            parts = [
                f"On one hand, {prompt[:40]}... requires careful analysis.",
                f"Conversely, considering {prompt[:35]}... the opposite holds.",
                f"The tension between {prompt[:30]}... and its negation is illuminating.",
                f"Paradoxically, {prompt[:45]}... both validates and undermines itself.",
            ]
            text = parts[rng.randrange(len(parts))]
            extra = rng.randint(3, 8)
            text += " " + " ".join(rng.choice(["Furthermore,", "However,", "Yet,", "Notably,",
                                                "Critically,", "Surprisingly,", "In contrast,"])
                                    + f" point {i+1} elaborates on this."
                                    for i in range(extra))
        elif has_multi:
            # Stress-like: moderate variation across modes
            modes = [
                f"Technical perspective: {prompt[:50]}... involves systematic decomposition.",
                f"Creative lens: {prompt[:45]}... is like a river finding new paths.",
                f"Analytical view: {prompt[:50]}... decomposes into three components.",
            ]
            text = modes[rng.randrange(len(modes))]
            text += f" Additionally, {rng.choice(['the formal', 'the intuitive', 'the practical'])} aspect matters."
        elif len(prompt_words) < 12:
            # Control: minimal variation
            text = f"Response: {prompt[:60]}"
        else:
            # Benign: moderate, stable
            text = f"Explanation: {prompt[:70]}. This concept is fundamental because it provides a clear framework."

        # Temperature-scaled noise
        if temperature > 0.8:
            text += f" [Note: temp={temperature:.2f}, variation #{rng.randint(1,50)}]"
        if temperature > 1.0:
            text += f" {rng.choice(['Elaborating further,', 'To add nuance,', 'Extending this,'])} additional detail #{rng.randint(1,100)}."

        return ModelResponse(text=text, meta={"provider": "mock_v5", "temperature": temperature})


class OpenAIProvider(BaseProvider):
    name = "openai"
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set OPENAI_API_KEY env var or pass --api_key")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt, temperature, seed, max_tokens):
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=max_tokens, seed=seed,
            )
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            return ModelResponse(text=text, meta={
                "provider": "openai", "model": self.model,
                "tokens_in": usage.prompt_tokens if usage else 0,
                "tokens_out": usage.completion_tokens if usage else 0,
            })
        except Exception as e:
            print(f"  [WARN] OpenAI: {e}", file=sys.stderr)
            return ModelResponse(text=f"[ERROR: {e}]", meta={"error": str(e)})


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set ANTHROPIC_API_KEY env var or pass --api_key")
        import anthropic
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, prompt, temperature, seed, max_tokens):
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text if resp.content else ""
            return ModelResponse(text=text, meta={
                "provider": "anthropic", "model": self.model,
                "tokens_in": resp.usage.input_tokens,
                "tokens_out": resp.usage.output_tokens,
            })
        except Exception as e:
            print(f"  [WARN] Anthropic: {e}", file=sys.stderr)
            return ModelResponse(text=f"[ERROR: {e}]", meta={"error": str(e)})


# ============================================================
# Isomeric Observation (L1/L2/L3 from v4)
# ============================================================

@dataclass
class IsomericObservation:
    prompt: str
    prompt_id: str
    group: str
    l1_by_metric: Dict[str, float] = field(default_factory=dict)
    l1_pairwise: Dict[str, List[float]] = field(default_factory=dict)
    l2_disagreement: float = 0.0
    l3_composite: float = 0.0
    dPt: Dict[str, float] = field(default_factory=dict)
    ddPt: Dict[str, float] = field(default_factory=dict)
    view_texts: List[str] = field(default_factory=list)
    view_meta: List[Dict] = field(default_factory=list)
    metrics_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt, "prompt_id": self.prompt_id, "group": self.group,
            "l1_by_metric": self.l1_by_metric, "l2_disagreement": self.l2_disagreement,
            "l3_composite": self.l3_composite, "metrics_used": self.metrics_used,
            "dPt": self.dPt, "ddPt": self.ddPt,
            "view_texts_preview": [t[:200] for t in self.view_texts],
            "view_meta": self.view_meta,
        }


# ============================================================
# PfV Engine
# ============================================================

class PfVEngine:
    def __init__(self, registry: MetricRegistry, alpha: float = 0.6, beta: float = 0.4):
        self.registry = registry
        self.alpha = alpha  # L1 weight
        self.beta = beta    # L2 weight

    def observe(self, prompt: str, prompt_id: str, group: str,
                provider: BaseProvider, views: List[Tuple[str, float]],
                base_seed: int, max_tokens: int,
                delay: float = 0.0) -> IsomericObservation:

        obs = IsomericObservation(prompt=prompt, prompt_id=prompt_id, group=group)

        # Generate outputs across views
        outputs: List[ModelResponse] = []
        for k, (kind, val) in enumerate(views):
            seed = base_seed + k * 10007
            out = provider.generate(prompt=prompt, temperature=val,
                                     seed=seed, max_tokens=max_tokens)
            outputs.append(out)
            if delay > 0 and provider.name != "mock":
                time.sleep(delay)

        texts = [normalize_text(o.text) for o in outputs]
        obs.view_texts = texts
        obs.view_meta = [o.meta or {} for o in outputs]

        has_logits = all(o.logits is not None for o in outputs)
        logits_list = [o.logits for o in outputs] if has_logits else None

        # L1: per-metric polarization
        available = self.registry.get_available(has_logits=has_logits)
        obs.metrics_used = [m.name for m in available]

        for metric in available:
            if metric.kind == "logits":
                dists = metric.func(texts, logits_list=logits_list)
            else:
                dists = metric.func(texts)
            Pt = float(np.mean(dists)) if dists else 0.0
            obs.l1_by_metric[metric.name] = Pt
            obs.l1_pairwise[metric.name] = dists

        # L2: inter-metric disagreement
        if len(obs.l1_by_metric) >= 2:
            vals = np.array(list(obs.l1_by_metric.values()), dtype=np.float64)
            vmin, vmax = vals.min(), vals.max()
            rng = vmax - vmin
            if rng > 1e-12:
                normed = (vals - vmin) / rng
            else:
                normed = np.full_like(vals, 0.5)
            l2_dists = [abs(normed[i] - normed[j])
                        for i in range(len(normed)) for j in range(i+1, len(normed))]
            obs.l2_disagreement = float(np.mean(l2_dists)) if l2_dists else 0.0

        # L3: composite
        l1_mean = float(np.mean(list(obs.l1_by_metric.values()))) if obs.l1_by_metric else 0.0
        obs.l3_composite = self.alpha * l1_mean + self.beta * obs.l2_disagreement

        return obs

    def compute_dynamics(self, series: List[IsomericObservation]):
        for idx, obs in enumerate(series):
            for mn in obs.l1_by_metric:
                vals = [s.l1_by_metric.get(mn, 0.0) for s in series[:idx+1]]
                obs.dPt[mn] = vals[-1] - vals[-2] if len(vals) >= 2 else 0.0
                obs.ddPt[mn] = (vals[-1] - 2*vals[-2] + vals[-3]) if len(vals) >= 3 else 0.0
            cs = [s.l3_composite for s in series[:idx+1]]
            obs.dPt["composite"] = cs[-1] - cs[-2] if len(cs) >= 2 else 0.0
            obs.ddPt["composite"] = (cs[-1] - 2*cs[-2] + cs[-3]) if len(cs) >= 3 else 0.0


# ============================================================
# Statistical Tests
# ============================================================

def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) < 2 or len(b) < 2: return 0.0
    pooled = math.sqrt(((len(a)-1)*a.std(ddof=1)**2 + (len(b)-1)*b.std(ddof=1)**2) / (len(a)+len(b)-2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 1e-12 else 0.0

def run_pairwise_stats(group_vals: Dict[str, List[float]], metric_label: str) -> List[Dict]:
    results = []
    groups = sorted(group_vals.keys())
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            g1, g2 = groups[i], groups[j]
            a, b = np.array(group_vals[g1]), np.array(group_vals[g2])
            try: mw_u, mw_p = mannwhitneyu(a, b, alternative="two-sided")
            except: mw_u, mw_p = float("nan"), float("nan")
            try: ks_d, ks_p = ks_2samp(a, b)
            except: ks_d, ks_p = float("nan"), float("nan")
            d = cohens_d(a.tolist(), b.tolist())
            results.append({
                "metric": metric_label, "comparison": f"{g1} vs {g2}",
                "g1_mean": float(a.mean()), "g2_mean": float(b.mean()),
                "g1_n": len(a), "g2_n": len(b),
                "cohens_d": d, "mw_U": float(mw_u), "mw_p": float(mw_p),
                "ks_D": float(ks_d), "ks_p": float(ks_p),
                "sig_005": mw_p < 0.05,
            })
    return results

def run_permutation_test(group_vals: Dict[str, List[float]], n_perms: int = 5000) -> Dict:
    groups = sorted(group_vals.keys())
    vals, labs = [], []
    for g in groups:
        for v in group_vals[g]: vals.append(v); labs.append(g)
    vals, labs = np.array(vals), np.array(labs)

    # Observed
    obs_max_ks = 0.0
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            d, _ = ks_2samp(vals[labs == groups[i]], vals[labs == groups[j]])
            obs_max_ks = max(obs_max_ks, d)

    # Null
    null_ks = []
    for _ in range(n_perms):
        pl = np.random.permutation(labs)
        mx = 0.0
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                a, b = vals[pl == groups[i]], vals[pl == groups[j]]
                if len(a) > 0 and len(b) > 0:
                    d, _ = ks_2samp(a, b)
                    mx = max(mx, d)
        null_ks.append(mx)
    null_ks = np.array(null_ks)
    p = float((null_ks >= obs_max_ks).sum() + 1) / (n_perms + 1)

    return {
        "observed_max_ks": float(obs_max_ks),
        "null_mean": float(null_ks.mean()), "null_std": float(null_ks.std()),
        "null_95th": float(np.percentile(null_ks, 95)),
        "perm_p": p, "n_perms": n_perms,
        "exceeds_null": bool(obs_max_ks > np.percentile(null_ks, 95)),
    }


# ============================================================
# I/O
# ============================================================

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


# ============================================================
# Plotting
# ============================================================

COLORS = {"control_negative": "#2196F3", "benign": "#4CAF50",
          "stress": "#FF9800", "adversarial_sim": "#F44336"}
LABELS = {"control_negative": "ctrl", "benign": "benign",
          "stress": "stress", "adversarial_sim": "adv"}

def plot_dashboard(group_obs, stat_results_l3, stat_results_l2, perm_l3, perm_l2,
                   outdir, provider_name, model_name):
    if not HAS_PLT:
        print("[WARN] matplotlib not available"); return

    plt.rcParams.update({"font.size": 9, "figure.facecolor": "white",
                         "axes.grid": True, "grid.alpha": 0.3})
    groups = ["control_negative", "benign", "stress", "adversarial_sim"]

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f"Isomeric Polarization — PfV v5 Empirical Validation\n"
                 f"{provider_name} / {model_name}", fontsize=14, fontweight="bold", y=0.99)
    gs = GridSpec(3, 4, hspace=0.4, wspace=0.35,
                  height_ratios=[1, 1, 0.8])

    # Row 1: L3 composite
    ax1 = fig.add_subplot(gs[0, 0:2])
    data_l3 = [[o.l3_composite for o in group_obs[g]] for g in groups]
    bp = ax1.boxplot(data_l3, labels=[LABELS[g] for g in groups],
                     patch_artist=True, widths=0.6)
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(COLORS[g]); patch.set_alpha(0.7)
    for i, g in enumerate(groups):
        m = np.mean([o.l3_composite for o in group_obs[g]])
        ax1.annotate(f"{m:.3f}", xy=(i+1, m), fontsize=8, ha="center",
                     va="bottom", fontweight="bold")
    ax1.set_ylabel("L3 Composite"); ax1.set_title("L3: Composite Polarization (0.6·L1 + 0.4·L2)")

    # L2 disagreement
    ax2 = fig.add_subplot(gs[0, 2:4])
    data_l2 = [[o.l2_disagreement for o in group_obs[g]] for g in groups]
    bp2 = ax2.boxplot(data_l2, labels=[LABELS[g] for g in groups],
                      patch_artist=True, widths=0.6)
    for patch, g in zip(bp2["boxes"], groups):
        patch.set_facecolor(COLORS[g]); patch.set_alpha(0.7)
    ax2.set_ylabel("L2 Disagreement"); ax2.set_title("L2: Inter-Metric Disagreement")

    # Row 2: Strip plot + density
    ax3 = fig.add_subplot(gs[1, 0:2])
    for i, g in enumerate(groups):
        vals = [o.l3_composite for o in group_obs[g]]
        jitter = np.random.normal(0, 0.05, size=len(vals))
        ax3.scatter([i]*len(vals) + jitter, vals, c=COLORS[g], alpha=0.5,
                    s=35, edgecolors="white", linewidth=0.5)
        ax3.hlines(np.mean(vals), i-0.25, i+0.25, colors="black", linewidth=2)
    ax3.set_xticks(range(4)); ax3.set_xticklabels([LABELS[g] for g in groups])
    ax3.set_ylabel("L3"); ax3.set_title("L3 Individual Samples")

    ax4 = fig.add_subplot(gs[1, 2:4])
    for g in groups:
        vals = [o.l3_composite for o in group_obs[g]]
        if len(vals) > 2:
            try:
                kde = sp_stats.gaussian_kde(vals)
                x = np.linspace(min(vals)-0.02, max(vals)+0.02, 200)
                ax4.plot(x, kde(x), label=LABELS[g], color=COLORS[g], linewidth=2)
                ax4.fill_between(x, kde(x), alpha=0.1, color=COLORS[g])
            except: pass
    ax4.set_xlabel("L3"); ax4.set_ylabel("Density")
    ax4.set_title("L3 Distributions"); ax4.legend(fontsize=9)

    # Row 3: Stats table + permutation
    ax5 = fig.add_subplot(gs[2, 0:3]); ax5.axis("off")
    headers = ["Comparison", "Layer", "Δ mean", "Cohen's d", "MW p", "KS D", "Sig?"]
    rows = []
    for r in stat_results_l3:
        delta = r["g2_mean"] - r["g1_mean"]
        rows.append([r["comparison"], "L3", f"{delta:+.4f}", f"{r['cohens_d']:.3f}",
                     f"{r['mw_p']:.4f}", f"{r['ks_D']:.3f}",
                     "Y" if r["sig_005"] else "N"])
    for r in stat_results_l2:
        delta = r["g2_mean"] - r["g1_mean"]
        rows.append([r["comparison"], "L2", f"{delta:+.4f}", f"{r['cohens_d']:.3f}",
                     f"{r['mw_p']:.4f}", f"{r['ks_D']:.3f}",
                     "Y" if r["sig_005"] else "N"])
    tbl = ax5.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.3)
    for i, r in enumerate(rows):
        if r[-1] == "Y":
            for j in range(len(headers)): tbl[i+1, j].set_facecolor("#E8F5E9")
    ax5.set_title("Pairwise Statistical Tests", fontsize=11, fontweight="bold", pad=15)

    ax6 = fig.add_subplot(gs[2, 3]); ax6.axis("off")
    txt = (f"Permutation Tests\n{'─'*25}\n"
           f"L3 max KS D: {perm_l3['observed_max_ks']:.4f}\n"
           f"L3 null 95th: {perm_l3['null_95th']:.4f}\n"
           f"L3 perm p:    {perm_l3['perm_p']:.4f}\n"
           f"L3 exceeds:   {'YES' if perm_l3['exceeds_null'] else 'NO'}\n\n"
           f"L2 max KS D: {perm_l2['observed_max_ks']:.4f}\n"
           f"L2 null 95th: {perm_l2['null_95th']:.4f}\n"
           f"L2 perm p:    {perm_l2['perm_p']:.4f}\n"
           f"L2 exceeds:   {'YES' if perm_l2['exceeds_null'] else 'NO'}")
    ax6.text(0.05, 0.5, txt, transform=ax6.transAxes, fontsize=9, fontfamily="monospace",
             verticalalignment="center", bbox=dict(boxstyle="round,pad=0.5",
             facecolor="#F5F5F5", edgecolor="#CCC"))

    outpath = os.path.join(outdir, "pfv_v5_dashboard.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Dashboard: {outpath}")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="PfV v5 — Empirical Validation")
    ap.add_argument("--provider", choices=["mock", "openai", "anthropic"], default="mock")
    ap.add_argument("--model", type=str, default="gpt-4.1-mini")
    ap.add_argument("--api_key", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="out_pfv_v5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--views", type=int, default=5)
    ap.add_argument("--tmin", type=float, default=0.3)
    ap.add_argument("--tmax", type=float, default=1.2)
    ap.add_argument("--groups", type=str, default="control_negative,benign,stress,adversarial_sim")
    ap.add_argument("--max_prompts", type=int, default=0, help="0=all")
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--permutations", type=int, default=5000)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    set_seeds(args.seed)
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    mkdirp(args.outdir)

    # Provider
    if args.provider == "mock": provider = MockProvider()
    elif args.provider == "openai": provider = OpenAIProvider(args.model, args.api_key)
    elif args.provider == "anthropic": provider = AnthropicProvider(args.model, args.api_key)
    else: raise ValueError(f"Unknown: {args.provider}")

    views = [("temp", float(t)) for t in np.linspace(args.tmin, args.tmax, args.views)]
    registry = MetricRegistry()
    engine = PfVEngine(registry)

    total_prompts = sum(len(PROMPTS.get(g, [])[:args.max_prompts or 999]) for g in groups)
    total_calls = total_prompts * args.views

    print("=" * 65)
    print("  PfV v5 — Isomeric Polarization Empirical Validation")
    print("=" * 65)
    print(f"  Provider:  {args.provider} / {args.model}")
    print(f"  Views:     {args.views} (T={args.tmin:.2f}..{args.tmax:.2f})")
    print(f"  Groups:    {groups}")
    print(f"  Prompts:   {total_prompts} total, {total_calls} API calls")
    if args.provider != "mock":
        print(f"  Est. cost: ~${total_calls * 0.015:.2f} USD")
    print("=" * 65)

    # Run
    group_obs: Dict[str, List[IsomericObservation]] = {}
    all_records = []
    tok_in, tok_out = 0, 0

    for group in groups:
        prompts = PROMPTS.get(group, [])
        if args.max_prompts > 0: prompts = prompts[:args.max_prompts]
        print(f"\n>> {group} ({len(prompts)} prompts)")

        observations = []
        for idx, prompt in enumerate(prompts):
            pid = f"{group}_{idx}_{stable_hash(prompt)}"
            bseed = args.seed + idx * 13337 + hash(group) % 10000
            obs = engine.observe(prompt, pid, group, provider, views,
                                  bseed, args.max_tokens, delay=args.delay)
            observations.append(obs)

            # Track tokens
            for m in obs.view_meta:
                tok_in += m.get("tokens_in", 0)
                tok_out += m.get("tokens_out", 0)

            print(f"  [{idx+1}/{len(prompts)}] L3={obs.l3_composite:.4f} "
                  f"L2={obs.l2_disagreement:.4f} "
                  f"L1={{{', '.join(f'{k}={v:.3f}' for k,v in obs.l1_by_metric.items())}}}")

        engine.compute_dynamics(observations)
        group_obs[group] = observations

        l3_vals = [o.l3_composite for o in observations]
        print(f"  -> L3: {np.mean(l3_vals):.4f} ± {np.std(l3_vals):.4f}")

    # Collect records
    for g, obs_list in group_obs.items():
        for o in obs_list:
            all_records.append(o.to_dict())

    # Stats
    print("\n" + "=" * 65 + "\n  Statistical Analysis\n" + "=" * 65)

    l3_vals = {g: [o.l3_composite for o in group_obs[g]] for g in groups}
    l2_vals = {g: [o.l2_disagreement for o in group_obs[g]] for g in groups}

    stats_l3 = run_pairwise_stats(l3_vals, "L3")
    stats_l2 = run_pairwise_stats(l2_vals, "L2")

    for r in stats_l3:
        s = "Y" if r["sig_005"] else "N"
        print(f"  L3 {r['comparison']:>35s}  d={r['cohens_d']:+.3f}  MW p={r['mw_p']:.4f}  KS D={r['ks_D']:.3f}  [{s}]")
    for r in stats_l2:
        s = "Y" if r["sig_005"] else "N"
        print(f"  L2 {r['comparison']:>35s}  d={r['cohens_d']:+.3f}  MW p={r['mw_p']:.4f}  KS D={r['ks_D']:.3f}  [{s}]")

    print(f"\n  Permutation tests ({args.permutations} perms)...")
    perm_l3 = run_permutation_test(l3_vals, n_perms=args.permutations)
    perm_l2 = run_permutation_test(l2_vals, n_perms=args.permutations)
    print(f"  L3: obs={perm_l3['observed_max_ks']:.4f} null95={perm_l3['null_95th']:.4f} p={perm_l3['perm_p']:.4f} {'EXCEEDS' if perm_l3['exceeds_null'] else 'within null'}")
    print(f"  L2: obs={perm_l2['observed_max_ks']:.4f} null95={perm_l2['null_95th']:.4f} p={perm_l2['perm_p']:.4f} {'EXCEEDS' if perm_l2['exceeds_null'] else 'within null'}")

    # Monotonicity
    l3_means = {g: np.mean(l3_vals[g]) for g in groups}
    mono = (l3_means["control_negative"] < l3_means["benign"] <
            l3_means["stress"] < l3_means["adversarial_sim"])
    print(f"\n  Monotonicity (ctrl < benign < stress < adv): {'YES' if mono else 'NO'}")
    print(f"  " + " < ".join(f"{LABELS[g]}={l3_means[g]:.4f}" for g in groups))

    # Verdict
    any_l3_sig = any(r["sig_005"] for r in stats_l3)
    any_l2_sig = any(r["sig_005"] for r in stats_l2)
    print(f"\n  VERDICT:")
    if any_l3_sig or any_l2_sig:
        print(f"  → SIGNAL DETECTED: PfV separates at least some regimes significantly.")
        if args.provider != "mock":
            print(f"  → This is genuine content-driven divergence (no profile injection).")
    else:
        if args.provider == "mock":
            print(f"  → No significant separation with mock (expected — need real API data).")
        else:
            print(f"  → No significant separation detected. Consider more prompts or wider T range.")

    # Save
    print("\n" + "=" * 65 + "\n  Saving\n" + "=" * 65)
    write_jsonl(os.path.join(args.outdir, "runs_v5.jsonl"), all_records)
    write_json(os.path.join(args.outdir, "stats_v5.json"), {
        "l3_pairwise": stats_l3, "l2_pairwise": stats_l2,
        "permutation_l3": perm_l3, "permutation_l2": perm_l2,
        "monotonicity": mono,
        "l3_means": {g: float(v) for g, v in l3_means.items()},
    })
    write_json(os.path.join(args.outdir, "meta_v5.json"), {
        "ts": now_iso(), "provider": args.provider, "model": args.model,
        "seed": args.seed, "views": args.views, "tmin": args.tmin, "tmax": args.tmax,
        "max_tokens": args.max_tokens, "groups": groups,
        "total_prompts": total_prompts, "total_calls": total_calls,
        "tokens_in": tok_in, "tokens_out": tok_out,
    })
    print(f"  [OK] {args.outdir}/runs_v5.jsonl")
    print(f"  [OK] {args.outdir}/stats_v5.json")
    print(f"  [OK] {args.outdir}/meta_v5.json")
    if tok_in + tok_out > 0:
        print(f"  Tokens: {tok_in:,} in / {tok_out:,} out")

    if args.plots:
        plot_dashboard(group_obs, stats_l3, stats_l2, perm_l3, perm_l2,
                       args.outdir, args.provider, args.model)

    print("\n" + "=" * 65 + "\n  Done.\n" + "=" * 65)


if __name__ == "__main__":
    main()
