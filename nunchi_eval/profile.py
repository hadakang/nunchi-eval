"""Tool-usage profile view: order-invariant trajectory statistics.

The position view (:mod:`nunchi_eval.traj`) asks "what did the agent do at
step i" — precise about *where* behaviour changed, but ordering noise
(parallel tool calls recorded in arbitrary order) inflates its noise floor
and can mask real changes at those positions.

This view drops the step axis entirely and asks, per tool: **"how many
times did this run call tool X?"** The per-run call count is a small
categorical value, so the same nunchi-drift math applies per tool. Ordering
cannot touch it; a tool that disappears (the "quietly stopped verifying"
regression) is maximally visible.

The two views complement each other — position says *where*, profile says
*what*. Run both (the CLI default) and read disagreements between them as
information: e.g. position=REGRESSION + profile=WITHIN_NOISE means the
agent calls the same tools in a different order.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from nunchi_drift import disagreement, flip_rate, to_distribution

PROFILE_REGRESSION = "REGRESSION"
PROFILE_WITHIN_NOISE = "WITHIN_NOISE"
PROFILE_INSUFFICIENT = "INSUFFICIENT_DATA"


def tool_counts(run: Sequence[str]) -> Dict[str, int]:
    """One run's tool-usage profile: tool name -> call count."""
    counts: Dict[str, int] = {}
    for tool in run:
        counts[tool] = counts.get(tool, 0) + 1
    return counts


def _tool_universe(*run_groups: Sequence[Sequence[str]]) -> List[str]:
    tools = set()
    for runs in run_groups:
        for run in runs:
            tools.update(run)
    return sorted(tools)


def _count_column(runs: Sequence[Sequence[str]], tool: str) -> List[int]:
    return [sum(1 for t in run if t == tool) for run in runs]


def per_tool_flip_rates(runs: Sequence[Sequence[str]],
                        tools: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Flip rate of each tool's per-run call count across runs."""
    if len(runs) < 2:
        return {}
    universe = list(tools) if tools is not None else _tool_universe(runs)
    return {t: flip_rate(_count_column(runs, t)) for t in universe}


def permutation_test_profiles(runs_a: Sequence[Sequence[str]],
                              runs_b: Sequence[Sequence[str]],
                              n_perm: int = 1000,
                              seed: int = 0) -> float:
    """One-sided permutation p-value that two tool-usage populations differ.

    Statistic: mean per-tool disagreement of call-count columns. Runs are
    the exchangeable unit (a run's counts move together).
    """
    if not runs_a or not runs_b:
        return 1.0
    tools = _tool_universe(runs_a, runs_b)
    if not tools:
        return 1.0
    pool = [tuple(r) for r in runs_a] + [tuple(r) for r in runs_b]
    na = len(runs_a)

    def stat(group_a, group_b) -> float:
        total = 0.0
        for t in tools:
            total += disagreement(_count_column(group_a, t),
                                  _count_column(group_b, t))
        return total / len(tools)

    observed = stat(pool[:na], pool[na:])
    rng = random.Random(seed)
    count_ge = 0
    indices = list(range(len(pool)))
    for _ in range(n_perm):
        rng.shuffle(indices)
        group_a = [pool[i] for i in indices[:na]]
        group_b = [pool[i] for i in indices[na:]]
        if stat(group_a, group_b) >= observed - 1e-12:
            count_ge += 1
    return (count_ge + 1) / (n_perm + 1)


@dataclass
class ToolDetail:
    """Per-tool diagnostics for the report table."""

    tool: str
    majority_a: Optional[Tuple[int, float]]   # (call count, share)
    majority_b: Optional[Tuple[int, float]]
    flip_a: float
    flip_b: float
    cross: float

    @property
    def floor(self) -> float:
        return max(self.flip_a, self.flip_b)

    @property
    def hotspot(self) -> bool:
        return self.cross > self.floor + 1e-12


@dataclass
class ProfileVerdict:
    """Statistical comparison of two tool-usage populations."""

    n_a: int
    n_b: int
    tools: List[ToolDetail] = field(default_factory=list)
    mean_cross: float = 0.0
    floor: float = 0.0
    p_value: Optional[float] = None
    verdict: str = PROFILE_INSUFFICIENT

    @property
    def signal(self) -> bool:
        return self.mean_cross > self.floor + 1e-12

    @property
    def hotspots(self) -> List[ToolDetail]:
        return [t for t in self.tools if t.hotspot]

    def summary(self) -> str:
        if self.verdict == PROFILE_INSUFFICIENT:
            return (f"tool profile: n_a={self.n_a} n_b={self.n_b} "
                    "— need >= 2 runs per side")
        rel = ">" if self.signal else "<="
        hot = (f", hotspot tools: {[t.tool for t in self.hotspots]}"
               if self.hotspots else "")
        return (f"tool profile: mean cross={self.mean_cross:.3f} {rel} "
                f"floor={self.floor:.3f}, p={self.p_value:.3f} "
                f"-> {self.verdict}{hot}")


def compare_profiles(runs_a: Sequence[Sequence[str]],
                     runs_b: Sequence[Sequence[str]],
                     alpha: float = 0.05, n_perm: int = 1000,
                     seed: int = 0) -> ProfileVerdict:
    """Judge whether two populations use tools differently (order-blind).

    Same dual gate as the other views: mean per-tool cross disagreement
    must exceed the noise floor AND the permutation p-value must be
    <= alpha. Small-sample note from :func:`nunchi_eval.traj.compare_trajectories`
    applies here too.
    """
    if len(runs_a) < 2 or len(runs_b) < 2:
        return ProfileVerdict(n_a=len(runs_a), n_b=len(runs_b))

    universe = _tool_universe(runs_a, runs_b)
    details = []
    flips_a, flips_b, crosses = [], [], []
    for t in universe:
        col_a, col_b = _count_column(runs_a, t), _count_column(runs_b, t)
        fa, fb = flip_rate(col_a), flip_rate(col_b)
        cross = disagreement(col_a, col_b)
        dist_a, dist_b = to_distribution(col_a), to_distribution(col_b)
        maj_a = max(dist_a.items(), key=lambda kv: kv[1]) if dist_a else None
        maj_b = max(dist_b.items(), key=lambda kv: kv[1]) if dist_b else None
        details.append(ToolDetail(tool=t, majority_a=maj_a, majority_b=maj_b,
                                  flip_a=fa, flip_b=fb, cross=cross))
        flips_a.append(fa)
        flips_b.append(fb)
        crosses.append(cross)

    n_tools = len(universe)
    mean_cross = sum(crosses) / n_tools if n_tools else 0.0
    floor = max(sum(flips_a) / n_tools if n_tools else 0.0,
                sum(flips_b) / n_tools if n_tools else 0.0)
    p = permutation_test_profiles(runs_a, runs_b, n_perm=n_perm, seed=seed)
    is_regression = mean_cross > floor + 1e-12 and p <= alpha
    return ProfileVerdict(
        n_a=len(runs_a), n_b=len(runs_b), tools=details,
        mean_cross=mean_cross, floor=floor, p_value=p,
        verdict=PROFILE_REGRESSION if is_regression else PROFILE_WITHIN_NOISE,
    )
