"""Trajectory statistics: step-level stability for tool-call sequences.

A *trajectory run* is the ordered list of tool names an agent called in one
execution. Treating the whole sequence as a single category explodes
cardinality (every run looks unique); instead each step position is treated
as one categorical decision — "what did the agent do at step i" — and the
nunchi-drift math applies per position.

Alignment rule (v0.1, decision E1): **position-based with absent-padding**.
Runs shorter than the longest run contribute the sentinel ``ABSENT`` at the
missing positions. A trajectory that got shorter or longer therefore *is* a
behavioral change and shows up in the statistics, rather than being hidden
by a cleverer alignment. Tool-name matching and edit-distance alignment are
explicitly out of scope for v0.1 (multi-run alignment is where they stop
being well-defined).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import random

from nunchi_drift import disagreement, flip_rate, to_distribution

ABSENT = "<absent>"

TRAJ_REGRESSION = "REGRESSION"
TRAJ_WITHIN_NOISE = "WITHIN_NOISE"
TRAJ_INSUFFICIENT = "INSUFFICIENT_DATA"

TrajRun = Tuple[str, ...]


def _pad(runs: Sequence[Sequence[str]], length: int) -> List[TrajRun]:
    return [tuple(r) + (ABSENT,) * (length - len(r)) for r in runs]


def _max_len(runs: Sequence[Sequence[str]]) -> int:
    return max((len(r) for r in runs), default=0)


def per_step_flip_rates(runs: Sequence[Sequence[str]],
                        length: Optional[int] = None) -> List[float]:
    """Flip rate of the decision at each step position across runs.

    ``length`` pads to a fixed number of positions (used when comparing two
    groups so both sides share a common axis); defaults to the longest run.
    """
    if len(runs) < 2:
        return []
    L = length if length is not None else _max_len(runs)
    padded = _pad(runs, L)
    return [flip_rate([r[i] for r in padded]) for i in range(L)]


def per_step_disagreement(runs_a: Sequence[Sequence[str]],
                          runs_b: Sequence[Sequence[str]],
                          length: Optional[int] = None) -> List[float]:
    """P(a run of A and a run of B differ at step i), per position."""
    if not runs_a or not runs_b:
        return []
    L = length if length is not None else max(_max_len(runs_a), _max_len(runs_b))
    pa, pb = _pad(runs_a, L), _pad(runs_b, L)
    return [disagreement([r[i] for r in pa], [r[i] for r in pb])
            for i in range(L)]


def mean_step_disagreement(runs_a: Sequence[Sequence[str]],
                           runs_b: Sequence[Sequence[str]]) -> float:
    steps = per_step_disagreement(runs_a, runs_b)
    return sum(steps) / len(steps) if steps else 0.0


def permutation_test_trajectories(runs_a: Sequence[Sequence[str]],
                                  runs_b: Sequence[Sequence[str]],
                                  n_perm: int = 1000,
                                  seed: int = 0) -> float:
    """One-sided permutation p-value that two trajectory populations differ.

    Statistic: mean per-step cross disagreement on the common padded axis.
    Runs are exchanged between groups whole (a run is the exchangeable unit,
    not a step — steps within a run are correlated).
    """
    if not runs_a or not runs_b:
        return 1.0
    L = max(_max_len(runs_a), _max_len(runs_b))
    pool = _pad(runs_a, L) + _pad(runs_b, L)
    na = len(runs_a)

    def stat(group_a: List[TrajRun], group_b: List[TrajRun]) -> float:
        total = 0.0
        for i in range(L):
            total += disagreement([r[i] for r in group_a],
                                  [r[i] for r in group_b])
        return total / L if L else 0.0

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
class StepDetail:
    """Per-position diagnostics for the report table."""

    index: int                      # 1-based step number
    majority_a: Optional[Tuple[str, float]]
    majority_b: Optional[Tuple[str, float]]
    flip_a: float
    flip_b: float
    cross: float

    @property
    def floor(self) -> float:
        return max(self.flip_a, self.flip_b)

    @property
    def hotspot(self) -> bool:
        """Does this step diverge beyond its own noise floor?"""
        return self.cross > self.floor + 1e-12


@dataclass
class TrajVerdict:
    """Statistical comparison of two trajectory populations."""

    n_a: int
    n_b: int
    steps: List[StepDetail] = field(default_factory=list)
    mean_cross: float = 0.0
    floor: float = 0.0
    p_value: Optional[float] = None
    verdict: str = TRAJ_INSUFFICIENT

    @property
    def signal(self) -> bool:
        return self.mean_cross > self.floor + 1e-12

    @property
    def hotspots(self) -> List[StepDetail]:
        return [s for s in self.steps if s.hotspot]

    def summary(self) -> str:
        if self.verdict == TRAJ_INSUFFICIENT:
            return (f"trajectories: n_a={self.n_a} n_b={self.n_b} "
                    "— need >= 2 runs per side")
        rel = ">" if self.signal else "<="
        hot = (f", hotspot steps: {[s.index for s in self.hotspots]}"
               if self.hotspots else "")
        return (f"trajectories: mean cross={self.mean_cross:.3f} {rel} "
                f"floor={self.floor:.3f}, p={self.p_value:.3f} "
                f"-> {self.verdict}{hot}")


def compare_trajectories(runs_a: Sequence[Sequence[str]],
                         runs_b: Sequence[Sequence[str]],
                         alpha: float = 0.05, n_perm: int = 1000,
                         seed: int = 0) -> TrajVerdict:
    """Judge whether two trajectory populations behave differently.

    Same dual gate as the battery: mean cross disagreement must exceed the
    noise floor (the noisier side's mean per-step flip rate) AND the
    permutation p-value must be <= alpha.

    Small-sample honesty: with ~3 or fewer runs per side the permutation
    test cannot reach p <= 0.05 (too few distinct group assignments), so
    REGRESSION cannot fire even on a total behaviour change. Collect more
    runs rather than raising alpha.
    """
    if len(runs_a) < 2 or len(runs_b) < 2:
        return TrajVerdict(n_a=len(runs_a), n_b=len(runs_b))

    L = max(_max_len(runs_a), _max_len(runs_b))
    flips_a = per_step_flip_rates(runs_a, length=L)
    flips_b = per_step_flip_rates(runs_b, length=L)
    crosses = per_step_disagreement(runs_a, runs_b, length=L)
    pa, pb = _pad(runs_a, L), _pad(runs_b, L)

    steps = []
    for i in range(L):
        dist_a = to_distribution([r[i] for r in pa])
        dist_b = to_distribution([r[i] for r in pb])
        maj_a = max(dist_a.items(), key=lambda kv: kv[1]) if dist_a else None
        maj_b = max(dist_b.items(), key=lambda kv: kv[1]) if dist_b else None
        steps.append(StepDetail(index=i + 1, majority_a=maj_a, majority_b=maj_b,
                                flip_a=flips_a[i], flip_b=flips_b[i],
                                cross=crosses[i]))

    mean_cross = sum(crosses) / L if L else 0.0
    floor = max(sum(flips_a) / L if L else 0.0,
                sum(flips_b) / L if L else 0.0)
    p = permutation_test_trajectories(runs_a, runs_b, n_perm=n_perm, seed=seed)
    is_regression = mean_cross > floor + 1e-12 and p <= alpha
    return TrajVerdict(
        n_a=len(runs_a), n_b=len(runs_b), steps=steps,
        mean_cross=mean_cross, floor=floor, p_value=p,
        verdict=TRAJ_REGRESSION if is_regression else TRAJ_WITHIN_NOISE,
    )
