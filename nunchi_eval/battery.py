"""Statistical verdicts for prompt batteries.

A *battery* is a set of prompts, each run N times against a model, with a
boolean outcome per run. EvalView's ``model-check`` produces exactly this
(``per_run_passed``), but the math is tool-agnostic: any List[bool] per
prompt works.

The verdict logic is the noise-floor gate from nunchi-drift, applied
per prompt:

- each side's own flip rate is its noise floor — how much this prompt
  wobbles *without* any change;
- cross disagreement between snapshot A and snapshot B only counts as a
  regression signal when it exceeds both floors **and** a permutation test
  puts it below alpha.

Fixed-threshold classifiers ("2 flips = MEDIUM") alarm at the same rate on
quiet and noisy prompts alike; this module replaces that with a per-prompt
statistical judgement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from nunchi_drift import (
    bootstrap_flip_ci,
    disagreement,
    flip_rate,
    permutation_test_disagreement,
)

VERDICT_REGRESSION = "REGRESSION"
VERDICT_WITHIN_NOISE = "WITHIN_NOISE"
VERDICT_INSUFFICIENT = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class PromptRuns:
    """One prompt's repeated outcomes in one snapshot."""

    prompt_id: str
    runs: Tuple[bool, ...]
    category: str = ""

    def __post_init__(self):
        object.__setattr__(self, "runs", tuple(bool(r) for r in self.runs))

    @property
    def n(self) -> int:
        return len(self.runs)

    @property
    def pass_rate(self) -> float:
        return sum(self.runs) / self.n if self.runs else 0.0

    @property
    def flip_rate(self) -> float:
        return flip_rate(self.runs)

    @property
    def ci(self) -> Tuple[float, float]:
        return bootstrap_flip_ci(self.runs)


@dataclass
class PromptVerdict:
    """Statistical comparison of one prompt across two snapshots."""

    prompt_id: str
    category: str
    a: PromptRuns
    b: PromptRuns
    cross: float
    floor: float
    p_value: Optional[float]
    verdict: str

    @property
    def signal(self) -> bool:
        return self.cross > self.floor + 1e-12

    def summary(self) -> str:
        if self.verdict == VERDICT_INSUFFICIENT:
            return (f"{self.prompt_id}: n_a={self.a.n} n_b={self.b.n} "
                    f"— too few runs for a verdict")
        rel = ">" if self.signal else "<="
        return (f"{self.prompt_id}: cross={self.cross:.3f} {rel} "
                f"floor={self.floor:.3f}, p={self.p_value:.3f} -> {self.verdict}")


@dataclass
class BatteryReport:
    """All prompt verdicts for an A-vs-B snapshot comparison."""

    verdicts: List[PromptVerdict] = field(default_factory=list)
    unmatched_a: List[str] = field(default_factory=list)
    unmatched_b: List[str] = field(default_factory=list)
    alpha: float = 0.05

    @property
    def regressions(self) -> List[PromptVerdict]:
        return [v for v in self.verdicts if v.verdict == VERDICT_REGRESSION]

    @property
    def within_noise(self) -> List[PromptVerdict]:
        return [v for v in self.verdicts if v.verdict == VERDICT_WITHIN_NOISE]

    @property
    def insufficient(self) -> List[PromptVerdict]:
        return [v for v in self.verdicts if v.verdict == VERDICT_INSUFFICIENT]

    def summary(self) -> str:
        parts = [f"{len(self.regressions)} regression(s)",
                 f"{len(self.within_noise)} within noise"]
        if self.insufficient:
            parts.append(f"{len(self.insufficient)} insufficient")
        if self.unmatched_a or self.unmatched_b:
            parts.append(f"{len(self.unmatched_a) + len(self.unmatched_b)} unmatched")
        return f"{len(self.verdicts)} prompt(s): " + ", ".join(parts)


def judge_prompt(a: PromptRuns, b: PromptRuns,
                 alpha: float = 0.05, n_perm: int = 1000,
                 seed: int = 0) -> PromptVerdict:
    """Compare one prompt's runs across two snapshots.

    REGRESSION requires *both* gates: cross disagreement above the noise
    floor (max of the two flip rates) and permutation p <= alpha. Either
    gate alone over-alarms: cross > floor happens by chance on small n,
    and p-values go tiny for consistent-but-identical behaviour shifts
    already explained by the floor.
    """
    if a.n < 2 or b.n < 2:
        return PromptVerdict(prompt_id=a.prompt_id, category=a.category,
                             a=a, b=b, cross=0.0, floor=0.0,
                             p_value=None, verdict=VERDICT_INSUFFICIENT)

    cross = disagreement(a.runs, b.runs)
    floor = max(a.flip_rate, b.flip_rate)
    p = permutation_test_disagreement(a.runs, b.runs, n_perm=n_perm, seed=seed)
    is_regression = cross > floor + 1e-12 and p <= alpha
    return PromptVerdict(
        prompt_id=a.prompt_id, category=a.category, a=a, b=b,
        cross=cross, floor=floor, p_value=p,
        verdict=VERDICT_REGRESSION if is_regression else VERDICT_WITHIN_NOISE,
    )


def compare_batteries(battery_a: Sequence[PromptRuns],
                      battery_b: Sequence[PromptRuns],
                      alpha: float = 0.05, n_perm: int = 1000,
                      seed: int = 0) -> BatteryReport:
    """Judge every prompt present in both batteries; report the rest as unmatched."""
    index_a: Dict[str, PromptRuns] = {p.prompt_id: p for p in battery_a}
    index_b: Dict[str, PromptRuns] = {p.prompt_id: p for p in battery_b}

    report = BatteryReport(alpha=alpha)
    for pid in index_a:
        if pid in index_b:
            report.verdicts.append(
                judge_prompt(index_a[pid], index_b[pid],
                             alpha=alpha, n_perm=n_perm, seed=seed))
        else:
            report.unmatched_a.append(pid)
    report.unmatched_b = [pid for pid in index_b if pid not in index_a]
    return report
