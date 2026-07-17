"""nunchi-eval — put p-values on snapshot diffs.

Snapshot tools tell you *what* changed between agent runs; this package
tells you whether that change exceeds the model's own noise floor, with a
permutation p-value. First adapter: EvalView model-check snapshots.

Built on nunchi-drift (the decision-stability metrics library).
"""

from .battery import (
    VERDICT_INSUFFICIENT,
    VERDICT_REGRESSION,
    VERDICT_WITHIN_NOISE,
    BatteryReport,
    PromptRuns,
    PromptVerdict,
    compare_batteries,
    judge_prompt,
)
from .traj import (
    ABSENT,
    TRAJ_INSUFFICIENT,
    TRAJ_REGRESSION,
    TRAJ_WITHIN_NOISE,
    StepDetail,
    TrajVerdict,
    compare_trajectories,
    mean_step_disagreement,
    per_step_disagreement,
    per_step_flip_rates,
    permutation_test_trajectories,
)
from .adapters.evalview import (
    EvalViewSnapshot,
    SnapshotFormatError,
    SuiteMismatchError,
    list_snapshots,
    load_reference_and_latest,
    load_snapshot,
    load_trace_runs,
)

__version__ = "0.1.0"

__all__ = [
    "PromptRuns", "PromptVerdict", "BatteryReport",
    "judge_prompt", "compare_batteries",
    "VERDICT_REGRESSION", "VERDICT_WITHIN_NOISE", "VERDICT_INSUFFICIENT",
    "EvalViewSnapshot", "load_snapshot", "list_snapshots",
    "load_reference_and_latest", "load_trace_runs",
    "SnapshotFormatError", "SuiteMismatchError",
    "ABSENT", "StepDetail", "TrajVerdict",
    "per_step_flip_rates", "per_step_disagreement",
    "mean_step_disagreement", "permutation_test_trajectories",
    "compare_trajectories",
    "TRAJ_REGRESSION", "TRAJ_WITHIN_NOISE", "TRAJ_INSUFFICIENT",
    "__version__",
]
