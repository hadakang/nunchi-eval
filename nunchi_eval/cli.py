"""nunchi-eval CLI: statistical verdicts over EvalView model-check snapshots.

Usage:
  nunchi-eval check .evalview/model_snapshots/<model>/
  nunchi-eval check --a reference.json --b latest.json
Exit code 1 when any prompt is judged REGRESSION (CI-friendly), else 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .battery import (
    VERDICT_INSUFFICIENT,
    VERDICT_REGRESSION,
    BatteryReport,
    compare_batteries,
)
from .traj import TRAJ_INSUFFICIENT, TrajVerdict, compare_trajectories
from .adapters.evalview import (
    EvalViewSnapshot,
    SnapshotFormatError,
    SuiteMismatchError,
    load_reference_and_latest,
    load_snapshot,
    load_trace_runs,
)


def _fmt_row(cols, widths):
    return "  ".join(str(c).ljust(w) for c, w in zip(cols, widths)).rstrip()


def render_report(a: EvalViewSnapshot, b: EvalViewSnapshot,
                  report: BatteryReport) -> str:
    lines = [
        f"suite: {a.suite_name} (hash {a.suite_hash[:12] or 'n/a'})  "
        f"model: {a.model_id}  temperature: {a.temperature}",
        f"A = {a.label()}",
        f"B = {b.label()}",
        "",
    ]
    header = ["prompt", "category", "flip A [95% CI]", "flip B [95% CI]",
              "cross", "floor", "p", "verdict"]
    rows = []
    for v in sorted(report.verdicts,
                    key=lambda v: (v.verdict != VERDICT_REGRESSION, v.prompt_id)):
        if v.verdict == VERDICT_INSUFFICIENT:
            rows.append([v.prompt_id, v.category, f"n={v.a.n}", f"n={v.b.n}",
                         "-", "-", "-", v.verdict])
            continue
        ca, cb = v.a.ci, v.b.ci
        rows.append([
            v.prompt_id, v.category,
            f"{v.a.flip_rate:.2f} [{ca[0]:.2f},{ca[1]:.2f}]",
            f"{v.b.flip_rate:.2f} [{cb[0]:.2f},{cb[1]:.2f}]",
            f"{v.cross:.2f}", f"{v.floor:.2f}", f"{v.p_value:.3f}",
            v.verdict,
        ])
    widths = [max(len(str(r[i])) for r in [header] + rows)
              for i in range(len(header))]
    lines.append(_fmt_row(header, widths))
    lines.append(_fmt_row(["-" * w for w in widths], widths))
    lines.extend(_fmt_row(r, widths) for r in rows)
    lines.append("")
    lines.append(report.summary())
    for pid in report.unmatched_a:
        lines.append(f"note: {pid} only in A (dropped from suite?)")
    for pid in report.unmatched_b:
        lines.append(f"note: {pid} only in B (added to suite?)")
    return "\n".join(lines)


def report_as_json(a: EvalViewSnapshot, b: EvalViewSnapshot,
                   report: BatteryReport) -> str:
    return json.dumps({
        "suite": a.suite_name,
        "suite_hash": a.suite_hash,
        "model_id": a.model_id,
        "a": a.path.name,
        "b": b.path.name,
        "alpha": report.alpha,
        "summary": report.summary(),
        "prompts": [{
            "prompt_id": v.prompt_id,
            "category": v.category,
            "n_a": v.a.n, "n_b": v.b.n,
            "flip_a": v.a.flip_rate, "ci_a": list(v.a.ci),
            "flip_b": v.b.flip_rate, "ci_b": list(v.b.ci),
            "cross": v.cross, "floor": v.floor,
            "p_value": v.p_value, "verdict": v.verdict,
        } for v in report.verdicts],
        "unmatched_a": report.unmatched_a,
        "unmatched_b": report.unmatched_b,
    }, indent=2)


def render_traj_report(v: TrajVerdict) -> str:
    lines = [f"runs: A={v.n_a}  B={v.n_b}  steps analyzed: {len(v.steps)}", ""]
    header = ["step", "majority A", "majority B", "flip A", "flip B",
              "cross", "floor", ""]
    rows = []
    for s in v.steps:
        ma = f"{s.majority_a[0]} ({s.majority_a[1]:.0%})" if s.majority_a else "-"
        mb = f"{s.majority_b[0]} ({s.majority_b[1]:.0%})" if s.majority_b else "-"
        rows.append([s.index, ma, mb, f"{s.flip_a:.2f}", f"{s.flip_b:.2f}",
                     f"{s.cross:.2f}", f"{s.floor:.2f}",
                     "<- hotspot" if s.hotspot else ""])
    widths = [max(len(str(r[i])) for r in [header] + rows)
              for i in range(len(header))]
    lines.append(_fmt_row(header, widths))
    lines.append(_fmt_row(["-" * w for w in widths], widths))
    lines.extend(_fmt_row(r, widths) for r in rows)
    lines.append("")
    lines.append(v.summary())
    return "\n".join(lines)


def traj_report_as_json(v: TrajVerdict) -> str:
    return json.dumps({
        "n_a": v.n_a, "n_b": v.n_b,
        "mean_cross": v.mean_cross, "floor": v.floor,
        "p_value": v.p_value, "verdict": v.verdict,
        "steps": [{
            "step": s.index,
            "majority_a": list(s.majority_a) if s.majority_a else None,
            "majority_b": list(s.majority_b) if s.majority_b else None,
            "flip_a": s.flip_a, "flip_b": s.flip_b,
            "cross": s.cross, "floor": s.floor, "hotspot": s.hotspot,
        } for s in v.steps],
    }, indent=2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="nunchi-eval",
        description="Put p-values on snapshot diffs: statistical regression "
                    "verdicts for EvalView model-check batteries.")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser(
        "check", help="Judge reference-vs-latest (or --a vs --b) snapshots.")
    check.add_argument("snapshot_dir", nargs="?", type=Path,
                       help=".evalview/model_snapshots/<model>/ directory")
    check.add_argument("--a", type=Path, help="explicit snapshot A (baseline)")
    check.add_argument("--b", type=Path, help="explicit snapshot B (candidate)")
    check.add_argument("--alpha", type=float, default=0.05,
                       help="significance level for REGRESSION (default 0.05)")
    check.add_argument("--n-perm", type=int, default=1000,
                       help="permutation test iterations (default 1000)")
    check.add_argument("--json", action="store_true", dest="as_json",
                       help="machine-readable JSON output")

    traj = sub.add_parser(
        "traj", help="Compare two trajectory populations (tool-call "
                     "sequences) statistically.")
    traj.add_argument("--a", type=Path, required=True,
                      help="baseline runs: dir of ExecutionTrace *.json, "
                           "or one file with a list of traces")
    traj.add_argument("--b", type=Path, required=True,
                      help="candidate runs: same formats as --a")
    traj.add_argument("--alpha", type=float, default=0.05)
    traj.add_argument("--n-perm", type=int, default=1000)
    traj.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    if args.command == "traj":
        try:
            runs_a = load_trace_runs(args.a)
            runs_b = load_trace_runs(args.b)
        except (FileNotFoundError, SnapshotFormatError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        verdict = compare_trajectories(runs_a, runs_b, alpha=args.alpha,
                                       n_perm=args.n_perm)
        print((traj_report_as_json if args.as_json
               else render_traj_report)(verdict))
        return 1 if verdict.verdict == "REGRESSION" else 0

    try:
        if args.a and args.b:
            snap_a, snap_b = load_snapshot(args.a), load_snapshot(args.b)
            if snap_a.suite_hash != snap_b.suite_hash:
                raise SuiteMismatchError(
                    f"suite_hash mismatch between {args.a} and {args.b}")
        elif args.snapshot_dir:
            snap_a, snap_b = load_reference_and_latest(args.snapshot_dir)
        else:
            check.error("give a snapshot directory, or both --a and --b")
    except (FileNotFoundError, SnapshotFormatError, SuiteMismatchError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    report = compare_batteries(snap_a.prompts, snap_b.prompts,
                               alpha=args.alpha, n_perm=args.n_perm)
    out = (report_as_json if args.as_json else render_report)(
        snap_a, snap_b, report)
    print(out)
    return 1 if report.regressions else 0


if __name__ == "__main__":
    sys.exit(main())
