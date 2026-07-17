"""Adapter for EvalView model-check snapshots.

Reads ``.evalview/model_snapshots/<model>/*.json`` files (EvalView >= 0.8,
snapshot ``schema_version`` 1) and converts each per-prompt
``per_run_passed`` list into a :class:`nunchi_eval.battery.PromptRuns`.

Only reads files; never writes into ``.evalview/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..battery import PromptRuns

SUPPORTED_SCHEMA_VERSION = 1
REFERENCE_FILENAME = "reference.json"


class SnapshotFormatError(ValueError):
    """Snapshot file exists but does not match the expected schema."""


class SuiteMismatchError(ValueError):
    """Two snapshots come from different canary suites (suite_hash differs).

    Mirrors EvalView's own rule: comparisons across different suites are
    meaningless, so refuse loudly instead of producing plausible numbers.
    """


@dataclass
class EvalViewSnapshot:
    """One parsed model-check snapshot."""

    path: Path
    model_id: str
    suite_name: str
    suite_hash: str
    snapshot_at: str
    temperature: float
    runs_per_prompt: int
    is_reference: bool
    prompts: List[PromptRuns] = field(default_factory=list)

    def label(self) -> str:
        kind = "reference" if self.is_reference else "snapshot"
        return f"{kind} {self.snapshot_at} ({self.path.name})"


def load_snapshot(path: Path) -> EvalViewSnapshot:
    """Parse one snapshot JSON file, validating schema version and shape."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise SnapshotFormatError(f"{path}: not valid JSON ({e})") from e

    meta = raw.get("metadata")
    results = raw.get("results")
    if not isinstance(meta, dict) or not isinstance(results, list):
        raise SnapshotFormatError(
            f"{path}: expected top-level 'metadata' and 'results'")

    version = meta.get("schema_version")
    if version != SUPPORTED_SCHEMA_VERSION:
        raise SnapshotFormatError(
            f"{path}: snapshot schema_version {version} not supported "
            f"(expected {SUPPORTED_SCHEMA_VERSION})")

    prompts = []
    for i, r in enumerate(results):
        per_run = r.get("per_run_passed")
        if not isinstance(per_run, list):
            raise SnapshotFormatError(
                f"{path}: results[{i}] missing per_run_passed "
                "(is this snapshot from EvalView >= 0.8?)")
        prompts.append(PromptRuns(
            prompt_id=str(r.get("prompt_id", f"prompt-{i}")),
            category=str(r.get("category", "")),
            runs=tuple(bool(x) for x in per_run),
        ))

    return EvalViewSnapshot(
        path=path,
        model_id=str(meta.get("model_id", "unknown")),
        suite_name=str(meta.get("suite_name", "unknown")),
        suite_hash=str(meta.get("suite_hash", "")),
        snapshot_at=str(meta.get("snapshot_at", "")),
        temperature=float(meta.get("temperature", 0.0)),
        runs_per_prompt=int(meta.get("runs_per_prompt", 0)),
        is_reference=bool(meta.get("is_reference", False)),
        prompts=prompts,
    )


def list_snapshots(snapshot_dir: Path) -> List[Path]:
    """Timestamped snapshot files in a model dir, oldest first (no reference)."""
    snapshot_dir = Path(snapshot_dir)
    return sorted(p for p in snapshot_dir.glob("*.json")
                  if p.name != REFERENCE_FILENAME)


def load_reference_and_latest(
        snapshot_dir: Path) -> Tuple[EvalViewSnapshot, EvalViewSnapshot]:
    """Default comparison pair: pinned reference vs. most recent snapshot.

    Falls back to (oldest, newest) when no reference.json is pinned.
    Raises SuiteMismatchError when the two sides' suite_hash differ.
    """
    snapshot_dir = Path(snapshot_dir)
    timestamped = list_snapshots(snapshot_dir)
    ref_path = snapshot_dir / REFERENCE_FILENAME

    if ref_path.exists():
        if not timestamped:
            raise FileNotFoundError(
                f"{snapshot_dir}: reference.json exists but no timestamped "
                "snapshots to compare against")
        a, b = load_snapshot(ref_path), load_snapshot(timestamped[-1])
    elif len(timestamped) >= 2:
        a, b = load_snapshot(timestamped[0]), load_snapshot(timestamped[-1])
    else:
        raise FileNotFoundError(
            f"{snapshot_dir}: need reference.json + 1 snapshot, or >= 2 "
            f"snapshots (found {len(timestamped)})")

    if a.suite_hash != b.suite_hash:
        raise SuiteMismatchError(
            f"suite_hash mismatch: {a.path.name} has {a.suite_hash!r}, "
            f"{b.path.name} has {b.suite_hash!r} — the canary suite changed; "
            "re-pin the reference before comparing")
    return a, b
