"""EvalView adapter: loading, validation, and pair selection against
synthetic snapshot files that mirror EvalView 0.8's on-disk schema."""

import json

import pytest

from nunchi_eval import (
    SnapshotFormatError,
    SuiteMismatchError,
    load_reference_and_latest,
    load_snapshot,
)
from nunchi_eval.cli import main


def make_snapshot(per_prompt, suite_hash="abc123", is_reference=False,
                  snapshot_at="2026-07-17T10:00:00Z", model="claude-test"):
    """Build a dict matching EvalView's ModelSnapshot.model_dump_json shape."""
    return {
        "metadata": {
            "schema_version": 1,
            "model_id": model,
            "provider": "anthropic",
            "snapshot_at": snapshot_at,
            "suite_name": "default-canary",
            "suite_version": "1",
            "suite_hash": suite_hash,
            "temperature": 0.0,
            "top_p": 1.0,
            "runs_per_prompt": 10,
            "is_reference": is_reference,
            "cost_total_usd": 0.01,
        },
        "results": [
            {
                "prompt_id": pid,
                "category": "tool_selection",
                "pass_rate": sum(bools) / len(bools),
                "n_runs": len(bools),
                "per_run_passed": bools,
            }
            for pid, bools in per_prompt.items()
        ],
    }


def write(path, data):
    path.write_text(json.dumps(data))
    return path


class TestLoadSnapshot:
    def test_roundtrip(self, tmp_path):
        f = write(tmp_path / "s.json",
                  make_snapshot({"p1": [True] * 9 + [False]}))
        snap = load_snapshot(f)
        assert snap.model_id == "claude-test"
        assert snap.suite_hash == "abc123"
        assert len(snap.prompts) == 1
        assert snap.prompts[0].runs == tuple([True] * 9 + [False])

    def test_rejects_wrong_schema_version(self, tmp_path):
        data = make_snapshot({"p1": [True]})
        data["metadata"]["schema_version"] = 2
        f = write(tmp_path / "s.json", data)
        with pytest.raises(SnapshotFormatError, match="schema_version"):
            load_snapshot(f)

    def test_rejects_missing_per_run_passed(self, tmp_path):
        data = make_snapshot({"p1": [True]})
        del data["results"][0]["per_run_passed"]
        f = write(tmp_path / "s.json", data)
        with pytest.raises(SnapshotFormatError, match="per_run_passed"):
            load_snapshot(f)

    def test_rejects_non_json(self, tmp_path):
        f = tmp_path / "s.json"
        f.write_text("not json{")
        with pytest.raises(SnapshotFormatError):
            load_snapshot(f)


class TestPairSelection:
    def test_reference_vs_latest(self, tmp_path):
        write(tmp_path / "reference.json",
              make_snapshot({"p1": [True] * 10}, is_reference=True))
        write(tmp_path / "2026-07-16T00-00-00.000000Z.json",
              make_snapshot({"p1": [True] * 10}))
        write(tmp_path / "2026-07-17T00-00-00.000000Z.json",
              make_snapshot({"p1": [False] * 10}))
        a, b = load_reference_and_latest(tmp_path)
        assert a.is_reference
        assert b.path.name.startswith("2026-07-17")

    def test_falls_back_to_oldest_vs_newest(self, tmp_path):
        write(tmp_path / "2026-07-16T00-00-00.000000Z.json",
              make_snapshot({"p1": [True] * 10}))
        write(tmp_path / "2026-07-17T00-00-00.000000Z.json",
              make_snapshot({"p1": [True] * 10}))
        a, b = load_reference_and_latest(tmp_path)
        assert a.path.name.startswith("2026-07-16")
        assert b.path.name.startswith("2026-07-17")

    def test_suite_hash_mismatch_refuses(self, tmp_path):
        write(tmp_path / "reference.json",
              make_snapshot({"p1": [True] * 10}, suite_hash="old",
                            is_reference=True))
        write(tmp_path / "2026-07-17T00-00-00.000000Z.json",
              make_snapshot({"p1": [True] * 10}, suite_hash="new"))
        with pytest.raises(SuiteMismatchError):
            load_reference_and_latest(tmp_path)

    def test_single_file_errors(self, tmp_path):
        write(tmp_path / "2026-07-17T00-00-00.000000Z.json",
              make_snapshot({"p1": [True] * 10}))
        with pytest.raises(FileNotFoundError):
            load_reference_and_latest(tmp_path)


class TestCli:
    def _dir_with(self, tmp_path, ref_runs, latest_runs):
        write(tmp_path / "reference.json",
              make_snapshot(ref_runs, is_reference=True))
        write(tmp_path / "2026-07-17T00-00-00.000000Z.json",
              make_snapshot(latest_runs))
        return tmp_path

    def test_exit_1_on_regression(self, tmp_path, capsys):
        d = self._dir_with(tmp_path,
                           {"p1": [True] * 10, "p2": [True, False] * 5},
                           {"p1": [False] * 10, "p2": [False, True] * 5})
        code = main(["check", str(d)])
        out = capsys.readouterr().out
        assert code == 1
        assert "REGRESSION" in out
        assert "WITHIN_NOISE" in out

    def test_exit_0_when_all_noise(self, tmp_path, capsys):
        d = self._dir_with(tmp_path,
                           {"p1": [True] * 10}, {"p1": [True] * 10})
        code = main(["check", str(d)])
        assert code == 0
        assert "0 regression(s)" in capsys.readouterr().out

    def test_json_output(self, tmp_path, capsys):
        d = self._dir_with(tmp_path,
                           {"p1": [True] * 10}, {"p1": [False] * 10})
        code = main(["check", str(d), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert code == 1
        assert payload["prompts"][0]["verdict"] == "REGRESSION"
        assert payload["prompts"][0]["p_value"] <= 0.05

    def test_error_exit_2_on_missing_dir(self, tmp_path, capsys):
        code = main(["check", str(tmp_path / "nope")])
        assert code == 2
        assert "error:" in capsys.readouterr().err
