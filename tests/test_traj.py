"""Trajectory statistics: alignment (E1: position + absent-padding),
per-step metrics, permutation test, and the traj CLI."""

import json

import pytest

from nunchi_eval import (
    ABSENT,
    TRAJ_INSUFFICIENT,
    TRAJ_REGRESSION,
    TRAJ_WITHIN_NOISE,
    compare_trajectories,
    load_trace_runs,
    per_step_flip_rates,
    permutation_test_trajectories,
)
from nunchi_eval.cli import main


SEARCH_FETCH_ANSWER = ("search", "fetch", "answer")


class TestPerStepFlipRates:
    def test_identical_runs_all_zero(self):
        runs = [SEARCH_FETCH_ANSWER] * 5
        assert per_step_flip_rates(runs) == [0.0, 0.0, 0.0]

    def test_divergent_middle_step(self):
        runs = [("search", "fetch", "answer"),
                ("search", "answer_directly", "answer")] * 3
        rates = per_step_flip_rates(runs)
        assert rates[0] == 0.0
        assert rates[1] > 0.4
        assert rates[2] == 0.0

    def test_length_difference_counts_as_flip(self):
        # E1: a run that stops early differs at the missing positions.
        runs = [("search", "fetch", "answer"), ("search", "fetch")]
        rates = per_step_flip_rates(runs)
        assert rates[2] == 1.0  # answer vs <absent>

    def test_fewer_than_two_runs(self):
        assert per_step_flip_rates([SEARCH_FETCH_ANSWER]) == []


class TestCompareTrajectories:
    def test_identical_populations_within_noise(self):
        a = [SEARCH_FETCH_ANSWER] * 10
        b = [SEARCH_FETCH_ANSWER] * 10
        v = compare_trajectories(a, b)
        assert v.verdict == TRAJ_WITHIN_NOISE
        assert v.mean_cross == 0.0
        assert v.hotspots == []

    def test_step_swap_regression_with_hotspot(self):
        # Stable on both sides, but step 2 changed tool: the clean regression.
        a = [("search", "fetch", "answer")] * 10
        b = [("search", "cached_lookup", "answer")] * 10
        v = compare_trajectories(a, b)
        assert v.verdict == TRAJ_REGRESSION
        assert v.p_value <= 0.05
        assert [s.index for s in v.hotspots] == [2]

    def test_truncated_trajectories_regression(self):
        # Agent stopped calling its verification tool — the EvalView blog's
        # "quietly started lying" scenario. Length change must be a signal.
        a = [("search", "fetch", "verify", "answer")] * 10
        b = [("search", "fetch", "answer")] * 10
        v = compare_trajectories(a, b)
        assert v.verdict == TRAJ_REGRESSION
        # steps 3 and 4 both shift (verify->answer, answer-><absent>)
        assert {s.index for s in v.hotspots} == {3, 4}

    def test_noisy_same_distribution_is_noise(self):
        # Both sides alternate the same two paths at the same rate:
        # cross is high but so is the floor -> not a regression claim.
        path1 = ("search", "fetch", "answer")
        path2 = ("search", "answer_directly", "answer")
        a = [path1, path2] * 5
        b = [path2, path1] * 5
        v = compare_trajectories(a, b)
        assert v.verdict == TRAJ_WITHIN_NOISE
        assert v.floor >= v.mean_cross - 1e-9

    def test_insufficient_runs(self):
        v = compare_trajectories([SEARCH_FETCH_ANSWER], [SEARCH_FETCH_ANSWER] * 5)
        assert v.verdict == TRAJ_INSUFFICIENT
        assert v.p_value is None

    def test_small_samples_cannot_fire(self):
        # n=3 per side: C(6,3)=20 group assignments — the permutation test's
        # smallest reachable p is ~0.1, so even a total change stays
        # WITHIN_NOISE. Documented small-sample behaviour, not a bug.
        v = compare_trajectories([("a", "b")] * 3, [("x", "y")] * 3)
        assert v.verdict == TRAJ_WITHIN_NOISE
        assert v.p_value > 0.05

    def test_permutation_deterministic(self):
        a = [("x", "y"), ("x", "z")] * 3
        b = [("x", "z"), ("x", "y")] * 3
        p1 = permutation_test_trajectories(a, b, seed=0)
        p2 = permutation_test_trajectories(a, b, seed=0)
        assert p1 == p2


def write_traces(path, sequences):
    path.write_text(json.dumps([
        {"session_id": f"s{i}", "steps": [
            {"step_id": f"s{i}-{j}", "tool_name": tool}
            for j, tool in enumerate(seq)]}
        for i, seq in enumerate(sequences)]))
    return path


class TestTraceLoader:
    def test_single_file_with_list(self, tmp_path):
        f = write_traces(tmp_path / "runs.json",
                         [SEARCH_FETCH_ANSWER, ("search", "answer")])
        runs = load_trace_runs(f)
        assert runs == [SEARCH_FETCH_ANSWER, ("search", "answer")]

    def test_directory_of_files(self, tmp_path):
        write_traces(tmp_path / "01.json", [SEARCH_FETCH_ANSWER])
        write_traces(tmp_path / "02.json", [("search", "answer")])
        runs = load_trace_runs(tmp_path)
        assert len(runs) == 2

    def test_rejects_missing_tool_name(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text(json.dumps({"steps": [{"step_id": "x"}]}))
        with pytest.raises(Exception, match="tool_name"):
            load_trace_runs(f)


class TestTrajCli:
    def test_regression_exit_1_and_hotspot_marker(self, tmp_path, capsys):
        a = write_traces(tmp_path / "a.json",
                         [("search", "fetch", "verify", "answer")] * 10)
        b = write_traces(tmp_path / "b.json",
                         [("search", "fetch", "answer")] * 10)
        code = main(["traj", "--a", str(a), "--b", str(b)])
        out = capsys.readouterr().out
        assert code == 1
        assert "REGRESSION" in out
        assert "hotspot" in out

    def test_json_output(self, tmp_path, capsys):
        a = write_traces(tmp_path / "a.json", [SEARCH_FETCH_ANSWER] * 5)
        b = write_traces(tmp_path / "b.json", [SEARCH_FETCH_ANSWER] * 5)
        code = main(["traj", "--a", str(a), "--b", str(b), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert code == 0
        assert payload["verdict"] == "WITHIN_NOISE"
        assert len(payload["steps"]) == 3
