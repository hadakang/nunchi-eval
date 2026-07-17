"""Tool-usage profile view (order-invariant) and parameter-key
discretization — including the complementarity contract between views."""

import json

import pytest

from nunchi_eval import (
    PROFILE_INSUFFICIENT,
    PROFILE_REGRESSION,
    PROFILE_WITHIN_NOISE,
    compare_profiles,
    compare_trajectories,
    load_trace_runs,
    per_tool_flip_rates,
    tool_counts,
)
from nunchi_eval.cli import main


class TestToolCounts:
    def test_counts_repeats(self):
        assert tool_counts(("search", "fetch", "search")) == {
            "search": 2, "fetch": 1}

    def test_per_tool_flip_rates(self):
        runs = [("search", "fetch"), ("search",)] * 3
        rates = per_tool_flip_rates(runs)
        assert rates["search"] == 0.0     # always exactly 1 call
        assert rates["fetch"] > 0.4       # 1 call vs 0 calls alternating


class TestCompareProfiles:
    def test_identical_within_noise(self):
        runs = [("search", "fetch", "answer")] * 10
        v = compare_profiles(runs, runs)
        assert v.verdict == PROFILE_WITHIN_NOISE
        assert v.mean_cross == 0.0

    def test_tool_disappearance_hotspot(self):
        a = [("search", "fetch", "verify", "answer")] * 10
        b = [("search", "fetch", "answer")] * 10
        v = compare_profiles(a, b)
        assert v.verdict == PROFILE_REGRESSION
        assert [t.tool for t in v.hotspots] == ["verify"]

    def test_call_count_change_detected(self):
        # Same tools, but the agent started calling search twice.
        a = [("search", "answer")] * 10
        b = [("search", "search", "answer")] * 10
        v = compare_profiles(a, b)
        assert v.verdict == PROFILE_REGRESSION
        assert [t.tool for t in v.hotspots] == ["search"]

    def test_insufficient(self):
        v = compare_profiles([("a",)], [("a",)] * 5)
        assert v.verdict == PROFILE_INSUFFICIENT


class TestViewComplementarity:
    """Order-only changes: position view fires, profile view stays quiet.
    The disagreement between views IS the diagnostic."""

    def test_pure_reorder_splits_the_views(self):
        a = [("search", "fetch", "answer")] * 10
        b = [("fetch", "search", "answer")] * 10
        pos = compare_trajectories(a, b)
        prof = compare_profiles(a, b)
        assert pos.verdict == "REGRESSION"        # order genuinely changed
        assert prof.verdict == PROFILE_WITHIN_NOISE  # usage did not

    def test_ordering_noise_masks_position_but_not_profile(self):
        # Parallel-ish recording: both sides shuffle search/fetch order run
        # to run, but B lost its verify call. The position view's floor is
        # inflated at steps 1-2; the profile view still nails verify.
        a = ([("search", "fetch", "verify", "answer"),
              ("fetch", "search", "verify", "answer")] * 5)
        b = ([("search", "fetch", "answer"),
              ("fetch", "search", "answer")] * 5)
        prof = compare_profiles(a, b)
        assert prof.verdict == PROFILE_REGRESSION
        assert [t.tool for t in prof.hotspots] == ["verify"]


def write_traces(path, sequences, params=None):
    path.write_text(json.dumps([
        {"session_id": f"s{i}", "steps": [
            {"step_id": f"{i}-{j}", "tool_name": tool,
             "parameters": (params or {}).get(tool, {})}
            for j, tool in enumerate(seq)]}
        for i, seq in enumerate(sequences)]))
    return path


class TestParamKeys:
    def test_param_keys_refine_categories(self, tmp_path):
        f = write_traces(tmp_path / "runs.json", [("search",)],
                         params={"search": {"query": "x", "limit": 5}})
        plain = load_trace_runs(f)
        refined = load_trace_runs(f, include_param_keys=True)
        assert plain == [("search",)]
        assert refined == [("search[limit,query]",)]

    def test_call_shape_regression_only_visible_with_param_keys(self, tmp_path):
        # Same tool, same position, same pass — but the agent stopped
        # filling `limit`. Invisible without --param-keys.
        a = write_traces(tmp_path / "a.json", [("search",)] * 10,
                         params={"search": {"query": "x", "limit": 5}})
        b = write_traces(tmp_path / "b.json", [("search",)] * 10,
                         params={"search": {"query": "x"}})
        plain = compare_trajectories(load_trace_runs(a), load_trace_runs(b))
        refined = compare_trajectories(
            load_trace_runs(a, include_param_keys=True),
            load_trace_runs(b, include_param_keys=True))
        assert plain.verdict == "WITHIN_NOISE"
        assert refined.verdict == "REGRESSION"


class TestCliViews:
    def _files(self, tmp_path):
        a = write_traces(tmp_path / "a.json",
                         [("search", "fetch", "verify", "answer")] * 10)
        b = write_traces(tmp_path / "b.json",
                         [("search", "fetch", "answer")] * 10)
        return a, b

    def test_both_views_default(self, tmp_path, capsys):
        a, b = self._files(tmp_path)
        code = main(["traj", "--a", str(a), "--b", str(b)])
        out = capsys.readouterr().out
        assert code == 1
        assert "position view" in out
        assert "tool profile view" in out

    def test_profile_only(self, tmp_path, capsys):
        a, b = self._files(tmp_path)
        code = main(["traj", "--a", str(a), "--b", str(b),
                     "--view", "profile"])
        out = capsys.readouterr().out
        assert code == 1
        assert "position view" not in out
        assert "verify" in out

    def test_json_contains_both(self, tmp_path, capsys):
        a, b = self._files(tmp_path)
        main(["traj", "--a", str(a), "--b", str(b), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["steps"]["verdict"] == "REGRESSION"
        assert payload["profile"]["verdict"] == "REGRESSION"
