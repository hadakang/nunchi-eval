"""Battery verdict logic: the noise-floor gate must separate three regimes."""

import pytest

from nunchi_eval import (
    VERDICT_INSUFFICIENT,
    VERDICT_REGRESSION,
    VERDICT_WITHIN_NOISE,
    PromptRuns,
    compare_batteries,
    judge_prompt,
)


def runs(prompt_id, bools, category="tool_selection"):
    return PromptRuns(prompt_id=prompt_id, category=category, runs=tuple(bools))


class TestJudgePrompt:
    def test_identical_stable_runs_are_noise(self):
        a = runs("p1", [True] * 10)
        b = runs("p1", [True] * 10)
        v = judge_prompt(a, b)
        assert v.verdict == VERDICT_WITHIN_NOISE
        assert v.cross == 0.0
        assert v.floor == 0.0

    def test_hard_regression_fires(self):
        # Perfectly stable on both sides, completely flipped between them:
        # cross=1.0, floor=0.0 — the clearest possible regression.
        a = runs("p1", [True] * 10)
        b = runs("p1", [False] * 10)
        v = judge_prompt(a, b)
        assert v.verdict == VERDICT_REGRESSION
        assert v.cross == 1.0
        assert v.p_value <= 0.05

    def test_noisy_prompt_same_distribution_is_noise(self):
        # A prompt that flips ~50% either side, with the same mix on both
        # sides. Fixed-threshold flip counting alarms here; the floor gate
        # must not.
        a = runs("p1", [True, False] * 5)
        b = runs("p1", [False, True] * 5)
        v = judge_prompt(a, b)
        assert v.verdict == VERDICT_WITHIN_NOISE
        # cross ~0.5 but the floor is also ~0.5 — explained by self-noise
        assert v.floor >= 0.5

    def test_small_shift_on_noisy_prompt_is_noise(self):
        # 8/10 -> 6/10 pass on an already-wobbly prompt: not a supportable
        # regression claim at n=10.
        a = runs("p1", [True] * 8 + [False] * 2)
        b = runs("p1", [True] * 6 + [False] * 4)
        v = judge_prompt(a, b)
        assert v.verdict == VERDICT_WITHIN_NOISE

    def test_insufficient_data(self):
        v = judge_prompt(runs("p1", [True]), runs("p1", [True] * 10))
        assert v.verdict == VERDICT_INSUFFICIENT
        assert v.p_value is None

    def test_deterministic_given_seed(self):
        a = runs("p1", [True] * 7 + [False] * 3)
        b = runs("p1", [False] * 7 + [True] * 3)
        p1 = judge_prompt(a, b, seed=0).p_value
        p2 = judge_prompt(a, b, seed=0).p_value
        assert p1 == p2


class TestCompareBatteries:
    def test_matching_and_unmatched_prompts(self):
        battery_a = [runs("shared", [True] * 10), runs("only-a", [True] * 10)]
        battery_b = [runs("shared", [False] * 10), runs("only-b", [True] * 10)]
        report = compare_batteries(battery_a, battery_b)
        assert [v.prompt_id for v in report.verdicts] == ["shared"]
        assert report.unmatched_a == ["only-a"]
        assert report.unmatched_b == ["only-b"]
        assert len(report.regressions) == 1

    def test_summary_counts(self):
        battery_a = [runs("p1", [True] * 10), runs("p2", [True, False] * 5)]
        battery_b = [runs("p1", [False] * 10), runs("p2", [False, True] * 5)]
        report = compare_batteries(battery_a, battery_b)
        assert len(report.regressions) == 1
        assert len(report.within_noise) == 1
        assert "1 regression(s)" in report.summary()


class TestPromptRuns:
    def test_coerces_to_bool_tuple(self):
        p = PromptRuns(prompt_id="p", runs=[1, 0, 1])
        assert p.runs == (True, False, True)
        assert p.pass_rate == pytest.approx(2 / 3)
