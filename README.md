# nunchi-eval

**Put p-values on snapshot diffs.**

Snapshot tools for AI agents ([EvalView](https://github.com/hidai25/eval-view)
and friends) tell you *what* changed between runs. They can't tell you whether
that change exceeds the model's own run-to-run noise — so fixed thresholds
("2 flips = MEDIUM drift") alarm at the same rate on quiet and noisy prompts.

nunchi-eval reads the raw per-run outcomes those tools already record and
answers the question that matters: **is this diff a regression, or is it the
noise floor?** Per prompt: flip rate with a bootstrap 95% CI on both sides,
cross-snapshot disagreement gated by the noise floor, and a permutation
p-value. Built on [nunchi-drift](https://github.com/hadakang/nunchi-drift).

```
$ nunchi-eval check .evalview/model_snapshots/claude-haiku-4-5/

prompt         category        flip A [95% CI]   flip B [95% CI]   cross  floor  p      verdict
-------------  --------------  ----------------  ----------------  -----  -----  -----  ------------
exact-4        exact_match     0.00 [0.00,0.00]  0.20 [0.00,0.47]  0.90   0.20   0.001  REGRESSION
json-schema-2  json_schema     0.53 [0.20,0.56]  0.56 [0.36,0.56]  0.50   0.56   1.000  WITHIN_NOISE
refusal-3      refusal         0.36 [0.00,0.53]  0.53 [0.20,0.56]  0.44   0.53   0.625  WITHIN_NOISE
tool-select-1  tool_selection  0.00 [0.00,0.00]  0.00 [0.00,0.00]  0.00   0.00   1.000  WITHIN_NOISE

4 prompt(s): 1 regression(s), 3 within noise
```

Note the third row: a prompt drops from 8/10 to 6/10 passing. Flip-count
thresholds alarm; the permutation test says p=0.625 — at n=10 on an
already-wobbly prompt, that's not a supportable regression claim. The first
row *is* one (p=0.001), and the exit code (1) is CI-friendly.

## Install & use

```bash
pip install nunchi-eval   # not yet published — coming with 0.1.0

# default: pinned reference vs latest snapshot
nunchi-eval check .evalview/model_snapshots/<model>/

# explicit pair, custom significance, JSON output
nunchi-eval check --a reference.json --b today.json --alpha 0.01 --json
```

Works with EvalView >= 0.8 `model-check` snapshots (`per_run_passed`,
schema_version 1). Refuses to compare across different canary suites
(`suite_hash` mismatch), same as EvalView itself.

## As a library

```python
from nunchi_eval import PromptRuns, compare_batteries

a = [PromptRuns("route-request", runs=[True]*10)]
b = [PromptRuns("route-request", runs=[True]*6 + [False]*4)]
report = compare_batteries(a, b, alpha=0.05)
for v in report.verdicts:
    print(v.summary())
```

The battery math is tool-agnostic — any per-prompt `List[bool]` works.
Adapters for other snapshot formats (trajectory-level statistics with
step alignment) are the roadmap.

MIT license.
