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

## Trajectory statistics (`nunchi-eval traj`)

Compare two populations of agent runs (ExecutionTrace JSONs) through two
complementary views, both noise-floor gated:

- **position view** — "what did the agent do at step i" (*where* it changed)
- **tool profile view** — "how many times did each tool get called",
  order-blind (*what* changed; immune to parallel-call ordering noise)

```bash
nunchi-eval traj --a baseline_runs/ --b candidate_runs/          # both views
nunchi-eval traj --a a.json --b b.json --view profile            # one view
nunchi-eval traj --a a.json --b b.json --param-keys              # see below
```

```
== position view (where) ==
step  majority A     majority B       cross  floor
3     verify (100%)  answer (70%)     1.00   0.47   <- hotspot
4     answer (100%)  <absent> (100%)  1.00   0.00   <- hotspot
trajectories: mean cross=0.575 > floor=0.233, p=0.001 -> REGRESSION

== tool profile view (what, order-blind) ==
tool    majority calls A  majority calls B  cross  floor
verify  1x (100%)         0x (100%)         1.00   0.00   <- hotspot
tool profile: mean cross=0.281 > floor=0.094, p=0.001 -> REGRESSION, hotspot tools: ['verify']
```

When the views disagree, that's a diagnosis, not a bug: position=REGRESSION
with profile=WITHIN_NOISE means the agent calls the same tools in a
different order.

`--param-keys` refines step categories from `search` to
`search[limit,query]` — the set of parameter keys each call filled. This
catches "same tool, same verdict, different call shape" regressions while
keeping categories finite (raw parameter values would explode cardinality).

Alignment is position-based with an `<absent>` sentinel: a trajectory that
got shorter *is* a behavioral change, not something to align away.

MIT license.
