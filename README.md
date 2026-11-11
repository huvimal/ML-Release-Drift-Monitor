# ML Release & Drift Monitor

A CPU-only, standard-library ML lifecycle lab: train logistic classifiers, gate a
candidate on held-out data, replay post-release windows, inspect descriptive drift,
and apply an evidence-guarded **local registry rollback**.

This is an offline simulation. It does not deploy a model server, receive live
production traffic, or change external infrastructure. All included data is synthetic.

## Quick start — Python 3.11+

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe monitor.py train
.\.venv\Scripts\python.exe monitor.py initialize
.\.venv\Scripts\python.exe monitor.py promote --candidate candidate-v2 --expected stable-v1 --request release-v2
.\.venv\Scripts\python.exe monitor.py watch --window w01
.\.venv\Scripts\python.exe monitor.py watch --window w02
.\.venv\Scripts\python.exe monitor.py watch --window w03
.\.venv\Scripts\python.exe monitor.py watch --window w04
.\.venv\Scripts\python.exe monitor.py rollback --expected candidate-v2 --target stable-v1 --evidence w04 --request rollback-001
.\.venv\Scripts\python.exe monitor.py watch --window w05
Start-Process reports/dashboard.html
```

No API key, GPU, network request, external package, or activation script is required.
`make_data.py` regenerates committed fixtures deterministically.

## Models and historical release gate

Historical data contains 500 unique events ordered into 300 train, 100 dev, and 100
test observations. Targets are synthetic binary urgency labels. Features are a
synthetic structured `urgency_score`, numeric `words`, and categorical `channel`.
There is no text encoder or real ticket classifier here: urgency_score is supplied
by the fixture, not inferred from Vietnamese language.

Two actual logistic models are trained by SGD:
- `stable-v1`: urgency_score only.
- `candidate-v2`: urgency_score and words; length is spuriously correlated with
  the target in historical data.

Scaling fits train rows only. Threshold selection uses dev accuracy over the fixed
grid 0.4/0.5/0.6. Test labels do not affect fitted weights or threshold selection.
`faulty-v0` is a deliberate fault-injection artifact with inverted candidate weights,
**not** a third independently trained model.

The release gate compares the fixed artifacts on the same held-out test events.
Default policy: at least 40 test labels, candidate accuracy >= 0.90, accuracy drop
versus active baseline <= 0.03, and positive-class recall >= 0.80. Promotion also
checks the expected active version under a SQLite write transaction.

Observed fixture test accuracy: stable-v1 = 0.99, candidate-v2 = 1.00, faulty-v0 = 0.00.
These are constructed teaching scenarios, not real-world performance estimates.
Inspect the faulty rejection explicitly:

```powershell
.\.venv\Scripts\python.exe monitor.py promote --candidate faulty-v0 --expected stable-v1 --request bad-v0
$LASTEXITCODE
```

Run that before promoting v2. The command returns 1 and leaves the registry unchanged.
Release metrics include Wilson accuracy intervals; policy uses fixed point-estimate
floors, not a calibrated noninferiority test. Do not repeatedly tune on the test gate.

## Monitoring windows

| Window | Visible labels | Scenario | Expected decision |
| --- | ---: | --- | --- |
| w01 | 100 | Original distribution | healthy |
| w02 | 0 | Feature/channel shift, labels arrive later | investigate |
| w03 | 100 | Spurious length correlation reverses | watch |
| w04 | 100 | Persistent degradation, fallback better | rollback_recommended |
| w05 | 100 | Normal data after manual local rollback | healthy |

Each window has 100 events. `labels.json` stores label availability separately.
The join uses event ID and `available_at <= as_of`; future labels are excluded,
and labels available before their event are invalid. A zero-label window reports
accuracy as `null`, never zero accuracy or an assumed success.

Numeric drift uses PSI against fixed train-reference quantile bins, plus an explicit
missing-value bucket. Categorical drift uses base-2 Jensen-Shannon **divergence**
(not distance), including unseen channel categories. Default PSI/JSD thresholds are
illustrative heuristics, not p-values, universal cutoffs, or statistically calibrated
alarms. PSI uses small additive smoothing on bin counts. Missing numeric features
are mean-imputed by the model but are still counted in drift reports.

Drift alone does not establish model degradation. Marginal distributions can remain
similar while feature/label relationships change; these univariate scores cannot
detect every joint or concept shift. Prediction probabilities are not calibrated
confidence estimates. No accuracy is inferred from drift scores.

## Recommendation and rollback policy

Label evidence is sufficient only when at least 40 labels and 60% of the window's
events are visible. A bad window requires all of:

1. Sufficient labels and active accuracy < 0.85.
2. Fallback accuracy exceeds active accuracy by at least 0.10 on the **same labeled events**.
3. The lower bound of a deterministic paired bootstrap interval for that gain is > 0.

Two consecutive bad windows within the same deployment epoch produce
`rollback_recommended`. Insufficient-label, healthy, or otherwise unsupported windows
reset the streak. The paired bootstrap assumes IID events; real correlated traffic
needs group/time-aware resampling and appropriately calibrated monitoring policy.
Wilson/bootstrap intervals are descriptive; they do not address selection bias from
which labels arrive, multiple monitoring tests, or repeated peeking.

`watch` never changes the active model. The explicit `rollback` command requires the
expected active version, the registered fallback target, and the **latest recorded
observation** of the current epoch to recommend that exact rollback. Latest means
latest in this registry, not a wall-clock freshness guarantee. No external deployment
or production traffic is changed. Promotion/rollback increment the epoch, which
prevents an earlier deployment's bad streak from carrying over.

## Registry and reproducibility

`private/registry.db` stores immutable registered artifacts, transition requests,
monitoring observations and event ownership; its active state changes transactionally.
Request IDs are idempotent: replaying the same transition returns the recorded result
and current state without applying it again. Reusing the ID for different arguments
fails. An observation replay returns its historical report with `replayed: true`;
it does not become fresh evidence or increment the bad streak.

Window event intervals must be disjoint and chronological. Duplicate event IDs across
windows are rejected. Window snapshots are immutable: changing input or as-of time
under the same window ID is rejected. This prototype does **not** revise old windows
when delayed labels eventually arrive. Use a separate experiment DB to reevaluate a
different as-of dataset; don't claim continuous late-label backfill support.

Registry configuration freezes policy, historical data, drift reference and core
implementation hashes. Changed configuration requires a separate DB. Model versions
are immutable: different weights require a new version. These checks catch normal
workflow errors, not a hostile database/filesystem owner. There is no authentication,
signed model registry, scheduler, Kubernetes integration, retraining job or live endpoint.

Global `--db PATH`, `--data DIRECTORY`, and `--policy FILE` precede the subcommand.
Alternative data must use the same feature/task schemas. CLI model training writes
to the shared `models/` folder, and reports to shared `reports/`; isolate the project
folder too when running different collections concurrently. Frozen-window summaries
are replay analyses using registered model artifacts, not historical serving logs.

## Artifacts

- `models/*.json`: weights, scaling, features, selected threshold, training fingerprint.
- `reports/training.json`: model fingerprints and dev metrics.
- `reports/release-*.json`: held-out release gate decisions.
- `reports/w01.json` through `w05.json`: labels, quality, drift, gain evidence and fingerprints.
- `reports/dashboard.html`: current local state and historical monitoring decisions.

The 24 unit tests cover training leakage, delayed labels, numerical/category drift,
missing values, insufficient evidence, idempotency, version guards, repeated windows,
stale rollback evidence and recovery. Offline CI exercises the full synthetic flow.
Generated artifacts and database files are gitignored; no real customer logs are included.

## Primary references

- [Evidently: data drift as a proxy without ground truth](https://docs.evidentlyai.com/metrics/preset_data_drift)
- [Python SQLite transactions](https://docs.python.org/3.11/library/sqlite3.html)

This implementation is independent of Evidently and does not depend on scikit-learn.
