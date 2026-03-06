# Phase 2 Execution Plan (HMM Core)

This plan covers:
- Issue breakdown for Phase 2 (`src/hmm/*`)
- Branch strategy and merge flow
- Test suite for Phase 2
- Combined regression tests for Phase 1 + Phase 2

It follows your preferred workflow: finish each feature branch, merge it, then move to the next branch.

## 1) Scope and Definition of Done

Phase 2 files:
- `src/hmm/forward.py`
- `src/hmm/backward.py`
- `src/hmm/forward_backward.py`
- `src/hmm/baum_welch.py`
- `src/hmm/viterbi.py`
- `src/hmm/model_selection.py`
- `src/hmm/inference.py`

Done criteria per function:
1. Function implemented with full math docstring (paper reference + equations + params + returns).
2. Targeted tests implemented and passing.
3. Edge cases covered (`empty`, `single observation`, `K=1` where applicable).
4. Existing tests remain green (no regressions).
5. Branch merged before starting next issue.

## 2) Issue and Branch Plan

Issue numbering continues after Phase 1 (`#1` to `#4` already used).

1. `#5` Forward algorithm
   - Branch: `feat/5-forward`
   - Files:
     - `src/hmm/forward.py`
     - `tests/test_forward.py`
   - Key acceptance:
     - log-space forward recursion implemented
     - synthetic test + hmmlearn log-likelihood comparison

2. `#6` Backward algorithm
   - Branch: `feat/6-backward`
   - Files:
     - `src/hmm/backward.py`
     - `tests/test_backward.py`
   - Key acceptance:
     - log-space backward recursion implemented
     - consistency with forward likelihood (`alpha * beta`)

3. `#7` Forward-Backward posteriors
   - Branch: `feat/7-forward-backward`
   - Files:
     - `src/hmm/forward_backward.py`
     - `tests/test_forward_backward.py`
   - Key acceptance:
     - `gamma` row sums to 1
     - `xi` normalization/shape checks

4. `#8` Baum-Welch M-step and training loop
   - Branch: `feat/8-baum-welch`
   - Files:
     - `src/hmm/baum_welch.py`
     - `tests/test_baum_welch.py`
   - Key acceptance:
     - M-step updates match formulas
     - EM log-likelihood is non-decreasing
     - synthetic parameter recovery sanity check

5. `#9` Viterbi decoding
   - Branch: `feat/9-viterbi`
   - Files:
     - `src/hmm/viterbi.py`
     - `tests/test_viterbi.py`
   - Key acceptance:
     - MAP state path and path score
     - known-sequence synthetic test

6. `#10` Model selection
   - Branch: `feat/10-model-selection`
   - Files:
     - `src/hmm/model_selection.py`
     - `tests/test_model_selection.py`
   - Key acceptance:
     - AIC/BIC formulas verified
     - `select_K` returns expected best `K` on controlled data

7. `#11` Online inference
   - Branch: `feat/11-inference`
   - Files:
     - `src/hmm/inference.py`
     - `tests/test_inference.py`
   - Key acceptance:
     - predict-update step correctness
     - batch/online consistency checks

8. `#12` Phase 2 test-hardening and utilities
   - Branch: `feat/12-phase2-test-hardening`
   - Files:
     - `tests/conftest.py` (if needed for reusable synthetic fixtures)
     - minor cleanup in `tests/test_*.py`
   - Key acceptance:
     - deterministic seeds everywhere
     - no flaky tests

9. `#13` Cross-layer integration tests (Phase 1 + Phase 2)
   - Branch: `feat/13-phase1-phase2-integration-tests`
   - Files:
     - `tests/test_phase1_phase2_integration.py`
   - Key acceptance:
     - data/features output flows into HMM functions correctly
     - end-to-end synthetic pipeline passes with valid shapes/probability constraints

## 3) Branch and Merge Workflow (Your Preferred Style)

For each issue:
1. Start from updated `master`
2. Create issue branch
3. Implement one function + tests
4. Run targeted tests
5. Run regression tests
6. Commit with issue reference
7. Merge branch into `master`
8. Move to next issue

Command template:

```bash
git checkout master
git pull
git checkout -b feat/<issue>-<name>

# code + tests
.venv/bin/python -m pytest -q <targeted tests>
.venv/bin/python -m pytest -q <regression scope>

git add <files>
git commit -m "feat: <change summary> (refs #<issue>)"

git checkout master
git merge --no-ff feat/<issue>-<name> -m "merge: integrate feat/<issue>-<name>"
```

## 4) Test Plan

### 4.1 Phase 2 Unit/Algorithm Suite

Target files:
- `tests/test_forward.py`
- `tests/test_backward.py`
- `tests/test_forward_backward.py`
- `tests/test_baum_welch.py`
- `tests/test_viterbi.py`
- `tests/test_model_selection.py`
- `tests/test_inference.py`

Minimum checks:
- numerical correctness on synthetic data
- probability constraints (normalization, valid distributions)
- shape/type checks
- edge cases and failure paths

Suggested command:

```bash
.venv/bin/python -m pytest -q \
  tests/test_forward.py \
  tests/test_backward.py \
  tests/test_forward_backward.py \
  tests/test_baum_welch.py \
  tests/test_viterbi.py \
  tests/test_model_selection.py \
  tests/test_inference.py
```

### 4.2 Combined Regression: Phase 1 + Phase 2

Phase 1 tests:
- `tests/test_loader.py`
- `tests/test_features.py`
- `tests/test_metrics.py`
- `tests/test_plotting.py`

Combined command:

```bash
.venv/bin/python -m pytest -q
```

Execution policy:
- After each Phase 2 issue: run targeted tests + quick full-suite check.
- At Phase 2 completion: run full suite twice (sanity + stability).

## 5) Proposed Integration Test (`tests/test_phase1_phase2_integration.py`)

Planned scenarios:
1. Synthetic price series -> `log_returns` -> `normalize_returns` -> `forward`
   - assert finite log-likelihood
   - assert output shapes
2. Synthetic returns -> `forward_backward`
   - assert each `gamma[t].sum() == 1`
3. Synthetic returns -> `baum_welch` -> `run_inference`
   - assert valid probabilities and prediction lengths

Notes:
- No network access in integration tests (do not call `yfinance` in tests).
- Use deterministic seeds for reproducibility.

## 6) Commit and Merge Naming

Commit pattern:
- `feat: implement <function> with tests (refs #<issue>)`
- `test: add integration checks for phase1+phase2 (refs #13)`

Merge commit pattern:
- `merge: integrate feat/<issue>-<name>`

## 7) Risks and Mitigations

1. Numerical instability in long sequences
   - Mitigation: log-space + `scipy.special.logsumexp`
2. EM local maxima / sensitivity to init
   - Mitigation: multiple restarts + deterministic test fixtures
3. Flaky stochastic tests
   - Mitigation: fixed seeds and tolerance bounds
4. Cross-branch drift
   - Mitigation: strict merge-before-next-branch policy

## 8) Immediate Next Step

Start Issue `#5` on `feat/5-forward`:
- implement `forward(...)`
- add `tests/test_forward.py`
- run targeted tests and full suite
- merge into `master`
