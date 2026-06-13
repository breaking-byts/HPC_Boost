# Detection-Aware Beam Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a server-runnable, leakage-free candidate-pool oracle using beam search and XGBoost.

**Architecture:** A single experiment module owns feature aggregation, nested
beam search, candidate evaluation, oracle routing, checkpointing, and CLI
orchestration. Focused unit tests exercise the search and routing functions with
synthetic arrays without requiring the RaDaR dataset.

**Tech Stack:** Python, NumPy, pandas, SciPy, scikit-learn, XGBoost, joblib,
pytest.

---

### Task 1: Define Search Behavior With Tests

**Files:**
- Create: `hpc_boost_v2/tests/test_detection_aware_beam_oracle.py`
- Create: `hpc_boost_v2/experiments/exp3_detection_aware_oracle/__init__.py`

- [ ] Write tests for canonical subset expansion, deterministic pruning,
  label-assisted oracle routing, and invalid parameters.
- [ ] Run `pytest -q hpc_boost_v2/tests/test_detection_aware_beam_oracle.py`
  and verify failure because the experiment module does not exist.

### Task 2: Implement Search and Routing Primitives

**Files:**
- Create: `hpc_boost_v2/experiments/exp3_detection_aware_oracle/run_beam_oracle.py`

- [ ] Implement canonical subset generation and validation.
- [ ] Implement inner-CV AUCPR scoring with fold-local scaling and XGBoost.
- [ ] Implement beam expansion with deterministic score and lexical tie breaks.
- [ ] Implement candidate-oracle routing that selects the highest-ranked correct
  candidate and falls back to the global candidate.
- [ ] Run the focused tests and verify they pass.

### Task 3: Implement Nested Evaluation and Artifacts

**Files:**
- Modify: `hpc_boost_v2/experiments/exp3_detection_aware_oracle/run_beam_oracle.py`

- [ ] Reuse `RadarDataLoader` and robust six-statistic aggregation.
- [ ] Implement outer stratified CV, fold-local 2SMaRT, Global-Beam, and
  Candidate-Oracle evaluation.
- [ ] Write fold candidate CSVs, prediction CSVs, summaries, checkpoints, and
  final aggregate files atomically.
- [ ] Add CLI arguments, progress logging, runtime estimates, and resume support.

### Task 4: Verify Locally and Prepare Server Commands

**Files:**
- Verify: `hpc_boost_v2/tests/test_detection_aware_beam_oracle.py`
- Verify: `hpc_boost_v2/experiments/exp3_detection_aware_oracle/run_beam_oracle.py`

- [ ] Run focused and existing tests.
- [ ] Run Python bytecode compilation and CLI help.
- [ ] Provide commands for `scp`, dependency verification, `tmux` execution,
  log monitoring, result packaging, and downloading artifacts.
