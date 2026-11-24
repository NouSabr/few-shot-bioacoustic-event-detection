#!/usr/bin/env bash
#
# Full masking sweep: verify, smoke-test, then run the real thing.
#
#   1. pre-flight   seconds   static config + wiring checks and unit tests
#   2. smoke        minutes   one train/val batch per configuration, real data
#   3. sweep        days      8 configurations x 3 seeds = 24 training runs
#
# Each stage gates the next, because stage 3 is a multi-day commitment and every
# failure mode worth catching is cheaper to catch in stage 1 or 2.
#
# The sweep is resumable. It runs under a single sweep id, and any interruption
# prints the exact command to pick up where it stopped -- completed runs are
# skipped by their stage-timings file, not re-run.
#
# Usage:
#   ./run_full_sweep.sh                      # all three stages, prompts before the sweep
#   ./run_full_sweep.sh --yes                # no prompt (nohup / tmux / screen)
#   ./run_full_sweep.sh --resume 20260917_101500
#   ./run_full_sweep.sh --smoke-only         # stop after stage 2
#   ./run_full_sweep.sh --skip-smoke         # stages 1 and 3 only
#   ./run_full_sweep.sh --progress-bar       # keep Lightning's per-batch bar
#
# Environment:
#   PYTHON   interpreter to use (default: python)
#   SEEDS    space-separated override of the seed list
#
# Output. Three destinations, each with one job:
#
#   terminal                    stage banners, per-run milestones, stderr
#   logs/sweeps/sweep_<id>.log  the above, plus each run's raw stdout
#   <run dir>/stdout.log        each run's raw stdout, written by train.py
#   <run dir>/train.log         each run's logging records, written by Hydra
#
# A training run's stdout is appended STRAIGHT to the sweep log, never through
# the tee, and Lightning's progress bar is OFF by default. Both are performance
# fixes, not preferences.
#
# TQDMProgressBar writes to stdout. train.py wraps sys.stdout in a _Tee whose
# __getattr__ proxies isatty() to the real stdout -- so under this script the bar
# asks a PIPE whether it is a terminal, gets no, and stops redrawing in place
# (symptom: one fresh line per batch). Worse, every refresh then has to be
# consumed by the terminal emulator before the writer unblocks. A four-hour run
# refreshes on the order of 10^5 times, and at num_workers=0 there is no
# prefetching worker to hide the stall behind, so a slow terminal can throttle
# the training loop itself.
#
# The bar is also written twice -- once by _Tee to the run's stdout.log, once to
# the sweep log -- which is harmless at a few KB and tens of MB of
# carriage-return spam on one line at 10^5 refreshes. Off by default, therefore.
# --progress-bar restores it if you want to watch: tail -f logs/sweeps/...
#
# Nothing is lost with the bar off. Per-epoch metrics go to the configured
# logger, results to <run dir>/<alpha>/Eval_raw.csv, logging records to
# train.log, and stage timings to logs/timings/<exp_name>.json.

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

PYTHON="${PYTHON:-python}"
read -r -a SEEDS <<< "${SEEDS:-1234}" # 2026 3009

ASSUME_YES=0
SMOKE_ONLY=0
SKIP_SMOKE=0
SKIP_PREFLIGHT=0
PROGRESS_BAR=0
RESUME_ID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --yes|-y)         ASSUME_YES=1 ;;
        --smoke-only)     SMOKE_ONLY=1 ;;
        --skip-smoke)     SKIP_SMOKE=1 ;;
        --skip-preflight) SKIP_PREFLIGHT=1 ;;
        --progress-bar)   PROGRESS_BAR=1 ;;
        --resume)         RESUME_ID="${2:?--resume needs a sweep id}"; shift ;;
        # Prints the header block: every line up to, but not including, the
        # first non-comment line. Keeps --help correct as the header grows.
        -h|--help)        sed -n '2,${/^[^#]/q;p;}' "$0" | sed 's/^# \{0,1\}//'
                          exit 0 ;;
        *) echo "Unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

SWEEP_ID="${RESUME_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="logs/sweeps"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sweep_${SWEEP_ID}.log"

# This script's OWN output is tee'd, so the console and the log agree on the
# stage banners and the pre-flight checks. The training runs in stages 2 and 3
# are the exception: their stdout is appended directly to $LOG_FILE, bypassing
# this pipe. See the "Output" note in the header for why.
exec > >(tee -a "$LOG_FILE") 2>&1

# Unbuffer the runners' stdout. Python block-buffers when stdout is a file
# rather than a terminal, so without this `tail -f` on the log would lag the run
# by up to 8 KB -- which for a progress bar is minutes of apparent silence.
export PYTHONUNBUFFERED=1

# Applied to every run in the plan. `++` and not a plain override, because
# configs/trainer/default.yaml has enable_progress_bar only as a comment (line
# 33), and Hydra's struct mode refuses a plain override of a key no composed
# config defines -- the same trap that broke the smoke stage earlier.
RUNNER_OVERRIDES=()
if [[ $PROGRESS_BAR -eq 0 ]]; then
    RUNNER_OVERRIDES+=(--extra-override "++trainer.enable_progress_bar=false")
fi

# Resuming keeps the pre-flight checks: they take seconds, and if anything was
# edited between sessions that is exactly when you want to hear about it. Only
# the smoke stage is skipped, having already passed for this sweep.
RESUME_CMD="./run_full_sweep.sh --resume $SWEEP_ID --skip-smoke --yes"

banner() {
    echo
    echo "=============================================================================="
    echo "$*"
    echo "=============================================================================="
}

stage_failed() {
    local code=$?
    [[ $code -eq 0 ]] && return 0
    banner "FAILED (exit $code) during: ${CURRENT_STAGE:-unknown}"
    echo "Log: $LOG_FILE"
    if [[ "${CURRENT_STAGE:-}" == "sweep" ]]; then
        echo
        echo "Completed runs are recorded, so resume rather than restarting:"
        echo "  $RESUME_CMD"
    fi
    exit "$code"
}
trap stage_failed EXIT

elapsed() {  # elapsed <start_epoch>
    local secs=$(( $(date +%s) - $1 ))
    printf '%dh %02dm %02ds' $((secs/3600)) $((secs%3600/60)) $((secs%60))
}

banner "Masking sweep $SWEEP_ID"
echo "Started:     $(date)"
echo "Interpreter: $PYTHON  ($($PYTHON --version 2>&1))"
echo "Seeds:       ${SEEDS[*]}"
echo "Log:         $LOG_FILE"
[[ -n "$RESUME_ID" ]] && echo "Resuming a previous sweep; finished runs will be skipped."

CURRENT_STAGE="environment"
[[ -f train.py ]] || { echo "train.py not found - run this from the task directory"; exit 1; }

# The dataset and GPU are checked just before stage 2, which is the first stage
# that needs them. Stage 1 needs neither, so it stays useful on a machine that
# has only a checkout.
check_run_environment() {
    $PYTHON -c "import torch, pytorch_lightning" >/dev/null 2>&1 \
        || { echo "torch / pytorch_lightning not importable by '$PYTHON'"; exit 1; }

    local data_dir
    data_dir="$($PYTHON - <<'PY'
import yaml
cfg = yaml.safe_load(open("configs/train.yaml"))
print(cfg["path"]["root_dir"])
PY
)"
    if [[ ! -d "$data_dir" ]]; then
        echo "Dataset root not found: $data_dir"
        echo "Fix path.root_dir in configs/train.yaml before starting a sweep."
        exit 1
    fi
    echo "Dataset root: $data_dir  (present)"

    if $PYTHON -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "CUDA:         available"
    else
        echo "CUDA:         NOT available - trainer.gpus=1 will fail. Aborting."
        exit 1
    fi
}

# --- stage 1: static checks ---------------------------------------------------
if [[ $SKIP_PREFLIGHT -eq 0 ]]; then
    CURRENT_STAGE="pre-flight"
    banner "Stage 1/3  Pre-flight (seconds, no data)"
    t0=$(date +%s)

    echo "--- config sanity check ---"
    $PYTHON tools/sanity_check_configs.py

    echo
    echo "--- masking wiring verification ---"
    $PYTHON tools/verify_masking_wiring.py

    echo
    echo "--- unit tests ---"
    $PYTHON -m pytest \
        tests/unit/test_phase1_frequency_masking.py \
        tests/unit/test_phase2_frequency_masking.py \
        tests/unit/test_phase3_frequency_masking.py \
        tests/unit/test_afri_frequency_masking.py \
        -q --no-header

    echo
    echo "Pre-flight passed in $(elapsed "$t0")."
else
    echo
    echo "Skipping stage 1 (--skip-preflight)."
fi

# --- stage 2: one batch per configuration, on the real data -------------------
if [[ $SKIP_SMOKE -eq 0 || $SMOKE_ONLY -eq 0 ]]; then
    CURRENT_STAGE="environment"
    banner "Checking the run environment"
    check_run_environment
fi

if [[ $SKIP_SMOKE -eq 0 ]]; then
    CURRENT_STAGE="smoke"
    banner "Stage 2/3  Smoke test (one train/val batch per configuration)"
    echo "Exercises what the static checks cannot: dataset construction, AFRI and"
    echo "energy profile computation or cache load, the tracker -> masker path,"
    echo "HDF5 reads, collate and the forward pass."
    echo
    echo "NOTE: if the AFRI or energy profile caches are cold this stage computes"
    echo "      them, which is slow the first time and near-instant afterwards."
    t0=$(date +%s)

    echo "      Milestones appear here; each run's full stdout goes to the log"
    echo "      and to its own run dir:  tail -f $LOG_FILE"
    echo

    $PYTHON run_ablation_experiments.py --fast-dev-run --no-pause \
        --seeds "${SEEDS[0]}" \
        ${RUNNER_OVERRIDES[@]+"${RUNNER_OVERRIDES[@]}"} \
        >>"$LOG_FILE"

    echo
    echo "Smoke test passed in $(elapsed "$t0")."
else
    echo
    echo "Skipping stage 2 (--skip-smoke)."
fi

if [[ $SMOKE_ONLY -eq 1 ]]; then
    banner "Stopping after the smoke test (--smoke-only)"
    echo "To run the sweep:  ./run_full_sweep.sh --skip-preflight --skip-smoke --yes"
    trap - EXIT
    exit 0
fi

# --- stage 3: the real thing --------------------------------------------------
CURRENT_STAGE="sweep"
N_CONFIGS="$($PYTHON - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("abl", "run_ablation_experiments.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(len(m.EXPERIMENTS))
PY
)"
N_RUNS=$(( N_CONFIGS * ${#SEEDS[@]} ))

banner "Stage 3/3  Ablation sweep"
echo "$N_CONFIGS configurations x ${#SEEDS[@]} seeds = $N_RUNS training runs, seed-major."
echo "At roughly 4-5 hours per run this is on the order of $(( N_RUNS * 9 / 2 )) hours."
echo "Sweep id: $SWEEP_ID"
echo
echo "Resume at any point with:"
echo "  $RESUME_CMD"

if [[ $ASSUME_YES -eq 0 ]]; then
    echo
    read -r -p "Start the sweep? [y/N] " reply
    if [[ ! "$reply" =~ ^[Yy]$ ]]; then
        banner "Aborted before the sweep"
        echo "Nothing was run. Pre-flight and smoke results are in $LOG_FILE."
        trap - EXIT
        exit 0
    fi
fi

echo
echo "Only per-run milestones and errors appear here. For detail:"
echo "  tail -f $LOG_FILE"
echo "  tail -f logs/experiments/runs/<config>_seed<seed>/*/train.log"
if [[ $PROGRESS_BAR -eq 0 ]]; then
    echo "Progress bar: off (--progress-bar to keep it; see the header for why)."
fi

t0=$(date +%s)
$PYTHON run_ablation_experiments.py \
    --seeds "${SEEDS[@]}" \
    --timestamp "$SWEEP_ID" \
    --skip-existing \
    --no-pause \
    ${RUNNER_OVERRIDES[@]+"${RUNNER_OVERRIDES[@]}"} \
    >>"$LOG_FILE"

banner "Sweep $SWEEP_ID complete"
echo "Finished: $(date)"
echo "Wall clock for stage 3: $(elapsed "$t0")"
echo
echo "Results:        logs/ablation_studies/ablation_results_${SWEEP_ID}.json"
echo "Stage timings:  logs/timings/<config>_seed<seed>_${SWEEP_ID}.json"
echo "Run dirs:       logs/experiments/runs/<config>_seed<seed>/"
echo "Full log:       $LOG_FILE"

trap - EXIT
