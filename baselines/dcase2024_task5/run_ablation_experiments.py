#!/usr/bin/env python3
"""
Automated Ablation Study Runner - Pure Isolated Configurations
All weights explicitly set to ensure no interference between mechanisms
"""

import subprocess
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
import json

# Experiment configurations with EXPLICIT weight isolation
EXPERIMENTS = [
    {
        "name": "baseline",
        "description": "Baseline (no augmentation)",
        "overrides": [
            "features.use_cife=false",
            "train_param.use_freq_masking=false",
            "train_param.use_afri=false"
        ]
    },
    {
        "name": "cife",
        "description": "CIFE only",
        "overrides": [
            "features.use_cife=true",
            "train_param.use_freq_masking=false",
            "train_param.use_afri=false"
        ]
    },
    {
        "name": "phase1_random",
        "description": "Phase 1: Pure Random Masking",
        "overrides": [
            "features.use_cife=false",
            "train_param.use_freq_masking=true",
            "train_param.freq_mask_mode=random",
            # Aggressive coverage
            "train_param.freq_mask_n_bands=4",
            "train_param.freq_mask_prob=0.8",
            "train_param.freq_mask_max_width=24",
            # Phase 1 uses no spectral signal, so none of the guided
            # strategies' parameters apply to it.
            "train_param.use_afri=false",
        ]
    },
    {
        # The all-else-equal control for the three guided strategies. 'random'
        # above differs from them in the placement rule as well as the guidance,
        # so it cannot isolate the guidance on its own. Without this row a null
        # result cannot be told apart from a geometry that cannot express
        # guidance.
        "name": "slot_uniform",
        "description": "Control: Uniform-Slot Masking (no guidance)",
        "overrides": [
            "features.use_cife=false",
            "train_param.use_freq_masking=true",
            "train_param.freq_mask_mode=slot_uniform",
            # Aggressive coverage
            "train_param.freq_mask_n_bands=4",
            "train_param.freq_mask_prob=0.8",
            "train_param.freq_mask_max_width=24",
            "train_param.use_afri=false",
        ]
    },
    {
        # Energy profile is per class and precomputed from the same first
        # afri_n_samples positives AFRI uses, so use_afri stays false here (the
        # AFRI algorithm is not run) but the segments profiled are the same.
        "name": "phase2_energy",
        "description": "Phase 2: Per-class Energy-guided Masking",
        "overrides": [
            "features.use_cife=false",
            "train_param.use_freq_masking=true",
            "train_param.freq_mask_mode=energy",
            # Aggressive coverage
            "train_param.freq_mask_n_bands=4",
            "train_param.freq_mask_prob=0.8",
            "train_param.freq_mask_max_width=24",
            "train_param.freq_mask_energy_weight=1.0",         # 100% energy-driven
            "train_param.use_afri=false",
        ]
    },
    {
        "name": "phase3_activation",
        "description": "Phase 3: Per-class Activation-guided Masking",
        "overrides": [
            "features.use_cife=false",
            "train_param.use_freq_masking=true",
            "train_param.freq_mask_mode=activation",
            # Aggressive coverage
            "train_param.freq_mask_n_bands=4",
            "train_param.freq_mask_prob=0.8",
            "train_param.freq_mask_max_width=24",
            "train_param.freq_mask_importance_weight=1.0",     # 100% activation-driven
            "train_param.use_afri=false",
            # Stable momentum so per-class scores shift only after consistent
            # encoder focus (ema = alpha*OLD + (1-alpha)*new)
            "train_param.freq_mask_importance_ema_alpha=0.9",
        ]
    },
    # AFRI runs its own strategy (freq_mask_mode=afri), so it needs none of the
    # energy/importance/afri weights that previously existed only to switch the
    # activation phase off. Direct and inverse differ in exactly one override.
    {
        "name": "afri_direct",
        "description": "AFRI Direct (mask inside the class frequency range)",
        "overrides": [
            "features.use_cife=false",
            "train_param.use_freq_masking=true",
            "train_param.freq_mask_mode=afri",
            # Aggressive coverage
            "train_param.freq_mask_n_bands=4",
            "train_param.freq_mask_prob=0.8",
            "train_param.freq_mask_max_width=24",
            "train_param.use_afri=true",
            "train_param.afri_mode=direct",
        ]
    },
    {
        "name": "afri_inverse",
        "description": "AFRI Inverse (mask outside the class frequency range)",
        "overrides": [
            "features.use_cife=false",
            "train_param.use_freq_masking=true",
            "train_param.freq_mask_mode=afri",
            # Aggressive coverage
            "train_param.freq_mask_n_bands=4",
            "train_param.freq_mask_prob=0.8",
            "train_param.freq_mask_max_width=24",
            "train_param.use_afri=true",
            "train_param.afri_mode=inverse",
        ]
    }
]

# Overrides that turn any experiment into a one-batch smoke test.
#
# fast_dev_run runs a single train and val batch, which is enough to exercise
# everything the static checker in tools/sanity_check_configs.py cannot reach:
# real dataset construction, energy/AFRI profile computation or cache load, the
# tracker -> training_step -> masker path, HDF5 reads, collate and the forward.
#
#   ++trainer....   NOTE THE ++. configs/trainer/default.yaml does not define
#                   fast_dev_run, and Hydra's struct mode rejects a plain
#                   override of an absent key ("Could not override
#                   'trainer.fast_dev_run'"). '++' appends-or-overrides, so it
#                   works whether or not the key is present -- which is also why
#                   it is preferred over '+', which fails if the key exists.
#                   tests/shell/test_basic_commands.py uses the same form.
#   test=false      the pipeline still calls trainer.test() under fast_dev_run
#                   (training_pipeline.py only forces ckpt_path=None), and one
#                   test batch is a whole file (~2 min). The test path has no
#                   masker, so it is not what we are smoke-testing.
#   logger=...      keeps eight throwaway runs out of wandb. A config-group
#                   selection, not a key override, so no '+' is wanted here.
SMOKE_OVERRIDES = [
    "++trainer.fast_dev_run=true",
    "test=false",
    "logger=tensorboard",
]


def announce(text):
    """
    Print a run-level milestone to BOTH stderr and stdout.

    run_full_sweep.sh sends this process's stdout to the sweep log and leaves
    stderr on the terminal, because Lightning's progress bar writes to stdout and
    a 4-hour bar tee'd to a terminal can stall the training loop behind the
    terminal's renderer (see the logging note in that script). So anything a
    human needs to see *while* the sweep runs has to reach stderr; the log still
    gets it via stdout, which keeps the log a complete record.
    """
    print(text, file=sys.stderr, flush=True)
    print(text, flush=True)


def stage_timings_path(exp_name):
    """
    Where src/training_pipeline.py writes a run's stage timings.

    Also the sweep's record of what has finished: the file is written as the last
    act of a completed pipeline, so its existence -- and nothing else -- is what
    --skip-existing tests. A run that crashed never wrote one and so is retried.
    """
    return Path("logs/timings") / f"{exp_name}.json"


def read_stage_timings(exp_name):
    """
    Per-stage wall-clock times written by src/training_pipeline.py.

    Total subprocess duration lumps setup, training and inference together;
    these are the numbers worth reporting, especially test_s_per_file, which is
    the comparable inference cost across configurations.
    """
    path = stage_timings_path(exp_name)
    if not path.exists():
        return {}
    try:
        with open(path) as handle:
            return json.load(handle)
    except Exception:
        return {}


def result_row(run_num, exp_config, seed, success, duration_s, stages,
               when=None, reconstructed=False):
    """
    One entry of the results JSON.

    Shared by the runs this invocation executed and the runs it skipped, because
    the results file is rewritten from scratch every invocation: without a row
    for each skipped run, a resumed sweep's summary would cover only the tail it
    happened to execute and the final table would not be an ablation table at
    all. The per-run truth always survives in logs/timings/, but reassembling the
    table from there by hand is exactly the work this avoids.
    """
    def minutes(key):
        return round(stages[key] / 60, 2) if key in stages else None

    return {
        "run": run_num,
        "name": exp_config["name"],
        "seed": seed,
        "description": exp_config["description"],
        "success": success,
        "duration_minutes": (round(duration_s / 60, 2)
                             if duration_s is not None else None),
        # Separated stages, for reporting. Absent if the run failed before
        # writing them.
        "setup_data_minutes": minutes("setup_data_s"),
        "train_minutes": minutes("train_s"),
        "train_epochs": stages.get("train_epochs"),
        "inference_minutes": minutes("test_s"),
        "inference_seconds_per_file": (round(stages["test_s_per_file"], 2)
                                       if "test_s_per_file" in stages else None),
        "test_files": stages.get("test_files"),
        # True when the row was rebuilt from an earlier session's timings file
        # rather than measured here. duration_minutes is then the pipeline's own
        # total_s, which excludes interpreter startup, imports and Hydra
        # composition, so it reads a little short of a measured wall clock. The
        # per-stage numbers are unaffected -- they come from the same file either
        # way.
        "reconstructed": reconstructed,
        "timestamp": (when or datetime.now()).isoformat(),
    }


#: Seeds used when --seeds is not given. One entry reproduces the original
#: single-seed ablation exactly.
DEFAULT_SEEDS = [1234]


def seed_overrides(seed):
    """
    Everything that has to change for a run to be an independent replicate.

    `seed` drives seed_everything, so model initialisation, episode sampling and
    dropout. The masker keeps its OWN RandomState (freq_mask_seed), deliberately,
    so that masking is independent of data sampling -- which also means varying
    `seed` alone would leave every replicate with an identical augmentation
    stream and understate the variance of exactly the six masking
    configurations. Both must move together.

    The AFRI and per-class energy profiles are deterministic -- there is no RNG in
    their computation, they take the first afri_n_samples events of each class --
    so they are identical in every replicate and are computed once and cached.
    Nothing is being left unvaried there; there is simply nothing to seed.
    """
    return [f"seed={seed}", f"train_param.freq_mask_seed={seed}"]


def experiment_run_name(exp_config, seed, timestamp, smoke=False):
    """Unique exp_name per (config, seed), which is also the timings file stem."""
    prefix = "smoke_" if smoke else ""
    return f"{prefix}{exp_config['name']}_seed{seed}_{timestamp}"


def experiment_run_dir(exp_config, seed, smoke=False):
    """
    Value for `name`, which configs/log_dir/default.yaml uses as the run
    directory group: logs/experiments/runs/${name}/<date>_<time>.

    Without this every run of a sweep shares one group and is distinguishable
    only by launch timestamp -- 24 anonymous directories. Deliberately excludes
    the sweep timestamp, since Hydra appends its own, so that
    `ls logs/experiments/runs/` reads as the ablation table.
    """
    prefix = "smoke_" if smoke else ""
    return f"{prefix}{exp_config['name']}_seed{seed}"


def run_experiment(run_num, n_runs, exp_config, seed, timestamp, python_cmd,
                   smoke=False, extra_overrides=()):
    """Run a single (configuration, seed) pair, or smoke-test it."""
    announce("\n" + "="*60)
    announce(f"Run {run_num}/{n_runs}: {exp_config['description']}  [seed {seed}]")
    announce(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    announce("="*60)

    exp_name = experiment_run_name(exp_config, seed, timestamp, smoke)
    # extra_overrides go last so they win over everything, which is what a
    # command-line escape hatch is for.
    cmd = (
        [python_cmd, "train.py",
         f"exp_name={exp_name}",
         f"name={experiment_run_dir(exp_config, seed, smoke)}"]
        + exp_config['overrides']
        + seed_overrides(seed)
        + (SMOKE_OVERRIDES if smoke else [])
        + list(extra_overrides)
    )

    print(f"\nCommand: {' '.join(cmd)}\n")

    # Run experiment
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        duration = time.time() - start_time
        announce(f"\n✓ Run {run_num} completed in {duration/60:.1f} minutes")
        stages = read_stage_timings(exp_name)
        if stages:
            parts = []
            for key, label in (("setup_data_s", "data setup"),
                               ("train_s", "train"),
                               ("test_s", "inference")):
                if key in stages:
                    parts.append(f"{label} {stages[key] / 60:.1f} min")
            if "test_s_per_file" in stages:
                parts.append(f"{stages['test_s_per_file']:.1f}s/file")
            if parts:
                print("  stages: " + ", ".join(parts))
        return True, duration, stages
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        announce(f"\n✗ Run {run_num} failed with exit code {e.returncode}")
        # Ask on stderr, and only if there is anybody there to ask.
        #
        # Under run_full_sweep.sh stdout is the log file, so a prompt printed
        # there is invisible: the sweep would sit blocked on input looking for
        # all the world like a hung training run. And under nohup/cron there is
        # no terminal at all, where blocking indefinitely is the worst possible
        # outcome -- it idles the machine for days. Stopping is cheap to undo,
        # because --skip-existing resumes without re-running anything finished.
        if not sys.stdin.isatty():
            announce("stdin is not a terminal, so there is nobody to ask. "
                     "Stopping. Resume with:  --timestamp "
                     f"{timestamp} --skip-existing")
            sys.exit(e.returncode)
        print("Continue? (y/n): ", end="", file=sys.stderr, flush=True)
        response = input()
        if response.lower() != 'y':
            sys.exit(e.returncode)
        return False, duration, {}
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run ablation study experiments")
    parser.add_argument("--pause", type=int, default=300,
                       help="Pause between experiments in seconds (default: 300)")
    parser.add_argument("--no-pause", action="store_true",
                       help="Skip pauses between experiments")
    parser.add_argument("--start-from", type=int, default=1,
                       help="Start from experiment N (1-based)")
    parser.add_argument("--python", default="python",
                       help="Python command (default: python)")
    parser.add_argument("--fast-dev-run", action="store_true",
                       help="Smoke-test every configuration with a single "
                            "train/val batch instead of training it. Minutes, "
                            "not hours; run this before the real ablation.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                       help=f"Run every configuration once per seed, for "
                            f"across-seed variance. Default: {DEFAULT_SEEDS}. "
                            f"Total runs = configs x seeds.")
    parser.add_argument("--timestamp", default=None,
                       help="Reuse a previous sweep's timestamp, so an "
                            "interrupted sweep can be resumed into the same "
                            "namespace (pair with --skip-existing).")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip runs whose stage-timings file already exists. "
                            "Resumes a partially finished sweep.")
    parser.add_argument("--extra-override", action="append", default=[],
                       dest="extra_overrides", metavar="KEY=VALUE",
                       help="Extra Hydra override appended to every run in the "
                            "plan; repeatable. For run-wide settings that are "
                            "not part of the ablation, e.g. "
                            "'++trainer.enable_progress_bar=false'. Prefer the "
                            "'++' form: a plain override fails when the key is "
                            "absent from the composed config, and '+' fails "
                            "when it is present. NOTE these bypass "
                            "tools/sanity_check_configs.py, which validates "
                            "EXPERIMENTS and SMOKE_OVERRIDES but cannot see a "
                            "command-line argument.")

    args = parser.parse_args()

    # Seed-major order: all configurations at seed 1, then all at seed 2, ...
    # If the sweep is cut short you are left with complete ablations at fewer
    # seeds, rather than some configurations having more seeds than others.
    plan = [(exp, seed) for seed in args.seeds for exp in EXPERIMENTS]
    n_runs = len(plan)

    if args.start_from < 1 or args.start_from > n_runs:
        print(f"Error: --start-from must be between 1 and {n_runs}")
        sys.exit(1)
    if len(set(args.seeds)) != len(args.seeds):
        print(f"Error: duplicate seeds in {args.seeds} - replicates would be "
              f"identical, not independent")
        sys.exit(1)
    malformed = [o for o in args.extra_overrides if "=" not in o]
    if malformed:
        print(f"Error: --extra-override takes KEY=VALUE, got: {malformed}")
        sys.exit(1)

    # Setup
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs/ablation_studies")
    log_dir.mkdir(parents=True, exist_ok=True)

    kind = "smoke test" if args.fast_dev_run else "ablation study"
    stem = "smoke" if args.fast_dev_run else "ablation"
    results_file = log_dir / f"{stem}_results_{timestamp}.json"

    print(f"Starting {kind} at {datetime.now()}")
    print(f"Results: {results_file}")
    print(f"Plan: {len(EXPERIMENTS)} configurations x {len(args.seeds)} seed(s) "
          f"= {n_runs} runs, seed-major")
    print(f"Seeds: {args.seeds}  (each sets both seed and freq_mask_seed)")
    if args.extra_overrides:
        print(f"Extra overrides applied to every run: "
              f"{' '.join(args.extra_overrides)}")
    if len(args.seeds) > 1:
        print("      Varies model init, episode sampling, dropout and masking.")
        print("      AFRI/energy profiles are deterministic and cached, so they")
        print("      are shared across seeds and cost nothing to repeat.")
    if args.fast_dev_run:
        print("Mode: --fast-dev-run, one train/val batch per configuration.")
        print("      Exercises the real dataset, profile computation and the")
        print("      tracker -> masker path. Pair it with:")
        print("        python tools/sanity_check_configs.py")
        print("        python tools/verify_masking_wiring.py")
    print()

    # Run experiments
    results = []

    def save_results():
        """
        Rewrite the results file from the rows collected so far.

        Called after every run *and* after every reconstructed skip, because a
        resume that finds everything already done executes nothing -- and would
        otherwise leave the previous session's partial file in place.
        """
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

    for i, (exp, seed) in enumerate(plan, 1):
        if i < args.start_from:
            print(f"Skipping run {i}: {exp['name']} [seed {seed}]")
            continue

        exp_name = experiment_run_name(exp, seed, timestamp, args.fast_dev_run)
        existing = read_stage_timings(exp_name) if args.skip_existing else {}
        if existing:
            print(f"Skipping run {i}: {exp['name']} [seed {seed}] "
                  f"- already has timings")
            # Carry the earlier session's run into this session's results, so a
            # resumed sweep still ends with a complete ablation table. The
            # timings file records no wall-clock time of day, so the row is
            # dated by its mtime -- which is when the pipeline wrote it.
            path = stage_timings_path(exp_name)
            results.append(result_row(
                i, exp, seed,
                # The file is written as the last act of a completed pipeline,
                # so its existence is the success record.
                success=True,
                duration_s=existing.get("total_s"),
                stages=existing,
                when=datetime.fromtimestamp(path.stat().st_mtime),
                reconstructed=True))
            save_results()
            continue

        success, duration, stages = run_experiment(
            i, n_runs, exp, seed, timestamp, args.python,
            smoke=args.fast_dev_run, extra_overrides=args.extra_overrides)

        results.append(result_row(i, exp, seed, success, duration, stages))

        # Save after each experiment
        save_results()

        # Pause (except after the last one). No point resting the machine
        # between one-batch smoke runs.
        if not args.no_pause and not args.fast_dev_run and i < n_runs:
            print(f"\nPausing {args.pause}s for PC rest...")
            try:
                time.sleep(args.pause)
            except KeyboardInterrupt:
                print("Skipping pause...")
    
    # Summary
    print("\n" + "="*78)
    print(f"{kind.upper()} COMPLETE")
    print("="*78)
    successful = sum(1 for r in results if r["success"])
    carried = sum(1 for r in results if r.get("reconstructed"))
    # None where a run failed before its timings were written, and for a
    # reconstructed row whose timings file predates the total_s field.
    total_time = sum(r["duration_minutes"] for r in results
                     if r["duration_minutes"] is not None)
    print(f"Success: {successful}/{len(results)} experiments")
    if carried:
        print(f"         {carried} carried over from earlier sessions of this "
              f"sweep (marked * below)")
    print(f"Total time: {total_time:.1f} minutes")

    # Stage breakdown, for reporting. Setup, training and inference are
    # separated because total wall clock conflates three different costs:
    # profile computation, optimisation, and per-recording inference.
    if any(r.get("train_minutes") is not None
           or r.get("inference_minutes") is not None for r in results):
        print()
        print(f"{'configuration':22s} {'seed':>6s} {'setup':>8s} {'train':>9s} "
              f"{'epochs':>7s} {'infer':>9s} {'s/file':>8s}")
        print("-" * 78)

        def fmt(value, width, digits=1):
            return (format(value, f"{width}.{digits}f") if value is not None
                    else "-".rjust(width))

        for r in results:
            label = r["name"] + (" *" if r.get("reconstructed") else "")
            print(f"{label:22s} {r.get('seed', '-'):>6} "
                  f"{fmt(r.get('setup_data_minutes'), 8)} "
                  f"{fmt(r.get('train_minutes'), 9)} "
                  f"{str(r.get('train_epochs') or '-'):>7s} "
                  f"{fmt(r.get('inference_minutes'), 9)} "
                  f"{fmt(r.get('inference_seconds_per_file'), 8)}")
        print("-" * 78)
        print("minutes, except the last column. Per-file inference is the "
              "comparable cost")
        print("across configurations: one test batch is one whole recording.")
        if carried:
            print("* carried over from an earlier session of this sweep. The "
                  "per-stage numbers")
            print("  are the same file this table always reads; only the total "
                  "excludes process")
            print("  startup, imports and Hydra composition, so it reads "
                  "slightly short.")

    # Across-seed spread, when there is more than one seed. Reported as
    # mean +/- sd with n stated: with a handful of seeds the sd is itself a
    # noisy estimate, so a normal-theory confidence interval would overstate
    # what these runs support. Score-level CIs come from the evaluation
    # notebooks; this is wall-clock only.
    seeds_present = sorted({r.get("seed") for r in results
                            if r.get("seed") is not None})
    if len(seeds_present) > 1:
        import statistics
        print()
        print(f"Across-seed spread over {len(seeds_present)} seeds "
              f"{seeds_present} (wall clock, mean +/- sd):")
        print(f"{'configuration':22s} {'train (min)':>18s} {'infer (min)':>18s} "
              f"{'epochs':>14s}")
        print("-" * 78)
        for name in [e["name"] for e in EXPERIMENTS]:
            rows = [r for r in results if r["name"] == name and r["success"]]
            if not rows:
                continue

            def spread(key):
                values = [r[key] for r in rows if r.get(key) is not None]
                if not values:
                    return "-".rjust(18)
                if len(values) == 1:
                    return f"{values[0]:.1f} (n=1)".rjust(18)
                return (f"{statistics.mean(values):.1f} "
                        f"+/- {statistics.stdev(values):.1f}").rjust(18)

            epochs = [r["train_epochs"] for r in rows
                      if r.get("train_epochs") is not None]
            epoch_text = ("-" if not epochs else
                          f"{statistics.mean(epochs):.1f}" if len(epochs) == 1 else
                          f"{statistics.mean(epochs):.1f} +/- "
                          f"{statistics.stdev(epochs):.1f}")
            print(f"{name:22s} {spread('train_minutes')} "
                  f"{spread('inference_minutes')} {epoch_text:>14s}")
        print("-" * 78)
        incomplete = [name for name in [e["name"] for e in EXPERIMENTS]
                      if len({r["seed"] for r in results
                              if r["name"] == name and r["success"]})
                      < len(seeds_present)]
        if incomplete:
            print(f"⚠️  fewer seeds than expected for: {', '.join(incomplete)}")
            print("    Re-run with --timestamp <this sweep> --skip-existing "
                  "to fill the gaps.")

    print(f"\nResults: {results_file}")
    print("Per-stage JSON: logs/timings/<exp_name>.json")
    if len(seeds_present) > 1:
        print(f"Seeds completed: {seeds_present}")

if __name__ == "__main__":
    main()
