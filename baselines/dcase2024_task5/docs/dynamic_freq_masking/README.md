# Dynamic Frequency Masking

Documentation for the frequency-masking augmentation used in the DCASE 2024
Task 5 few-shot bioacoustic system.

This directory previously held a forward-looking implementation plan, with
predicted outcomes ("+5.9% F1"), per-phase durations and decision points. The
work is finished and the measured results did not match those predictions, so the
planning material has been removed. What remains describes the implemented
system.

---

## Read these

| Document | Contents |
|---|---|
| **[01_OVERVIEW.md](01_OVERVIEW.md)** | The masking strategies, how masking reaches the data, quick start. **Start here.** |
| **[05_HYPERPARAMETERS.md](05_HYPERPARAMETERS.md)** | Every config key, what reads it, and the keys that were removed. |
| **[08_AFRI_INTEGRATION.md](08_AFRI_INTEGRATION.md)** | AFRI range masking: the algorithm, why it was rewritten, edge cases. |
| **[09_WIRING_VERIFICATION.md](09_WIRING_VERIFICATION.md)** | Recorded masking checks, comparability caveats, and known open issues. **Read before interpreting the ablation.** |

The pre-split design guides have been retired. Use the current references
above for implementation details rather than the old class/configuration designs.

---

## Code

| Path | Role |
|---|---|
| [`src/datamodules/components/frequency_masker.py`](../../src/datamodules/components/frequency_masker.py) | Implemented masking strategies |
| [`src/datamodules/components/dynamic_pcen_dataset.py`](../../src/datamodules/components/dynamic_pcen_dataset.py) | Strategy factory, AFRI profile computation, mask application |
| [`src/utils/frequency_importance_tracker.py`](../../src/utils/frequency_importance_tracker.py) | Encoder-activation importance (activation mode) |
| [`src/utils/afri_core.py`](../../src/utils/afri_core.py) | AFRI range identification |
| [`src/utils/afri_cache.py`](../../src/utils/afri_cache.py) | Profile caching |
| [`src/callbacks/frequency_masking_callback.py`](../../src/callbacks/frequency_masking_callback.py) | Logging |
| [`run_ablation_experiments.py`](../../run_ablation_experiments.py) | The eight-configuration ablation |

## Verify

Three layers, cheapest first. Run all three before launching the ablation.

```bash
# 1. Are the configs sane?  seconds, no data needed
python tools/sanity_check_configs.py

# 2. Does each strategy steer the mask as intended?  seconds, no data needed
python tools/verify_masking_wiring.py

# 3. Unit tests
pytest tests/unit/test_phase1_frequency_masking.py \
       tests/unit/test_phase2_frequency_masking.py \
       tests/unit/test_phase3_frequency_masking.py \
       tests/unit/test_afri_frequency_masking.py -v

# 4. Does each config actually run?  minutes, needs the data
python run_ablation_experiments.py --fast-dev-run
```

**`tools/sanity_check_configs.py`** reads the experiment list from
`run_ablation_experiments.py`, applies each set of Hydra overrides, and checks
that the config produces the intended masking: the mode builds the expected
class, every override key is one something reads, per-class profiles have an
attachment point and change the mask once attached, `afri` has `use_afri=true`,
and — the check that matters most — **no two masking configs behave
identically**. Eight runs is a long time to discover otherwise.

It also validates `SMOKE_OVERRIDES`, the block that turns each configuration
into a one-batch run. That check was added after those overrides broke stage 2 of
a sweep with `Could not override 'trainer.fast_dev_run'`: they are not
per-experiment keys, so they never passed through `apply_overrides()` and escaped
the per-config validation entirely — a whole class of override error was going
unchecked. **Hydra override syntax is the trap.** Under struct mode:

| form | requires | use when |
|---|---|---|
| `key=value` | key already exists | it is a real config key |
| `+key=value` | key must NOT exist | almost never — it fails once someone adds the key |
| `++key=value` | either | **prefer this** for keys the YAML does not define |

`configs/trainer/default.yaml` defines neither `fast_dev_run` nor
`enable_progress_bar` (the latter is present only as a comment), so both need
`++`. The checker now classifies every entry as a top-level key, a config-group
selection (`logger=tensorboard` — a group, so no `+`), or a dotted key resolved
against `train.yaml` and then against the group file the defaults list selects,
and prints the exact remedy when a plain override targets an absent key.

**`tools/verify_masking_wiring.py`** measures per-bin mask rates for all five
strategies and asserts them against recorded reference values, including that
each guided strategy gives different classes different masks. Needs only numpy
and `.afri_cache/profiles.npz` — no model, dataset or GPU.

**`--fast-dev-run`** runs one train/val batch per configuration, which covers
what the static checks cannot reach: real dataset construction, energy/AFRI
profile computation or cache load, the tracker → `training_step` → masker path,
HDF5 reads, collate and the forward pass.

Stage timings (setup, training, inference, and seconds per test recording) are
written to `logs/timings/<exp_name>.json` and tabulated at the end of an ablation
run.

---

## Running the sweep end to end

`run_full_sweep.sh` gates a multi-day commitment behind two cheap stages:
pre-flight checks (seconds, no data), a one-batch smoke test of every
configuration (minutes, real data), then the sweep. It is resumable — an
interruption prints the exact command to pick up where it stopped, and finished
runs are skipped by their stage-timings file rather than re-run.

### Where the output goes

Four destinations, each with one job. Nothing here is the single point of failure
for a result.

| destination | written by | contents |
|---|---|---|
| terminal | this script | stage banners, per-run milestones, **stderr only** from runs |
| `logs/sweeps/sweep_<id>.log` | this script | the above, plus each run's raw stdout |
| `<run dir>/stdout.log` | [`train.py`](../../train.py)'s `_Tee` | that run's raw stdout |
| `<run dir>/train.log` | Hydra's `job_logging` file handler | that run's `log.*` records |

where `<run dir>` is `logs/experiments/runs/<config>_seed<seed>/<date>_<time>/`.

Two deliberate choices, both performance fixes rather than preferences:

**A run's stdout is appended straight to the sweep log, never through the tee.**
`TQDMProgressBar` writes to stdout, and `train.py` wraps `sys.stdout` in a `_Tee`
whose `__getattr__` proxies `isatty()` to the real stdout — so under the sweep
script the bar asks a *pipe* whether it is a terminal, gets no, and stops
redrawing in place. Symptom: one fresh line per batch instead of one updating
line. Worse, every refresh then has to be consumed by the terminal emulator
before the writer unblocks; a four-hour run refreshes on the order of 10^5 times,
and at `num_workers=0` there is no prefetching worker to hide the stall behind,
so a slow terminal can throttle the training loop itself.

**The progress bar is off by default** (`++trainer.enable_progress_bar=false`).
`_Tee` means stdout is written twice — once to the run's `stdout.log`, once to the
sweep log. That is nothing at a few KB and tens of MB of carriage-return spam on
one unreadable line at 10^5 refreshes. `--progress-bar` restores it when you want
something to `tail -f`.

Anything a human needs *while* the sweep runs goes to **stderr**, which stays on
the terminal. `run_ablation_experiments.announce()` writes to both, so the log
remains a complete record.

**Results never depend on any of this.** `Best Result` and the per-alpha metrics
are `self.log()`-ed to the configured logger and written to
`<run dir>/<alpha>/Eval_raw.csv`; `Testing finished in X min (Ys per file)` and
every other `log.*` line lands in `train.log`; stage timings go to
`logs/timings/<exp_name>.json`. The redirect changes only where you *watch*
results, never whether they are recorded. `Best Result` is additionally echoed to
stderr so the headline number of a four-hour run still reaches the terminal.

### Resuming keeps the results file complete

`logs/ablation_studies/ablation_results_<id>.json` is rewritten from scratch on
every invocation, so a resumed session would otherwise leave a file describing
only the runs it happened to execute — and the end-of-sweep stage-breakdown
table, the thing you actually read the ablation off, would be missing every
configuration finished before the interruption. So a skipped run is not dropped:
its row is rebuilt from `logs/timings/<exp_name>.json` and marked
`"reconstructed": true` (`*` in the printed table).

Rebuilt rows are recorded as successful, because the timings file is written as
the last act of a completed pipeline — a crashed run has none, which is also
exactly why it gets retried. Their per-stage numbers come from the same file the
table always reads, so they are not approximations. Only two fields differ:
`duration_minutes` is the pipeline's own `total_s` rather than measured
subprocess wall clock, so it excludes interpreter startup, imports and Hydra
composition and reads a few seconds short; and `timestamp` is the timings file's
mtime, since the file records no time of day. Hence the flag — the distinction is
visible rather than silently mixed into the table.

### Failures are not allowed to hang the machine

On a failed run the runner asks whether to continue — on stderr, since a prompt
written to the log is invisible and the sweep would look hung rather than blocked
on input. When stdin is not a terminal (`nohup`, `cron`) it does not ask at all:
it stops and prints the resume command. Blocking indefinitely there idles the
machine for days, whereas `--skip-existing` makes resuming free.

### Overrides outside the ablation

`--extra-override KEY=VALUE` (repeatable) appends a Hydra override to every run
in the plan — for run-wide settings that are not part of the ablation, which is
how the progress-bar default is implemented. Prefer the `++` form, per the table
above. These bypass `tools/sanity_check_configs.py`, which cannot see a
command-line argument; only the `KEY=VALUE` shape is checked.

---

## Multi-seed sweep

`configs/train.yaml` ships one seed. To get across-seed variance, pass several:

```bash
# the original single-seed ablation (8 runs)
python run_ablation_experiments.py

# three seeds (24 runs), seed-major
python run_ablation_experiments.py --seeds 1234 2025 7 --no-pause
```

**Both seeds move together.** Each run sets `seed` *and*
`train_param.freq_mask_seed`. `seed` drives `seed_everything`, so model
initialisation, episode sampling and dropout; the masker deliberately keeps its
own RandomState, so varying `seed` alone would leave every replicate with an
**identical augmentation stream** and understate the variance of exactly the six
masking configurations. `tools/sanity_check_configs.py` asserts that two seeds
produce different mask streams, and that a repeated seed reproduces exactly.

**Seed-major order.** All configurations at seed 1, then all at seed 2, and so
on. If the sweep is cut short you are left with complete ablations at fewer
seeds, rather than some configurations having more seeds than others.

**Resuming.** A 24-run sweep is days of compute. Reuse the sweep's namespace and
skip what already finished:

```bash
python run_ablation_experiments.py --seeds 1234 2025 7 \
    --timestamp 20260916_143000 --skip-existing
```

`--skip-existing` skips any run whose `logs/timings/<exp_name>.json` exists, and
carries it into the results file rather than dropping it (see
[Resuming keeps the results file complete](#resuming-keeps-the-results-file-complete)),
so the summary covers the whole sweep and not just this session. The final
summary warns if a configuration ended up with fewer seeds than the others.

Forgetting `--timestamp` is the one way to lose work: `exp_name` is
`<config>_seed<seed>_<timestamp>`, so a fresh timestamp is a fresh namespace in
which nothing has finished, and all 24 runs start over without complaint.
`run_full_sweep.sh --resume <id>` passes it for you.

### What the seed variance covers

The seeds vary everything in the pipeline that is actually random: model
initialisation, episode sampling, dropout, and the masking stream.

The AFRI and per-class energy profiles are **deterministic** — there is no RNG in
their computation at all. Both take the first `afri_n_samples` events of each
class (indices `0..n-1` of `meta[cls]["info"]`), which is deliberate: those stand
for the five annotated support examples available at deployment, and that framing
is the point of AFRI. So the profiles are identical in every replicate not
because a seed was omitted, but because there is nothing there to seed. They are
also computed once and cached, so the extra seeds cost only training and
inference.

If you wanted to know how sensitive the results are to *which* five segments
define each profile, that is a separate experiment — deliberately varying the
subset — and no seed would ever have reached it.

Seed spread is distinct from the bootstrap confidence intervals computed over
evaluation files: those describe uncertainty from which recordings are scored,
whereas seed spread describes uncertainty from how the model was trained. They
are not interchangeable, and neither subsumes the other.

One genuine determinism gap, unrelated to seeds: `get_all_csv_files` uses
`os.walk` + `glob` without sorting, so the order of `meta[cls]["info"]` follows
filesystem enumeration order. Within one machine and an unchanged data directory
that is stable, and every run of a sweep sees the same profiles. Across machines
it is not guaranteed, so "the first five events" could resolve differently.
Sorting the enumeration would close this, but note the cache key hashes the CSV
list *sorted* already — so sorting the walk could change profile **content**
without changing the **key**, silently reusing a stale cache. Clear
`.afri_cache/` if you make that change.

With a handful of seeds the standard deviation is itself a noisy estimate, so the
runner reports **mean ± sd with n stated** rather than a normal-theory confidence
interval, which would overstate what three runs support.
