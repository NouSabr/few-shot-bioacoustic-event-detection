#!/usr/bin/env python
"""
Sanity-check every ablation configuration before launching the runs.

Reads the experiment list straight from run_ablation_experiments.py, applies each
one's Hydra overrides to configs/train.yaml, and checks that the resulting config
actually produces the intended masking. Cheap: no audio, no model, no GPU, a
couple of seconds for all eight configurations.

Eight runs at several hours each is a long time to discover that a config selected
the wrong strategy, so this exists to catch the failure modes that have actually
happened in this project:

  * an override key that nothing reads (freq_mask_afri_weight, freq_mask_ema_alpha,
    afri_min_bin_expand, and num_workers all looked wired and were not)
  * a mode that silently falls back to a different strategy
  * two configurations that are secretly the same experiment (AFRI direct and
    inverse were nearly identical; Phase 3 was effectively random)
  * a per-class strategy whose profiles never get attached, collapsing it to
    unguided masking

What it cannot check: anything requiring the dataset. Profile *computation* needs
audio, so this attaches stub profiles to prove the plumbing and the steering are
right. Run tools/verify_masking_wiring.py alongside it for the measured behaviour.

Usage:
    python tools/sanity_check_configs.py
    python tools/sanity_check_configs.py --only afri_direct phase2_energy
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import yaml

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _REPO)


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_REPO, relpath))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Loaded by path so this runs with numpy + pyyaml alone.
_fm = _load("frequency_masker",
            "src/datamodules/components/frequency_masker.py")
_abl = _load("run_ablation_experiments", "run_ablation_experiments.py")

N_MELS = 128

# What each experiment name is supposed to be testing. Kept here, separate from
# the config, so a config edit cannot quietly redefine the experiment.
EXPECTED = {
    'baseline':          dict(masker=None, cife=False),
    'cife':              dict(masker=None, cife=True),
    'phase1_random':     dict(masker='RandomFrequencyMasker', cife=False),
    'slot_uniform':      dict(masker='UniformSlotFrequencyMasker', cife=False),
    'phase2_energy':     dict(masker='EnergyBasedFrequencyMasker', cife=False),
    'phase3_activation': dict(masker='ActivationGuidedFrequencyMasker', cife=False),
    'afri_direct':       dict(masker='AFRIRangeMasker', cife=False,
                              afri_mode='direct'),
    'afri_inverse':      dict(masker='AFRIRangeMasker', cife=False,
                              afri_mode='inverse'),
}

# Keys any config may set. Anything outside this is either a typo or a key that
# nothing reads.
CONSUMED_KEYS = {
    'features.use_cife',
    'train_param.use_freq_masking', 'train_param.freq_mask_mode',
    'train_param.freq_mask_n_bands', 'train_param.freq_mask_prob',
    'train_param.freq_mask_max_width', 'train_param.freq_mask_seed',
    'train_param.freq_mask_energy_weight',
    'train_param.freq_mask_importance_weight',
    'train_param.freq_mask_importance_ema_alpha',
    'train_param.freq_mask_afri_area_budget',
    'train_param.use_afri', 'train_param.afri_mode',
    'train_param.afri_n_samples', 'train_param.afri_value_threshold',
    'train_param.afri_search_threshold', 'train_param.afri_debris_size',
    'train_param.afri_cache_enabled', 'train_param.afri_cache_dir',
    'train_param.afri_force_recompute', 'train_param.afri_min_samples',
    # Added per run by run_ablation_experiments.seed_overrides()
    'seed',
}

FAILURES = []


def fail(config_name, message):
    FAILURES.append(f"{config_name}: {message}")
    print(f"    FAIL  {message}")


def ok(message):
    print(f"    ok    {message}")


# ----------------------------------------------------------------------
# Config resolution
# ----------------------------------------------------------------------

def load_train_yaml():
    """
    Load configs/train.yaml as plain dicts.

    Deliberately not OmegaConf: the file interpolates ${original_work_dir}, which
    only Hydra's runtime can resolve. The masking subtrees need no interpolation.
    """
    with open(os.path.join(_REPO, "configs", "train.yaml")) as handle:
        return yaml.safe_load(handle)


def resolve_default_groups(cfg):
    """
    Map each config group to the file train.yaml's defaults list selects for it,
    e.g. {'trainer': 'default.yaml', 'logger': 'wandb.yaml'}.

    Needed because a key like `trainer.fast_dev_run` lives in
    configs/trainer/default.yaml, not in train.yaml, so it cannot be found by
    walking train.yaml alone.
    """
    groups = {}
    for entry in cfg.get("defaults") or []:
        if not isinstance(entry, dict):
            continue                      # '_self_', 'override hydra/...'
        for key, value in entry.items():
            if key.startswith("override ") or value is None:
                continue                  # not a group we can resolve here
            groups[key.replace("optional ", "")] = value
    return groups


def load_group_config(group, filename):
    """Load configs/<group>/<filename>, or None if it is not there."""
    if not filename.endswith(".yaml"):
        filename += ".yaml"
    path = os.path.join(_REPO, "configs", group, filename)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def key_exists(tree, dotted):
    """Walk a dotted path through nested dicts."""
    node = tree
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def check_smoke_overrides(cfg):
    """
    The --fast-dev-run overrides must be ones Hydra will actually accept.

    Hydra runs in struct mode, so `key=value` fails outright when `key` is not
    already in the composed config -- "Could not override 'trainer.fast_dev_run'".
    That is what happened: configs/trainer/default.yaml does not define
    fast_dev_run, so every smoke run died before doing any work. Appending '++'
    (append-or-override) is the fix, and is what tests/shell/ has always used.

    This check exists because the smoke overrides bypass apply_overrides() -- they
    are not per-experiment keys -- and so escaped the per-config validation
    entirely. A whole class of override error was going unchecked.
    """
    print("\n[smoke] --fast-dev-run overrides must be ones Hydra accepts")
    groups = resolve_default_groups(cfg)

    for override in _abl.SMOKE_OVERRIDES:
        key, _, value = override.partition("=")
        appends = key.startswith("+")
        key = key.lstrip("+")

        # No dot: either a top-level key of train.yaml, or a config-group choice
        # like `logger=tensorboard`.
        if "." not in key:
            if key in cfg:
                ok(f"{override}  (top-level key, present in train.yaml)")
            elif os.path.isdir(os.path.join(_REPO, "configs", key)):
                if load_group_config(key, value) is None:
                    fail("smoke", f"{override} selects configs/{key}/{value}.yaml, "
                                  f"which does not exist")
                else:
                    ok(f"{override}  (config group, configs/{key}/{value}.yaml exists)")
            else:
                fail("smoke", f"{override} names neither a train.yaml key nor a "
                              f"config group")
            continue

        # Dotted: find the key in train.yaml, or in the group file the defaults
        # list selects for its first segment.
        group, _, rest = key.partition(".")
        present = key_exists(cfg, key)
        where = "train.yaml"
        if not present and group in groups:
            group_cfg = load_group_config(group, groups[group])
            if group_cfg is not None:
                present = key_exists(group_cfg, rest)
                where = f"configs/{group}/{groups[group]}"

        if present and appends:
            ok(f"{override}  ('++' is safe; {key} is set in {where})")
        elif present:
            ok(f"{override}  ({key} exists in {where})")
        elif appends:
            ok(f"{override}  ({key} is absent, which is why '++' is required)")
        else:
            fail("smoke", f"{override} overrides {key!r}, which no composed config "
                          f"defines - Hydra will refuse it in struct mode. "
                          f"Write it as '++{override}'.")


def apply_overrides(cfg, overrides, config_name):
    """Apply Hydra-style key=value overrides, reporting keys that do not exist."""
    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in cfg.items()}
    for override in overrides:
        key, _, raw = override.partition("=")
        value = yaml.safe_load(raw)

        if key not in CONSUMED_KEYS:
            fail(config_name, f"override sets {key!r}, which nothing reads")

        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                fail(config_name, f"override key {key!r} does not exist in train.yaml")
                node = None
                break
            node = node[part]
        if node is None:
            continue
        if parts[-1] not in node:
            fail(config_name, f"override key {key!r} does not exist in train.yaml")
            continue
        node[parts[-1]] = value
    return cfg


# ----------------------------------------------------------------------
# Stub profiles, standing in for what build_meta() attaches
# ----------------------------------------------------------------------

SLOT_W = 24
SLOTS = [(i * SLOT_W, min((i + 1) * SLOT_W, N_MELS))
         for i in range(int(np.ceil(N_MELS / SLOT_W)))]
STUB_CLASSES = {f"cls{i}": i for i in range(len(SLOTS))}


def stub_energy_profiles():
    profiles = {}
    for name, slot in STUB_CLASSES.items():
        v = np.full(N_MELS, 0.05, dtype=np.float32)
        a, b = SLOTS[slot]
        v[a:b] = 1.0
        profiles[name] = v
    return profiles


def stub_afri_profiles():
    profiles = {}
    for name, slot in STUB_CLASSES.items():
        a, b = SLOTS[slot]
        weights = np.full(N_MELS, 0.1, dtype=np.float32)
        weights[a:b] = 0.9
        profiles[name] = {
            'mask_weights': weights, 'inverse_weights': 1.0 - weights,
            'auto_minbin': a, 'auto_maxbin': b, 'valid': True,
        }
    return profiles


class StubTracker:
    def __init__(self):
        self.profiles = {}
        for name, slot in STUB_CLASSES.items():
            v = np.full(N_MELS, 0.05)
            a, b = SLOTS[slot]
            v[a:b] = 1.0
            self.profiles[name] = v

    def get_importance(self, class_name=None):
        return self.profiles.get(class_name)

    def get_stats(self):
        return {'n_classes_tracked': len(self.profiles), 'total_updates': 1,
                'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 1.0,
                'mean_spread': 0.95}


def attach_stubs(masker, mode):
    """Attach whatever the dataset would attach after build_meta()."""
    if mode == 'energy':
        masker.energy_profiles = stub_energy_profiles()
    elif mode == 'afri':
        masker.afri_profiles = stub_afri_profiles()
    elif mode == 'activation':
        masker.importance_tracker = StubTracker()


# ----------------------------------------------------------------------
# Per-config checks
# ----------------------------------------------------------------------

def spec(seed=0, t=30):
    return np.random.RandomState(seed).rand(t, N_MELS) + 1.0


def mask_stream(masker, class_name, n=6, seed=3):
    return [tuple(np.flatnonzero((masker(spec(seed + i), class_name) == 0.0)
                                 .all(axis=0)))
            for i in range(n)]


def check_config(experiment, cfg_template):
    name = experiment["name"]
    print(f"\n[{name}]  {experiment['description']}")

    cfg = apply_overrides(cfg_template, experiment["overrides"], name)
    train_param = cfg["train_param"]
    expected = EXPECTED.get(name)
    if expected is None:
        fail(name, "no expectation recorded in EXPECTED - is this a new config?")
        return None

    # CIFE
    if bool(cfg["features"]["use_cife"]) != expected["cife"]:
        fail(name, f"use_cife is {cfg['features']['use_cife']}, "
                   f"expected {expected['cife']}")
    else:
        ok(f"use_cife = {expected['cife']}")

    # The factory must select the class this experiment is named after
    try:
        masker = _fm.build_frequency_masker(train_param, N_MELS)
    except ValueError as exc:
        fail(name, f"build_frequency_masker rejected the config: {exc}")
        return None

    actual = type(masker).__name__ if masker is not None else None
    if actual != expected["masker"]:
        fail(name, f"mode {train_param.get('freq_mask_mode')!r} built "
                   f"{actual}, expected {expected['masker']}")
        return None
    ok(f"masker = {actual or 'none (masking disabled)'}")

    if masker is None:
        if train_param.get("use_freq_masking", False):
            fail(name, "use_freq_masking is true but no masker was built")
        return None

    mode = train_param.get("freq_mask_mode", "random")

    # Common parameters must survive from the override to the object
    for key, attr in [("freq_mask_n_bands", "n_bands"),
                      ("freq_mask_prob", "mask_prob"),
                      ("freq_mask_max_width", "max_band_width"),
                      ("freq_mask_seed", "seed")]:
        want, got = train_param.get(key), getattr(masker, attr)
        if want != got:
            fail(name, f"{key}={want} but masker.{attr}={got}")
    ok(f"n_bands={masker.n_bands} prob={masker.mask_prob} "
       f"max_width={masker.max_band_width} seed={masker.seed}")

    # AFRI mode, and the switch it depends on
    if mode == 'afri':
        if masker.afri_mode != expected["afri_mode"]:
            fail(name, f"afri_mode is {masker.afri_mode!r}, "
                       f"expected {expected['afri_mode']!r}")
        else:
            ok(f"afri_mode = {masker.afri_mode}")
        if not train_param.get("use_afri", False):
            fail(name, "freq_mask_mode=afri requires use_afri=true, else every "
                       "class falls back to unguided masking")
        else:
            ok("use_afri = True")

    # Per-class strategies must expose the attribute the dataset attaches to,
    # and must fall back to unguided masking until it is populated.
    attribute = _fm.PROFILE_ATTRIBUTE_FOR_MODE[mode]
    if attribute is not None:
        if not hasattr(masker, attribute):
            fail(name, f"per-class mode {mode!r} has no {attribute!r} attribute, "
                       f"so the dataset cannot attach its profiles")
        else:
            ok(f"profiles attach via .{attribute}")
        before = mask_stream(masker, "cls2")
        attach_stubs(masker, mode)
        masker.rng = np.random.RandomState(masker.seed)
        after = mask_stream(masker, "cls2")
        if before == after:
            fail(name, f"attaching profiles changed nothing - {mode!r} is not "
                       f"actually using them")
        else:
            ok("attached profiles change the mask (guidance is live)")
    else:
        ok("no per-class profiles needed")

    # It must actually mask, and honour mask_prob
    masker.rng = np.random.RandomState(masker.seed)
    sample = spec(1)
    out = masker(sample, "cls2")
    if out.shape != sample.shape:
        fail(name, f"output shape {out.shape} != input {sample.shape}")
    if not np.any(out == 0.0):
        fail(name, "masked nothing at all")
    if np.array_equal(out, sample) and masker.mask_prob > 0:
        fail(name, "output is identical to the input")
    n_masked = [int((masker(spec(i), "cls2") == 0.0).all(axis=0).sum())
                for i in range(300)]
    rate = np.mean([n > 0 for n in n_masked])
    if abs(rate - masker.mask_prob) > 0.08:
        fail(name, f"masked {rate:.3f} of calls, expected ~{masker.mask_prob}")
    else:
        ok(f"masks {rate:.3f} of calls, mean {np.mean([n for n in n_masked if n]):.1f} "
           f"bins ({np.mean([n for n in n_masked if n]) / N_MELS:.0%} of spectrum)")
    if max(n_masked) >= N_MELS:
        fail(name, "some call masked the entire spectrum")

    # Reproducible from the seed
    a = _fm.build_frequency_masker(train_param, N_MELS)
    b = _fm.build_frequency_masker(train_param, N_MELS)
    attach_stubs(a, mode)
    attach_stubs(b, mode)
    if mask_stream(a, "cls2") != mask_stream(b, "cls2"):
        fail(name, "not reproducible from the configured seed")
    else:
        ok("reproducible from the seed")

    fresh = _fm.build_frequency_masker(train_param, N_MELS)
    attach_stubs(fresh, mode)
    return mask_stream(fresh, "cls2", n=12)


def check_seeds_are_wired(cfg_template):
    """
    Replicates must actually differ.

    A multi-seed sweep only measures variance if every source of run-to-run
    randomness moves with the seed. The masker holds its own RandomState
    (freq_mask_seed) separate from the global seed, so if the sweep set only
    `seed` then every replicate of a masking configuration would see an
    IDENTICAL augmentation stream and the reported spread would be too small.
    This checks that the overrides the runner actually applies do change the
    masking, and that a repeated seed reproduces exactly.
    """
    print("\n[seeds] replicates must differ, and repeat exactly")
    for experiment in _abl.EXPERIMENTS:
        name = experiment["name"]
        streams = {}
        for seed in (1234, 2025):
            overrides = experiment["overrides"] + _abl.seed_overrides(seed)
            cfg = apply_overrides(cfg_template, overrides, name)
            masker = _fm.build_frequency_masker(cfg["train_param"], N_MELS)
            if masker is None:
                break
            attach_stubs(masker, cfg["train_param"].get("freq_mask_mode"))
            streams[seed] = mask_stream(masker, "cls2", n=10)
        if len(streams) < 2:
            continue                      # no masker: nothing seed-dependent here
        if streams[1234] == streams[2025]:
            fail(name, "different seeds produce an IDENTICAL mask stream - "
                       "replicates would not be independent")
        else:
            ok(f"{name}: seeds 1234 and 2025 give different masks")

        again = apply_overrides(cfg_template,
                                experiment["overrides"] + _abl.seed_overrides(1234),
                                name)
        repeat = _fm.build_frequency_masker(again["train_param"], N_MELS)
        attach_stubs(repeat, again["train_param"].get("freq_mask_mode"))
        if mask_stream(repeat, "cls2", n=10) != streams[1234]:
            fail(name, "the same seed did not reproduce the mask stream")


def check_configs_differ(streams):
    """No two masking configurations may be the same experiment."""
    print("\n[pairwise] every masking config must behave differently")
    names = sorted(streams)
    duplicates = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if streams[a] == streams[b]:
                duplicates.append((a, b))
    if duplicates:
        for a, b in duplicates:
            FAILURES.append(f"{a}/{b}: identical mask streams")
            print(f"    FAIL  {a} and {b} produce identical masks")
    else:
        print(f"    ok    all {len(names)} masking configs distinct: "
              f"{', '.join(names)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None,
                        help="check only these experiment names")
    args = parser.parse_args()

    cfg_template = load_train_yaml()
    experiments = [e for e in _abl.EXPERIMENTS
                   if args.only is None or e["name"] in args.only]

    print("=" * 74)
    print(f"Sanity-checking {len(experiments)} of {len(_abl.EXPERIMENTS)} "
          f"ablation configurations")
    print("=" * 74)

    streams = {}
    for experiment in experiments:
        stream = check_config(experiment, cfg_template)
        if stream is not None:
            streams[experiment["name"]] = stream

    if len(streams) > 1:
        check_configs_differ(streams)
    if args.only is None:
        check_seeds_are_wired(cfg_template)
        check_smoke_overrides(cfg_template)

    # Every experiment in EXPECTED should exist, so a deleted config is noticed
    if args.only is None:
        missing = set(EXPECTED) - {e["name"] for e in _abl.EXPERIMENTS}
        if missing:
            FAILURES.append(f"configs missing from the ablation: {sorted(missing)}")
            print(f"\n    FAIL  expected configs absent from "
                  f"run_ablation_experiments.py: {sorted(missing)}")

    print()
    print("=" * 74)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} problem(s)")
        for failure in FAILURES:
            print(f"  - {failure}")
    else:
        print("RESULT: all configurations sane")
    print("=" * 74)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
