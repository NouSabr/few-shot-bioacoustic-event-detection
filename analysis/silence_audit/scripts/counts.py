"""Reconstruct paper audit quantities from saved matched counts, not audio."""
from __future__ import annotations

from collections import defaultdict
import csv
from fractions import Fraction
from pathlib import Path

SUBSETS = ('HB', 'ME', 'PB', 'PB24', 'PW', 'RD')
EPS = Fraction(1, 100000)


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def integer(value):
    number = Fraction(str(value))
    if number.denominator != 1:
        raise ValueError(f'Expected an integer count, got {value!r}')
    return int(number)


def f1(counts):
    tp, fp, fn = counts
    if min(counts) < 0:
        raise ValueError(f'Negative event count: {counts}')
    if not tp:
        return EPS
    value = Fraction(2 * tp, 2 * tp + fp + fn)
    # The archived data do not enter the positive-score floor edge case.
    if value <= EPS:
        raise ValueError('Positive F1 enters the evaluator/helper floor ambiguity')
    return value


def harmonic(values):
    values = tuple(values)
    return Fraction(len(values), 1) / sum(Fraction(1, 1) / v for v in values)


def load_effects(paths):
    keyed = {}
    for path in paths:
        for raw in read_csv(path):
            row = dict(raw)
            row['seed'] = integer(row['seed'])
            for column in ('file_tp', 'file_fp', 'file_fn', 'silenced_file_fn', 'other_tp', 'other_fp', 'other_fn'):
                row[column] = integer(row[column])
            if row['originally_silent'] not in ('True', 'False'):
                raise ValueError('Unexpected originally_silent encoding')
            row['originally_silent'] = row['originally_silent'] == 'True'
            key = tuple(row[c] for c in ('configuration', 'seed', 'evaluator_variant', 'target_filename', 'subset'))
            if key in keyed:
                raise ValueError(f'Duplicate subset-effect key: {key}')
            keyed[key] = row
    return keyed


def reconstruct(effects):
    runs = defaultdict(lambda: defaultdict(dict))
    targets = defaultdict(dict)
    for (configuration, seed, variant, name, subset), row in effects.items():
        runs[configuration, seed, variant][subset][name] = row
        targets[configuration, seed, variant, name][subset] = row
    originals = {}
    for run, subsets in runs.items():
        if set(subsets) != set(SUBSETS):
            raise ValueError(f'Incomplete subsets: {run}')
        original = {}
        for subset, records in subsets.items():
            total = tuple(sum(r['file_' + k] for r in records.values()) for k in ('tp', 'fp', 'fn'))
            for row in records.values():
                advertised = tuple(row['file_' + k] + row['other_' + k] for k in ('tp', 'fp', 'fn'))
                if advertised != total:
                    raise ValueError(f'Inconsistent subset totals: {run}, {subset}')
                expected = 100 * float(f1(total))
                if abs(expected - float(row['original_subset_f'])) > 1e-9:
                    raise ValueError('Original subset count/score disagreement')
            original[subset] = total
        originals[run] = original

    interventions = []
    rule_points = []
    for key, affected in sorted(targets.items()):
        configuration, seed, variant, name = key
        run = configuration, seed, variant
        signatures = {(r['file_tp'], r['file_fp'], r['file_fn'], r['originally_silent']) for r in affected.values()}
        if len(signatures) != 1:
            raise ValueError(f'Shared-name file-count disagreement: {key}')
        tp, fp, fn, silent = next(iter(signatures))
        before = originals[run]
        after = dict(before)
        for subset, row in affected.items():
            total = before[subset]
            changed = (total[0] - tp, total[1] - fp, total[2] - fn + row['silenced_file_fn'])
            expected_penalty = 5 if variant == 'released' and not silent else 0
            if row['silenced_file_fn'] - fn - tp != expected_penalty:
                raise ValueError(f'Unexpected support penalty: {key}')
            after[subset] = changed
            if abs(100 * float(f1(changed)) - float(row['silenced_subset_f'])) > 1e-9:
                raise ValueError('Deleted subset count/score disagreement')
            if variant == 'corrected' and len(affected) == 1 and not silent:
                if tp + fp <= 0:
                    raise ValueError('Nonempty primary block has no counted predictions')
                margin = Fraction(tp, tp + fp) - f1(total) / 2
                delta = 100 * (f1(changed) - f1(total))
                if (margin < 0) != (delta > 0):
                    raise ValueError('Illustrative rule-point sign disagreement')
                rule_points.append(dict(configuration=configuration, seed=seed, recording=name, subset=subset,
                                        precision=float(Fraction(tp, tp + fp)), half_subset_f1=float(f1(total) / 2),
                                        margin=float(margin), delta_subset_points=float(delta), target_tp=tp, target_fp=fp))
        original = 100 * harmonic(f1(before[s]) for s in SUBSETS)
        deleted = 100 * harmonic(f1(after[s]) for s in SUBSETS)
        delta = deleted - original
        interventions.append(dict(configuration=configuration, seed=seed, evaluator_variant=variant,
                                  recording=name, affected_subsets=';'.join(s for s in SUBSETS if s in affected),
                                  originally_silent=silent, target_tp=tp, target_fp=fp, target_fn=fn,
                                  original_f=float(original), deleted_f=float(deleted), delta_exact=float(delta),
                                  gain=delta > 0, primary=not silent and len(affected) == 1))
    return runs, interventions, rule_points


def strata(interventions, legacy):
    rows = []
    for variant in ('released', 'corrected'):
        for positive in (True, False):
            group = [r for r in interventions if r['evaluator_variant'] == variant and r['primary']
                     and (r['target_tp'] > 0) == positive]
            exact = max(group, key=lambda r: r['delta_exact'])
            old = max(group, key=lambda r: float(legacy[r['configuration'], r['seed'], variant, r['recording']]['delta_fmeasure_harmonic']))
            legacy_max = float(legacy[old['configuration'], old['seed'], variant, old['recording']]['delta_fmeasure_harmonic'])
            rows.append(dict(evaluator_variant=variant, output='tp_positive' if positive else 'fp_only',
                             cases=len(group), gains=sum(r['gain'] for r in group),
                             rate_percent=100 * sum(r['gain'] for r in group) / len(group),
                             max_delta_exact=exact['delta_exact'], max_delta_legacy=legacy_max,
                             exact_max_configuration=exact['configuration'], exact_max_seed=exact['seed'],
                             exact_max_recording=exact['recording'], legacy_max_same_target=
                             (exact['configuration'], exact['seed'], exact['recording']) ==
                             (old['configuration'], old['seed'], old['recording'])))
    return rows


def subset_table(runs, interventions):
    populations = {}
    for (configuration, seed, variant), subsets in runs.items():
        if variant != 'corrected':
            continue
        for subset, records in subsets.items():
            value = (len(records), sum(r['file_tp'] + r['file_fn'] for r in records.values()))
            if subset in populations and populations[subset] != value:
                raise ValueError('Reference population differs between outputs')
            populations[subset] = value
    rows = []
    for subset in SUBSETS:
        group = [r for r in interventions if r['evaluator_variant'] == 'released' and r['primary']
                 and r['target_tp'] > 0 and r['affected_subsets'] == subset]
        recordings, events = populations[subset]
        rows.append(dict(subset=subset, recordings=recordings, events=events, cases=len(group),
                         gains=sum(r['gain'] for r in group), rate_percent=100 * sum(r['gain'] for r in group) / len(group)))
    total = {key: sum(r[key] for r in rows) for key in ('recordings', 'events', 'cases', 'gains')}
    rows.append(dict(subset='Total', **total, rate_percent=100 * total['gains'] / total['cases']))
    return rows


def recurrence(interventions, configurations, seeds, legacy):
    by_name = defaultdict(list)
    for r in interventions:
        if r['evaluator_variant'] == 'released' and r['primary'] and r['target_tp'] > 0:
            by_name[r['recording']].append(r)
    recurrent = {name: rows for name, rows in by_name.items() if all(r['gain'] for r in rows)}
    names = sorted(recurrent, key=lambda n: (-sum(r['delta_exact'] for r in recurrent[n]) / len(recurrent[n]), n))
    keyed = {(r['configuration'], r['seed'], r['recording']): r for r in interventions if r['evaluator_variant'] == 'released'}
    cells = []
    for name in names:
        for seed in seeds:
            for configuration in configurations:
                r = keyed[configuration, seed, name]
                eligible = r['primary'] and r['target_tp'] > 0
                old = legacy[configuration, seed, 'released', name]
                cells.append(dict(recording=name, configuration=configuration, seed=seed, eligible=eligible,
                                  delta_exact=r['delta_exact'] if eligible else '',
                                  legacy_delta=float(old['delta_fmeasure_harmonic']) if eligible else '',
                                  legacy_display_2dp=f"{float(old['delta_fmeasure_harmonic']):.2f}" if eligible else 'n/a',
                                  exact_display_2dp=f"{r['delta_exact']:.2f}" if eligible else 'n/a'))
    return cells


def joint_endpoints(runs, selections, bound_function=None):
    rows = []
    for selection in selections:
        configuration, seed = selection['configuration'], integer(selection['seed'])
        subsets = runs[configuration, seed, 'released']
        removed = set(filter(None, selection['removed_recordings'].split(';')))
        names = {name for records in subsets.values() for name in records}
        if len(names) != 42 or not removed <= names:
            raise ValueError('Invalid joint-selection population')
        before, after = {}, {}
        hits = set()
        effective = set()
        for subset, records in subsets.items():
            before[subset] = tuple(sum(r['file_' + k] for r in records.values()) for k in ('tp', 'fp', 'fn'))
            contributions = []
            for name, r in records.items():
                if r['file_tp'] > 0:
                    hits.add(name)
                if name in removed and not r['originally_silent']:
                    effective.add(name)
                contributions.append((0, 0, r['silenced_file_fn']) if name in removed else
                                     (r['file_tp'], r['file_fp'], r['file_fn']))
            after[subset] = tuple(sum(c[i] for c in contributions) for i in range(3))
        original = 100 * float(harmonic(f1(before[s]) for s in SUBSETS))
        deleted = 100 * float(harmonic(f1(after[s]) for s in SUBSETS))
        for key, expected in [('original_fmeasure_harmonic', original), ('oracle_fmeasure_harmonic', deleted)]:
            if abs(float(selection[key]) - expected) > 1e-9:
                raise ValueError(f'Archived selected-set score disagrees: {configuration}, {seed}, {key}')
        for subset in SUBSETS:
            if abs(float(selection['oracle_subset_' + subset]) - 100 * float(f1(after[subset]))) > 1e-9:
                raise ValueError('Archived selected subset score disagrees')
        if len(removed) != integer(selection['n_removed_recordings']):
            raise ValueError('Archived removal-count disagreement')
        optimum = bound_function(subsets) if bound_function else None
        if optimum is not None and abs(optimum - deleted) > 1e-9:
            raise ValueError('Selected endpoint is not the enumerated restricted maximum')
        rows.append(dict(configuration=configuration, seed=seed, original_f=original, selected_f=deleted,
                         delta=deleted - original, original_coverage=100 * len(hits) / len(names),
                         selected_coverage=100 * len(hits - removed) / len(names), selected_names=len(removed),
                         effective_nonempty_removals=len(effective), selected_recordings=';'.join(sorted(removed)),
                         restricted_optimum_verified=bound_function is not None))
    return rows
