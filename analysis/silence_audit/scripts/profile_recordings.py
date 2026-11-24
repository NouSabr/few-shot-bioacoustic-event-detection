#!/usr/bin/env python3
"""Profile gain-contributing recordings from bundled counts; no matching or audio."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import platform
import sys

import counts
from fresh_counts import derive_effects
from run_audit import CONFIGS, SEEDS, require, sha, write_csv

ROOT = Path(__file__).resolve().parents[1]


def load_cases():
    memberships = defaultdict(set)
    populations, references = {}, {}
    rows = counts.read_csv(ROOT / 'inputs/reference_populations.csv')
    for row in rows:
        subset, name = row['subset'], row['recording']
        require(subset in counts.SUBSETS and subset not in memberships[name],
                f'Invalid or duplicate membership: {subset}, {name}')
        memberships[name].add(subset)
        full, query = int(row['full_positive_events']), int(row['query_positive_events'])
        require(query > 0 and full - query == 5, f'Unsupported population: {name}')
        population = dict(full_positive_events=full, query_positive_events=query)
        require(name not in populations or populations[name] == population,
                f'Shared-name population mismatch: {name}')
        require(name not in references or references[name] == row['reference_sha256'],
                f'Shared-name reference hash mismatch: {name}')
        populations[name], references[name] = population, row['reference_sha256']
    require(len(populations) == 42 and len(rows) == 43, 'Expected 42 names / 43 memberships')
    pairs = {name: (p['full_positive_events'], p['query_positive_events'])
             for name, p in populations.items()}
    cohort = {(configuration, seed) for configuration in CONFIGS for seed in SEEDS}
    effects = derive_effects(counts.read_csv(ROOT / 'inputs/fresh_counts.csv'),
                             memberships, pairs, cohort)
    _, cases, _ = counts.reconstruct(effects)
    require(len(cases) == 2016, 'Expected 2,016 single-deletion cases')
    return cases, memberships, populations


def compare_tables(actual, expected_path, keys):
    expected_rows = counts.read_csv(expected_path)
    expected = {tuple(row[key] for key in keys): row for row in expected_rows}
    actual_keys = {tuple(str(row[key]) for key in keys) for row in actual}
    require(len(actual) == len(actual_keys) == len(expected) == len(expected_rows)
            and actual_keys == set(expected), f'Comparison cohort differs: {expected_path.name}')
    reconciliation = []
    for row in actual:
        key = tuple(str(row[field]) for field in keys)
        saved = expected[key]
        require(set(row) == set(saved), f'Comparison columns differ: {expected_path.name}')
        errors = []
        for field, value in row.items():
            if isinstance(value, float):
                error = abs(value - float(saved[field]))
                require(math.isfinite(error) and error < 1e-9,
                        f'Numeric mismatch: {expected_path.name}, {key}, {field}')
                errors.append(error)
            else:
                require(str(value) == saved[field],
                        f'Value mismatch: {expected_path.name}, {key}, {field}')
        reconciliation.append(dict(table=str(expected_path.relative_to(ROOT)),
                                   key=';'.join(key), maximum_numeric_error=max(errors, default=0.0),
                                   matched=True))
    return reconciliation


def profile(cases, memberships, populations):
    eligible = [row for row in cases if row['evaluator_variant'] == 'released'
                and row['primary'] and row['target_tp'] > 0]
    gains = [row for row in eligible if row['gain']]
    names = {row['recording'] for row in gains}
    require((len(eligible), len(gains), len(names)) == (932, 255, 13), 'Primary cohort differs')
    require(all(row['target_fp'] > row['target_tp'] for row in gains),
            'Expected false-positive-heavy gaining outputs')
    profiles, output_rows = [], []
    ordered_names = sorted(names, key=lambda name: (
        min(counts.SUBSETS.index(subset) for subset in memberships[name]), name))
    for name in ordered_names:
        group = [row for row in gains if row['recording'] == name]
        eligible_outputs = [row for row in eligible if row['recording'] == name]
        all_outputs = [row for row in cases if row['evaluator_variant'] == 'released'
                       and row['recording'] == name]
        require(len(all_outputs) == 24, f'Incomplete output cohort: {name}')
        population = populations[name]
        for row in all_outputs:
            predicted = row['target_tp'] + row['target_fp']
            output_rows.append(dict(
                **row, **population,
                original_precision_percent=100 * row['target_tp'] / predicted if predicted else 0.0,
                original_query_recall_percent=100 * row['target_tp'] / population['query_positive_events']))
        row = dict(recording=name, subset=next(iter(memberships[name])), **population,
                   total_outputs=len(all_outputs), gains=len(group),
                   eligible_tp_positive_outputs=len(eligible_outputs),
                   gains_whenever_tp_positive=len(group) == len(eligible_outputs),
                   gain_seeds=';'.join(str(seed) for seed in SEEDS
                                       if any(r['seed'] == seed for r in group)),
                   gain_configurations=';'.join(configuration for configuration in CONFIGS
                                                if any(r['configuration'] == configuration for r in group)))
        for field in ('tp', 'fp', 'fn'):
            values = [r['target_' + field] for r in group]
            row['gaining_' + field + '_min'] = min(values)
            row['gaining_' + field + '_max'] = max(values)
        for field, values in (
            ('precision', [100 * r['target_tp'] / (r['target_tp'] + r['target_fp']) for r in group]),
            ('recall', [100 * r['target_tp'] / population['query_positive_events'] for r in group])):
            row['gaining_' + field + '_min_percent'] = min(values)
            row['gaining_' + field + '_max_percent'] = max(values)
        row['gaining_delta_min_points'] = min(r['delta_exact'] for r in group)
        row['gaining_delta_max_points'] = max(r['delta_exact'] for r in group)
        profiles.append(row)
    subsets = [dict(
        subset=subset,
        gaining_names=len({r['recording'] for r in gains if r['affected_subsets'] == subset}),
        gains=sum(r['affected_subsets'] == subset for r in gains),
        eligible_tp_positive_cases=sum(r['affected_subsets'] == subset for r in eligible))
        for subset in counts.SUBSETS]
    recurrent = sum(row['gains_whenever_tp_positive'] for row in profiles)
    require(recurrent == 10, 'Recording recurrence differs')
    summary = dict(primary_tp_positive_cases=len(eligible), official_gaining_cases=len(gains),
                   gaining_recording_names=len(names), recurrent_names=recurrent,
                   reconstructed_cases=len(cases), distinct_recording_names=len(populations),
                   reference_subset_rows=sum(map(len, memberships.values())), outputs=len(CONFIGS) * len(SEEDS),
                   acoustic_explanation_verified=False, event_matching_executed=False,
                   raw_annotations_read=False)
    return profiles, output_rows, subsets, summary


def report_text(profiles, subsets, summary):
    lines = [
        '# Gain-contributing recording profiles', '',
        'Counts-only characterization; no model, matching, raw annotation, or audio execution.', '',
        f"**{summary['official_gaining_cases']} gains / {summary['primary_tp_positive_cases']} eligible "
        f"cases, from {summary['gaining_recording_names']} distinct names.**", '',
        'Eligibility: released scoring, originally nonempty predictions, one affected subset,',
        'and at least one original TP. The shared PB/PB24 name is excluded from this cohort.',
        'The 24 outputs reuse recordings; these are dependent descriptive observations.', '',
        '## Subsets', '',
        '| Subset | Gaining names | Gains | Eligible cases |',
        '| --- | ---: | ---: | ---: |',
    ]
    lines += [f"| {r['subset']} | {r['gaining_names']} | {r['gains']} | {r['eligible_tp_positive_cases']} |"
              for r in subsets]
    lines += ['', '## Profiles', '',
              'Precision/recall ranges below describe gaining outputs only.',
              'Query POS excludes five support events; every eligible denominator is explicit.', '',
              '| Recording | Subset | Query POS | Gains/eligible | Precision (%) | Recall (%) |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in profiles:
        precision = (f"{row['gaining_precision_min_percent']:.2f}–"
                     f"{row['gaining_precision_max_percent']:.2f}")
        recall = f"{row['gaining_recall_min_percent']:.2f}–{row['gaining_recall_max_percent']:.2f}"
        lines.append(f"| `{row['recording']}` | {row['subset']} | {row['query_positive_events']} | "
                     f"{row['gains']}/{row['eligible_tp_positive_outputs']} | {precision} | {recall} |")
    lines += [
        '', f"{summary['recurrent_names']} names gain whenever originally TP-positive.",
        'Full TP/FP/FN and score-delta ranges are in recording_profiles.csv.',
        'recording_output_evidence.csv includes all 24 outputs for each profiled name,',
        'including nongaining and TP-zero cases.', '',
        '## Scope and provenance', '',
        'All deletion cases and profile/subset summaries reconcile with frozen comparison tables.',
        'Population counts and reference hashes are read from bundled metadata; raw annotation',
        'bytes and audio identity are not re-verified. No event durations or cached SNR are used.',
        'An acoustic/ecological explanation remains unverified. No listening or causal claim is made.',
        'The run manifest records package/input/output hashes and the Python version.',
    ]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='NEW directory; never overwritten')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    require(not out.exists(), f'Refusing to overwrite: {out}')
    package = json.loads((ROOT / 'manifest.json').read_text())
    inputs = {**package['files_sha256'], 'manifest.json': sha(ROOT / 'manifest.json')}
    for name, expected in inputs.items():
        require(sha(ROOT / name) == expected, f'Package integrity failure: {name}')
    cases, memberships, populations = load_cases()
    reconciliation = compare_tables(cases, ROOT / 'expected/interventions.csv',
                                    ('configuration', 'seed', 'evaluator_variant', 'recording'))
    profiles, output_rows, subsets, summary = profile(cases, memberships, populations)
    for filename, rows, keys in (
        ('recording_profiles.csv', profiles, ('recording',)),
        ('subset_summary.csv', subsets, ('subset',)),
    ):
        reconciliation += compare_tables(rows, ROOT / 'recording_evidence/expected' / filename, keys)
    for name, expected in inputs.items():
        require(sha(ROOT / name) == expected, f'Input changed during analysis: {name}')
    out.mkdir(parents=True, exist_ok=False)
    for filename, rows in (
        ('recording_profiles.csv', profiles), ('recording_output_evidence.csv', output_rows),
        ('gaining_cases.csv', [row for row in output_rows
                              if row['primary'] and row['target_tp'] > 0 and row['gain']]),
        ('subset_summary.csv', subsets), ('reconciliation.csv', reconciliation),
    ):
        write_csv(out / filename, rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (out / 'report.md').write_text(report_text(profiles, subsets, summary), encoding='utf-8')
    (out / 'sources').mkdir()
    (out / 'sources/profile_recordings.py').write_bytes(Path(__file__).read_bytes())
    for name, expected in inputs.items():
        require(sha(ROOT / name) == expected, f'Input changed while writing: {name}')
    outputs = {str(path.relative_to(out)): sha(path)
               for path in sorted(out.rglob('*')) if path.is_file()}
    manifest = dict(python=platform.python_version(), inputs_sha256=inputs, outputs_sha256=outputs,
                    event_matching_executed=False, raw_annotations_read=False,
                    acoustic_measurement_recomputed=False, training_or_inference=False)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        sys.exit(f'ERROR: {error}')
