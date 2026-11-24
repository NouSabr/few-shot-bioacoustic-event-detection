"""Derive silence-audit counts from bundled matched counts and populations."""
from __future__ import annotations

from counts import SUBSETS, f1, integer

POLICIES = ('released', 'corrected')
N_SHOTS = 5


def boolean(value):
    if value not in ('True', 'False'):
        raise ValueError(f'Expected True/False, got {value!r}')
    return value == 'True'


def derive_effects(rows, memberships, populations, cohort):
    """Derive file/subset deletion counts without archived intervention values."""
    keyed = {}
    for raw in rows:
        key = raw['configuration'], integer(raw['seed']), raw['evaluator_variant'], raw['recording']
        if key in keyed:
            raise ValueError(f'Duplicate fresh count: {key}')
        counts = tuple(integer(raw[k]) for k in ('tp', 'fp', 'fn'))
        present = boolean(raw['has_predictions'])
        if min(counts) < 0 or (not present and counts[:2] != (0, 0)):
            raise ValueError(f'Invalid count/presence combination: {key}')
        keyed[key] = (*counts, present)
    wanted = {(c, s, v, name) for c, s in cohort for v in POLICIES for name in memberships}
    if set(keyed) != wanted:
        raise ValueError('Fresh-count cohort/policy/name coverage is incomplete or unexpected')
    if set(populations) != set(memberships):
        raise ValueError('Reference populations and memberships differ')
    for c, s in sorted(cohort):
        for name in memberships:
            released = keyed[c, s, 'released', name]
            corrected = keyed[c, s, 'corrected', name]
            t, f, n, present = released
            full, query = populations[name]
            if full - query != N_SHOTS:
                raise ValueError('Reference populations violate the five-event policy')
            if corrected != (t, f, n - (0 if present else N_SHOTS), present):
                raise ValueError(f'Unexpected official/support-consistent count difference: {(c, s, name)}')
            if t + n != (query if present else full):
                raise ValueError(f'Fresh counts disagree with reference population: {(c, s, name)}')

    effects = {}
    for c, s in sorted(cohort):
        for variant in POLICIES:
            totals = {subset: tuple(sum(keyed[c, s, variant, name][i] for name, ss in memberships.items()
                                        if subset in ss) for i in range(3)) for subset in SUBSETS}
            for name, subsets in sorted(memberships.items()):
                t, f, n, present = keyed[c, s, variant, name]
                deleted_fn = populations[name][0 if variant == 'released' else 1]
                for subset in SUBSETS:
                    if subset not in subsets:
                        continue
                    total = totals[subset]
                    other = tuple(total[i] - (t, f, n)[i] for i in range(3))
                    changed = (other[0], other[1], other[2] + deleted_fn)
                    key = c, s, variant, name, subset
                    effects[key] = dict(configuration=c, seed=s, evaluator_variant=variant,
                                        target_filename=name, subset=subset, originally_silent=not present,
                                        file_tp=t, file_fp=f, file_fn=n, silenced_file_fn=deleted_fn,
                                        other_tp=other[0], other_fp=other[1], other_fn=other[2],
                                        original_subset_f=100 * float(f1(total)),
                                        silenced_subset_f=100 * float(f1(changed)))
    return effects


def derived_logged_rows(interventions):
    """Recreate reported-score deltas by rounding both fresh scores to 3 decimals."""
    rows = {}
    for r in interventions:
        original, deleted = round(r['original_f'], 3), round(r['deleted_f'], 3)
        key = r['configuration'], r['seed'], r['evaluator_variant'], r['recording']
        if key in rows:
            raise ValueError(f'Duplicate intervention: {key}')
        rows[key] = dict(configuration=key[0], seed=key[1], evaluator_variant=key[2],
                         target_filename=key[3], original_fmeasure_harmonic=original,
                         silenced_fmeasure_harmonic=deleted,
                         delta_fmeasure_harmonic=deleted - original)
    return rows
