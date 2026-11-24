#!/usr/bin/env python3
"""Portable frozen-count audit: no model, audio, event matching, or installs."""
from collections import Counter, defaultdict
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics
import sys

import counts
from fresh_counts import derive_effects, derived_logged_rows

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ('baseline','cife','phase1_random','slot_uniform','phase2_energy',
           'phase3_activation','afri_direct','afri_inverse')
SEEDS = (1234,2026,3009)
LABELS = ('Baseline','CIFE-inspired','Random','Slot-uniform','Energy','Activation','AFRI direct','AFRI inverse')


def require(value,message):
    if not value: raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path,rows):
    require(bool(rows),f'Empty output: {path}')
    with path.open('x',newline='',encoding='utf-8') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def ordinary_f1(tp,fp,fn):
    total=2*tp+fp+fn
    return Fraction(2*tp,total) if total else Fraction(0)


def quantile(values,p):
    values=sorted(values); position=(len(values)-1)*p
    lower=position.numerator//position.denominator; upper=min(lower+1,len(values)-1)
    return values[lower]+(position-lower)*(values[upper]-values[lower])


def enumerate_joint_maximum(subsets):
    """Verify only the maximum score; never substitute a tied set for archived choices.

    Independent subsets optimize separately. PB/PB24 share one decision, so
    minimize their combined reciprocal F1 contribution over their nine-name union.
    Gray-code enumeration updates each candidate with integer count arithmetic.
    """
    def candidates(subset_names):
        names=sorted({n for s in subset_names for n in subsets[s]})
        totals={s:[0,0,sum(r['silenced_file_fn'] for r in subsets[s].values())] for s in subset_names}
        yield totals
        previous=0
        for number in range(1,1<<len(names)):
            gray=number^(number>>1); changed=gray^previous
            name=names[changed.bit_length()-1]; sign=1 if gray&changed else -1
            for s in subset_names:
                if name not in subsets[s]: continue
                r=subsets[s][name]
                for i,delta in enumerate((r['file_tp'],r['file_fp'],r['file_fn']-r['silenced_file_fn'])):
                    totals[s][i]+=sign*delta
            yield totals
            previous=gray
    reciprocal=Fraction(0)
    for subset in ('HB','ME','PW','RD'):
        best=max(counts.f1(tuple(t[subset])) for t in candidates((subset,)))
        reciprocal+=1/best
    pair_best=min(1/counts.f1(tuple(t['PB']))+1/counts.f1(tuple(t['PB24']))
                  for t in candidates(('PB','PB24')))
    return 100*float(Fraction(6)/(reciprocal+pair_best))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True,help='NEW directory; never overwritten')
    parser.add_argument('--verify-maxima',action='store_true',help='Also enumerate all 24 exact joint maxima')
    args=parser.parse_args()
    out=args.output_dir.resolve()
    require(not out.exists(),f'Refusing to overwrite: {out}')
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,digest in manifest['files_sha256'].items():
        require(sha(ROOT/name)==digest,f'Bundle integrity failure: {name}')
    source=ROOT/'inputs'
    population_rows=counts.read_csv(source/'reference_populations.csv')
    memberships=defaultdict(set); populations={}
    for r in population_rows:
        name=r['recording']; memberships[name].add(r['subset'])
        pair=int(r['full_positive_events']),int(r['query_positive_events'])
        require(pair[1]>0 and pair[0]-pair[1]==5,f'Unsupported reference population: {name}')
        require(name not in populations or populations[name]==pair,f'Shared-reference mismatch: {name}')
        populations[name]=pair
    require(len(populations)==42 and sum(map(len,memberships.values()))==43,'Expected 42 names / 43 memberships')
    cohort={(c,s) for c in CONFIGS for s in SEEDS}
    raw=counts.read_csv(source/'fresh_counts.csv')
    effects=derive_effects(raw,memberships,populations,cohort)
    runs,interventions,rule=counts.reconstruct(effects)
    require(len(interventions)==2016 and len(rule)==971,'Unexpected deletion/rule cohort')
    legacy=derived_logged_rows(interventions)
    strata=counts.strata(interventions,legacy)
    subset_summary=counts.subset_table(runs,interventions)
    recurrence=counts.recurrence(interventions,CONFIGS,SEEDS,legacy)
    expected_interventions={tuple(r[k] for k in ('configuration','seed','evaluator_variant','recording')):r
                            for r in counts.read_csv(ROOT/'expected/interventions.csv')}
    require(len(expected_interventions)==len(interventions),'Incomplete intervention reference')
    comparisons=[]
    def compare(group,key,actual,expected,tolerance=1e-9):
        difference=abs(float(actual)-float(expected))
        require(difference<=tolerance,f'Reconciliation failed {group}, {key}: {actual} != {expected}')
        comparisons.append(dict(group=group,key=str(key),actual=float(actual),expected=float(expected),
                                absolute_difference=difference,tolerance=tolerance,passed=True))
    for r in interventions:
        key=(r['configuration'],str(r['seed']),r['evaluator_variant'],r['recording'])
        require(key in expected_interventions,f'Missing expected intervention {key}')
        expected=expected_interventions[key]
        for field in ('original_f','deleted_f','delta_exact'):
            compare('single_deletion',(*key,field),r[field],expected[field])
        for field in ('gain','primary','originally_silent'):
            require(r[field]==(expected[field]=='True'),f'Boolean mismatch {key,field}')
    selections=counts.read_csv(source/'joint_deletions.csv')
    require({(r['configuration'],int(r['seed'])) for r in selections}==cohort and len(selections)==24,'Invalid selection cohort')
    endpoints=counts.joint_endpoints(runs,selections,enumerate_joint_maximum if args.verify_maxima else None)
    expected_endpoints={(r['configuration'],int(r['seed'])):r for r in counts.read_csv(source/'archived_endpoints.csv')}
    for r in endpoints:
        e=expected_endpoints[r['configuration'],r['seed']]
        for actual,expected in [('original_f','original_f'),('selected_f','oracle_f'),
                                ('original_coverage','original_coverage'),('selected_coverage','oracle_coverage')]:
            compare('joint_endpoint',(r['configuration'],r['seed'],actual),r[actual],e[expected])
    original_keyed={(r['configuration'],int(r['seed']),r['recording']):r
                    for r in raw if r['evaluator_variant']=='released'}
    selected={(r['configuration'],r['seed']):r for r in endpoints}
    expected_metrics={(r['configuration'],int(r['seed'])):r
                      for r in counts.read_csv(ROOT/'expected/metrics_by_output.csv')}
    metrics=[]; recordings=[]; quantiles=[]
    for c,label in zip(CONFIGS,LABELS):
        for seed in SEEDS:
            endpoint=selected[c,seed]
            removed=set(filter(None,endpoint['selected_recordings'].split(';')))
            row=dict(configuration=c,configuration_label=label,seed=seed,n_recordings=42,n_subset_memberships=43,
                     n_selected_names=len(removed),
                     n_selected_tp_positive=sum(int(original_keyed[c,seed,n]['tp'])>0 for n in removed),
                     n_selected_originally_empty=sum(original_keyed[c,seed,n]['has_predictions']=='False' for n in removed))
            stages={}
            for stage in ('original','deleted'):
                fvalues=[]; recalls=[]
                for name in sorted(populations):
                    r=original_keyed[c,seed,name]; full,query=populations[name]
                    delete=stage=='deleted' and name in removed
                    tp,fp=(0,0) if delete else (int(r['tp']),int(r['fp']))
                    fn=query-tp; fvalue=ordinary_f1(tp,fp,fn); recall=Fraction(tp,query)
                    fvalues.append(fvalue); recalls.append(recall)
                    recordings.append(dict(configuration=c,seed=seed,recording=name,stage=stage,
                        subset_memberships=';'.join(s for s in counts.SUBSETS if s in memberships[name]),
                        selected_for_deletion=name in removed,query_positive_events=query,
                        tp=tp,fp=fp,fn_query=fn,f1_score=100*float(fvalue),recall_percent=100*float(recall)))
                stages[stage]=dict(official=endpoint['original_f' if stage=='original' else 'selected_f'],
                    macro_f1=100*sum(fvalues)/42,recall_mean=100*sum(recalls)/42,
                    coverage=100*Fraction(sum(v>0 for v in recalls),42),zero_count=sum(v==0 for v in recalls))
                q=dict(configuration=c,seed=seed,stage=stage,n_recordings=42,zero_recall_count=stages[stage]['zero_count'])
                for key,p in [('minimum',Fraction(0)),('q1',Fraction(1,4)),('median',Fraction(1,2)),('q3',Fraction(3,4)),('maximum',Fraction(1))]:
                    q[key+'_recall_percent']=100*float(quantile(recalls,p))
                q['mean_recall_percent']=float(stages[stage]['recall_mean']); quantiles.append(q)
            for field in ('official','macro_f1','recall_mean','coverage','zero_count'):
                for stage in ('original','deleted'):
                    row[stage+'_'+field]=float(stages[stage][field])
                row['delta_'+field]=float(stages['deleted'][field]-stages['original'][field])
            for key,value in row.items():
                if isinstance(value,(float,int)):
                    compare('recording_metric',(c,seed,key),value,expected_metrics[c,seed][key])
            metrics.append(row)
    expected_subsets={r['subset']:r for r in counts.read_csv(ROOT/'expected/subset_summary.csv')}
    for r in subset_summary:
        for key in ('recordings','events','cases','gains','rate_percent'):
            compare('subset_summary',(r['subset'],key),r[key],expected_subsets[r['subset']][key])
    official_primary=next(r for r in strata if r['evaluator_variant']=='released' and r['output']=='tp_positive')
    require(official_primary['cases']==932 and official_primary['gains']==255,'Headline prevalence differs')
    summary=[]
    for metric in ('official','macro_f1','recall_mean','coverage','zero_count'):
        for stage in ('original','deleted','delta'):
            values=[r[stage+'_'+metric] for r in metrics]
            summary.append(dict(metric=metric,stage=stage,n_outputs=24,mean=statistics.mean(values),
                                sample_sd=statistics.stdev(values),minimum=min(values),maximum=max(values)))
    require(all(sha(ROOT/name)==digest for name,digest in manifest['files_sha256'].items()),'Bundle changed while calculating')
    out.mkdir(parents=True)
    for name,rows in [('single_deletions.csv',interventions),('deletion_summary.csv',strata),
                      ('subset_summary.csv',subset_summary),('rule_check.csv',rule),('recurrence.csv',recurrence),
                      ('joint_endpoints.csv',endpoints),('metrics_by_output.csv',metrics),
                      ('recording_metrics.csv',recordings),('recall_quantiles_by_output.csv',quantiles),
                      ('summary.csv',summary),('reconciliation.csv',comparisons)]:
        write_csv(out/name,rows)
    result=dict(status='passed',n_outputs=24,n_recording_names=42,n_memberships=43,
                single_deletions=len(interventions),checks=len(comparisons),
                joint_maxima_verified=24 if args.verify_maxima else 0,
                official_tp_positive_cases=932,official_tp_positive_gains=255,
                matching_performed=False,training_or_inference=False,
                selection_note='Archived sets retained; a maximum score need not have a unique set/coverage.',
                files_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'audit_report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(f"Passed {len(comparisons)} count/metric checks; 255/932 TP-positive gains.")
    print(f"Joint maxima verified: {result['joint_maxima_verified']}; no event matching or model execution.")


if __name__=='__main__':
    try: main()
    except (ValueError,OSError,KeyError) as error: sys.exit(f'ERROR: {error}')
