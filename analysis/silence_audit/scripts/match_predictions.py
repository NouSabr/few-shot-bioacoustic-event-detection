#!/usr/bin/env python3
"""OPTIONAL native original-prediction matching; requires external reference CSVs.

Not run during bundle staging. No audio/model/training required. Official scoring
is unmodified; support-consistent counts use the historical backfill convention.
"""
import argparse
from contextlib import redirect_stdout
import csv
from decimal import Decimal
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def require(value,message):
    if not value: raise ValueError(message)
def rows(p):
    with p.open(newline='',encoding='utf-8-sig') as h: return list(csv.DictReader(h))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references-root',type=Path,required=True,
                        help='Directory recursively containing the exact reference CSVs named in the manifest')
    parser.add_argument('--output-dir',type=Path,required=True,help='NEW directory; no overwrite')
    parser.add_argument('--limit',type=int,default=24,help='1 for an explicit preflight; full cohort is 24')
    args=parser.parse_args()
    require(1<=args.limit<=24,'Limit must be 1..24')
    out=args.output_dir.resolve()
    require(not out.exists(),f'Refusing to overwrite: {out}')
    bundle=json.loads((ROOT/'manifest.json').read_text())
    for name,digest in bundle['files_sha256'].items(): require(sha(ROOT/name)==digest,f'Integrity failure: {name}')
    # Third-party imports are deliberately optional and occur only after argument parsing.
    require(importlib.metadata.version('mir-eval')=='0.7','Expected the recorded mir-eval 0.7 runtime')
    sys.path.insert(0,str(ROOT/'vendor/dcase2024'))
    from src.utils import evaluation as evaluator
    from src.utils import metrics as matcher
    import mir_eval.util
    runtime=json.loads((ROOT/'matching_runtime.json').read_text())
    require(sha(Path(mir_eval.util.__file__))==runtime['mir_eval_util_sha256'],
            'mir_eval.util source differs from the recorded replay; review rather than silently substitute')
    require(sha(Path(evaluator.__file__))==bundle['native_sources']['evaluation.py'],'Evaluator source mismatch')
    require(sha(Path(matcher.__file__))==bundle['native_sources']['metrics.py'],'Matcher source mismatch')
    populations=rows(ROOT/'inputs/reference_populations.csv')
    available={}
    for p in sorted(args.references_root.resolve().rglob('*.csv')):
        available.setdefault(p.name,[]).append(p)
    selected_references={}
    for r in populations:
        name=r['recording'][:-4]+'.csv'
        matches=[p for p in available.get(name,[]) if sha(p)==r['reference_sha256']]
        require(bool(matches),f'Missing byte-matching reference CSV: {name}; see DATA_ACCESS.md')
        selected_references[r['subset'],r['recording']]=matches[0]
    # Inputs must remain bytes/row-order preserving, including the shared PB/PB24 name.
    out.mkdir(parents=True)
    reference_dir=out/'references'
    for (subset,name),p in selected_references.items():
        destination=reference_dir/subset/(name[:-4]+'.csv')
        destination.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,destination)
    expected={(r['configuration'],int(r['seed']),r['evaluator_variant'],r['recording']):r
              for r in rows(ROOT/'inputs/fresh_counts.csv')}
    unique_pop={r['recording']:r for r in populations}
    result=[]
    cohort=rows(ROOT/'inputs/prediction_manifest.csv')[:args.limit]
    for item in cohort:
        c,s=item['configuration'],int(item['seed'])
        prediction=ROOT/item['bundle_path']
        require(sha(prediction)==item['sha256'],f'Prediction hash mismatch: {c,s}')
        folder=out/'native_reports'/f'{c}_seed{s}'
        folder.mkdir(parents=True)
        captured={}; writer=evaluator.build_report
        def observe(*pos,**kw):
            frame=sys._getframe(1)
            require(frame.f_code is evaluator.evaluate.__code__,'Unexpected report-writer caller')
            captured.update({n:dict(v) for n,v in frame.f_locals['counts_per_audiofile'].items()})
            return writer(*pos,**kw)
        evaluator.build_report=observe
        try:
            with (folder/'stdout.log').open('x') as log,redirect_stdout(log):
                _,nonempty,_,_=evaluator.evaluate(str(prediction),str(reference_dir),'audit_bundle','VAL',str(folder))
        finally:
            evaluator.build_report=writer
        require(set(captured)==set(unique_pop),'Incomplete native count capture')
        for name,r in sorted(unique_pop.items()):
            original=captured[name]
            present=name in nonempty
            for policy in ('released','corrected'):
                tp,fp,fn=[int(original[k]) for k in ('TP','FP','FN')]
                if policy=='corrected' and not present:
                    require(tp==fp==0,'Unexpected nonempty missing-output counts')
                    fn=int(r['query_positive_events'])
                value=dict(configuration=c,seed=s,evaluator_variant=policy,recording=name,
                           tp=tp,fp=fp,fn=fn,has_predictions=present)
                target=expected[c,s,policy,name]
                require((tp,fp,fn)==tuple(int(target[k]) for k in ('tp','fp','fn')),
                        f'Native count mismatch: {c,s,policy,name}')
                require(present==(target['has_predictions']=='True'),'Prediction-presence mismatch')
                result.append(value)
    for name,digest in bundle['files_sha256'].items(): require(sha(ROOT/name)==digest,f'Input changed: {name}')
    with (out/'fresh_counts.csv').open('x',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(result[0])); writer.writeheader(); writer.writerows(result)
    report=dict(outputs_matched=len(cohort),count_rows=len(result),status='passed',
                observer='report-writer observer only; native evaluator and matcher unmodified',
                support_consistent='empty-output reference counts corrected after official matching, not another matcher',
                native_direct_patch_applied=False,training_or_inference=False,
                original_historical_environment_recreated=False)
    (out/'matching_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(f'Matched {len(cohort)} outputs; {len(result)} recording/policy rows match the frozen reference.')


if __name__=='__main__':
    try: main()
    except (ValueError,OSError,KeyError,ImportError) as error: sys.exit(f'ERROR: {error}')
