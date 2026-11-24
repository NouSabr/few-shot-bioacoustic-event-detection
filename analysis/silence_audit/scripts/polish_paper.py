#!/usr/bin/env python3
"""Portable final-font/tick export from audited SVGs and archived plot counts.

Requires requirements-plotting.txt. Never installs packages or runs models/matching.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--svg-dir', type=Path, required=True,
                        help='Newly generated standard-library plot_paper.py outputs')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), 'Refusing to overwrite a figure directory')
    bundle = Path(__file__).resolve().parents[1]
    inputs = bundle / 'inputs/plotting'
    font = bundle / 'assets/fonts/LiberationSans-Regular.ttf'
    expected = json.loads((bundle / 'expected/formatting_invariants.json').read_text())
    from svg_signature import svg_signature
    for name, digest in expected['legacy_svg_shape_sha256'].items():
        require(svg_signature(args.svg_dir / name) == digest, 'Calculated SVG drawing differs: ' + name)
    svg_byte_differences = [name for name, digest in expected['legacy_svg_sha256'].items()
                            if sha(args.svg_dir / name) != digest]
    for name, digest in expected['plot_input_sha256'].items():
        require(sha(bundle / name) == digest, 'Archived plotting input differs: ' + name)
    require(sha(font) == expected['font_sha256'], 'Font bytes differ')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import formatting_drawing as drawing
    import rule_check_drawing as original

    args.output_dir.mkdir(parents=True)
    out = args.output_dir
    plt.rcParams.update({'pdf.fonttype': 42, 'pdf.use14corefonts': False})
    for stem in ('audit_workflow', 'score_coverage_paired_panels', 'recall_boxplots'):
        drawing.reexport_svg(args.svg_dir / (stem + '.svg'), out / stem, font)

    frames = [pd.read_csv(inputs / name) for name in
              ('subset_effects_seeds1234_2026.csv', 'subset_effects_seed3009.csv')]
    data = pd.concat(frames, ignore_index=True)
    keys = ['configuration', 'seed', 'evaluator_variant', 'target_filename', 'subset']
    require(not data.duplicated(keys).any(), 'Unexpected duplicate plot observations')
    evidence = {}

    def save_with_ticks(fig, directory, stem):
        ax, = fig.axes
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        before = [hashlib.sha256(np.ma.getdata(c.get_offsets()).tobytes()).hexdigest()
                  for c in ax.collections]
        ax.set_yticks([-10, -1, 0, 1, 10], labels=['−10', '−1', '0', '1', '10'])
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        fig.tight_layout(pad=0.35)
        fig.canvas.draw()
        boxes = [t.get_window_extent(fig.canvas.get_renderer()) for t in ax.get_yticklabels()]
        require(not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:]),
                'Rule-check tick labels overlap')
        after = [hashlib.sha256(np.ma.getdata(c.get_offsets()).tobytes()).hexdigest()
                 for c in ax.collections]
        require(before == after == expected['rule_check']['point_array_sha256'],
                'Plotted points differ from approved formatting run')
        transform = ax.yaxis.get_transform()
        require(list(ax.get_xlim()) == expected['rule_check']['xlim'] and
                list(ax.get_ylim()) == expected['rule_check']['ylim'] and
                ax.get_yscale() == 'symlog' and transform.linthresh == 0.5 and transform.linscale == 0.8,
                'Plot limits/transform differ')
        evidence.update(point_array_sha256=before, points_unchanged=True,
                        yscale='symlog', linthresh=0.5, linscale=0.8,
                        yticks=[-10, -1, 0, 1, 10], ytick_labels_do_not_overlap=True,
                        population=sum(len(c.get_offsets()) for c in ax.collections))
        fig.savefig(out / 'fig_rule_check.pdf', bbox_inches='tight', metadata={
            'CreationDate': None, 'ModDate': None,
            'Creator': 'paper_revision/tooling/polish_figures.py',
        })
        fig.savefig(out / 'fig_rule_check.png', bbox_inches='tight', dpi=180)
        plt.close(fig)

    original.save_figure = save_with_ticks
    sns.set_theme(style='whitegrid', context='paper')
    plt.rcParams.update({'pdf.fonttype': 42, 'pdf.use14corefonts': False})
    original.rule_check_figure(data, out)
    require(evidence['population'] == 971, 'Expected 971 rule-check observations')
    # Supplied heatmap with explicit seed labels; matrix drawing is preserved from the historical PDF.
    shutil.copyfile(bundle / 'paper_figures/fig_tpos_recurrence_heatmap.pdf',
                    out / 'fig_tpos_recurrence_heatmap.pdf')
    hashes = json.loads((bundle / 'expected/paper_figure_sha256.json').read_text())
    for name, digest in hashes.items():
        require(sha(out / name) == digest, 'Final PDF differs: ' + name)
    report = dict(status='passed', final_pdfs_sha256=hashes, rule_check=evidence,
                  svg_shape_precision_digits=9, intermediate_svg_byte_differences=svg_byte_differences,
                  versions={m.__name__: m.__version__ for m in (matplotlib, np, pd, sns)},
                  full_historical_heatmap_regeneration_claimed=False,
                  new_metrics_matching_or_models=False)
    (out / 'formatting_report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, ImportError) as error:
        sys.exit('ERROR: ' + str(error))
