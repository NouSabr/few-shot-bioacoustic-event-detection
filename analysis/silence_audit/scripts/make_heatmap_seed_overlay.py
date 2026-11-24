#!/usr/bin/env python3
"""Draw only the three replacement seed headers; never redraw heatmap cells."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Refusing to overwrite an overlay directory')
    bundle = Path(__file__).resolve().parents[1]
    recipe = json.loads((bundle / 'expected/heatmap_seed_labels.json').read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans-Bold.ttf'
    if hashlib.sha256(font.read_bytes()).hexdigest() != recipe['font_sha256']:
        raise ValueError('Replacement-header font differs from recorded runtime')
    width, height = recipe['media_box_pt']
    fig = plt.figure(figsize=(width / 72, height / 72))
    fig.patch.set_alpha(0)
    plt.rcParams.update({'pdf.fonttype': 42, 'pdf.use14corefonts': False})
    for label, center in zip(recipe['new_labels'], recipe['centers_x_pt']):
        fig.text(center / width, recipe['baseline_y_pt'] / height, label,
                 fontproperties=FontProperties(fname=str(font), size=9),
                 ha='center', va='baseline', color='#262626')
    args.output_dir.mkdir(parents=True)
    fig.savefig(args.output_dir / 'heatmap_seed_labels_overlay.pdf', transparent=True,
                metadata={'CreationDate': None, 'ModDate': None,
                          'Creator': 'make_heatmap_seed_overlay.py'})
    plt.close(fig)


if __name__ == '__main__':
    main()
