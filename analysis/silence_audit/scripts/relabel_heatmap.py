#!/usr/bin/env python3
"""Replace only Seed A/B/C text blocks in the supplied historical heatmap.

Requires pypdf. The base drawing stream is preserved byte-for-byte except for
three removed header text blocks. No matrix, cell, axis or data redraw occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (ArrayObject, DecodedStreamObject, DictionaryObject,
                           FloatObject, NameObject)


class ExactBoxFloat(FloatObject):
    def myrepr(self):
        return f'{float(self):.12f}'.rstrip('0').rstrip('.') or '0'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--overlay', type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Refusing to overwrite a heatmap directory')
    bundle = Path(__file__).resolve().parents[1]
    recipe = json.loads((bundle / 'expected/heatmap_seed_labels.json').read_text())
    base = bundle / 'assets/heatmap_seed_labels_base.pdf'
    overlay = args.overlay or bundle / 'assets/heatmap_seed_labels_overlay.pdf'
    if sha(base) != recipe['base_sha256'] or sha(overlay) != recipe['overlay_sha256']:
        raise ValueError('Heatmap base or replacement-header overlay differs')
    reader, overlay_reader = PdfReader(base), PdfReader(overlay)
    if len(reader.pages) != 1 or len(overlay_reader.pages) != 1:
        raise ValueError('Expected one-page base and header PDFs')
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    page = writer.pages[0]
    if [float(page.mediabox.width), float(page.mediabox.height)] != recipe['media_box_pt']:
        raise ValueError('Heatmap dimensions differ')
    original_stream = reader.pages[0].get_contents().get_data()
    cleaned = original_stream
    for label in recipe['old_labels']:
        block = b'BT\n/F2 9 Tf\n0 0 Td\n[ (' + label.encode('ascii') + b') ] TJ\nET'
        if cleaned.count(block) != 1:
            raise ValueError('Unexpected historical seed-header text block: ' + label)
        cleaned = cleaned.replace(block, b'', 1)
    overlay_page = overlay_reader.pages[0]
    overlay_data = overlay_page.get_contents().get_data()
    resources = page['/Resources']
    # Give every overlay resource a fresh name, avoiding merge_page's rewriting
    # and rounding of the original drawing stream's numeric coordinates.
    for group, values in overlay_page['/Resources'].items():
        if group == '/ProcSet':
            continue
        dictionary = values.get_object()
        if not isinstance(dictionary, DictionaryObject):
            raise ValueError('Unexpected overlay resource type: ' + group)
        if group not in resources:
            resources[NameObject(group)] = DictionaryObject()
        target = resources[group].get_object()
        for name, value in dictionary.items():
            replacement = '/SeedHeader' + group[1:] + name[1:]
            if replacement in target:
                raise ValueError('Overlay resource collision')
            target[NameObject(replacement)] = value.clone(writer)
            overlay_data = re.sub(re.escape(name.encode('ascii')) + rb'(?=\s)',
                                  replacement.encode('ascii'), overlay_data)
    base_stream, header_stream = DecodedStreamObject(), DecodedStreamObject()
    base_stream.set_data(cleaned)
    header_stream.set_data(b'q\n' + overlay_data + b'\nQ\n')
    page[NameObject('/Contents')] = ArrayObject([writer._add_object(base_stream), writer._add_object(header_stream)])
    page[NameObject('/MediaBox')] = ArrayObject([ExactBoxFloat(v) for v in reader.pages[0].mediabox])
    writer.add_metadata({'/Creator': 'relabel_heatmap.py', '/Producer': 'pypdf'})
    args.output_dir.mkdir(parents=True)
    target = args.output_dir / 'fig_tpos_recurrence_heatmap.pdf'
    with target.open('xb') as handle:
        writer.write(handle)
    check = PdfReader(target).pages[0]
    if check['/Contents'][0].get_object().get_data() != cleaned:
        raise ValueError('Base drawing stream changed during PDF serialization')
    report = dict(status='passed', removed_headers=recipe['old_labels'],
        new_headers=recipe['new_labels'], base_sha256=sha(base), overlay_sha256=sha(overlay),
        pdf_sha256=sha(target), base_drawing_stream_preserved_except_seed_text_blocks=True,
        matrix_redrawn=False, data_or_counts_recomputed=False)
    (args.output_dir / 'relabel_report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
