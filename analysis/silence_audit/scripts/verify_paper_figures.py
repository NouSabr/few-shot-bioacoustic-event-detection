#!/usr/bin/env python3
"""Check current-paper PDF hashes and embedded fonts; requires pypdf only."""
import argparse
import hashlib
import json
from pathlib import Path
from pypdf import PdfReader


def fonts(resources, seen=None):
    seen = set() if seen is None else seen
    resources = resources.get_object()
    if id(resources) in seen:
        return []
    seen.add(id(resources))
    result = []
    for ref in resources.get('/Font', {}).get_object().values() if '/Font' in resources else []:
        font = ref.get_object()
        candidates = [font] + [f.get_object() for f in font.get('/DescendantFonts', [])]
        embedded = any(any(key in f.get('/FontDescriptor', {}).get_object()
                           for key in ('/FontFile', '/FontFile2', '/FontFile3'))
                       for f in candidates if '/FontDescriptor' in f)
        if font.get('/Subtype') == '/Type3' and '/CharProcs' in font:
            embedded = True
        result.append(dict(name=str(font.get('/BaseFont', '<Type3>')), embedded=embedded))
    if '/XObject' in resources:
        for ref in resources['/XObject'].get_object().values():
            obj = ref.get_object()
            if obj.get('/Subtype') == '/Form' and '/Resources' in obj:
                result.extend(fonts(obj['/Resources'], seen))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure-dir', type=Path, required=True)
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parents[1]
    expected = json.loads((bundle / 'expected/paper_figure_sha256.json').read_text())
    inventories = {}
    for name, digest in expected.items():
        path = args.figure_dir / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Changed PDF: ' + name)
        reader = PdfReader(str(path))
        if len(reader.pages) != 1:
            raise ValueError('Expected a one-page figure: ' + name)
        inventory = fonts(reader.pages[0]['/Resources'])
        if not inventory or not all(f['embedded'] for f in inventory):
            raise ValueError('Missing/unembedded font: ' + name)
        inventories[name] = inventory
    print(json.dumps(dict(status='passed', figures=len(inventories), fonts=inventories),
                     indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
