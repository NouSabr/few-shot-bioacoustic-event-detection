"""Stable SVG drawing signature; numeric coordinates normalized to 1e-9 point.

The old portable renderer's PDF writer rounds to 1e-5 point; original SVG
floating-point strings can differ below that precision after count replay.
Text, styles, node order and all nonnumeric attributes remain exact.
Final publication PDFs are separately required to match full SHA-256 hashes.
"""
import hashlib
import json
import math
import xml.etree.ElementTree as ET


def svg_signature(path):
    numeric = {'x', 'y', 'x1', 'x2', 'y1', 'y2', 'width', 'height', 'font-size', 'stroke-width'}

    def number(value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('Nonfinite SVG coordinate')
        return format(value, '.9f')

    def node(element):
        attributes = {}
        for key, value in element.attrib.items():
            if key in numeric:
                value = number(value)
            elif key == 'viewBox':
                value = ' '.join(number(v) for v in value.split())
            elif key == 'points':
                value = ' '.join(','.join(number(v) for v in pair.split(',')) for pair in value.split())
            attributes[key] = value
        return [element.tag, attributes, element.text if element.tag.endswith(('text', 'title')) else None,
                [node(child) for child in element]]

    payload = json.dumps(node(ET.parse(path).getroot()), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()
