import hashlib
import xml.etree.ElementTree as ET

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save_figure(fig, stem):
    fig.savefig(stem.with_suffix('.pdf'), metadata={
        'CreationDate': None, 'ModDate': None,
        'Creator': ('generate_vertical_workflow.py' if stem.name == 'audit_workflow' else 'paper_revision/tooling/polish_figures.py'),
    })
    fig.savefig(stem.with_suffix('.png'), dpi=100)

def reexport_svg(source, stem, font_path):
    """Replay the existing Canvas SVG vocabulary without changing its values."""
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Rectangle

    svg = ET.parse(source).getroot()
    width, height = float(svg.attrib['width']), float(svg.attrib['height'])
    if svg.attrib['viewBox'].split() != ['0', '0', str(int(width)), str(int(height))]:
        raise ValueError('Unexpected SVG viewBox')
    fig = plt.figure(figsize=(width / 72, height / 72))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(height, 0))
    ax.set_axis_off()
    text_labels = []
    for order, node in enumerate(svg.iter(), 1):
        kind = node.tag.rsplit('}', 1)[-1]
        a = node.attrib
        f = lambda key: float(a[key])
        if kind in ('svg', 'g', 'title'):
            continue
        common = dict(zorder=order, clip_on=False)
        if kind == 'rect':
            ax.add_patch(Rectangle((f('x'), f('y')), f('width'), f('height'),
                facecolor=a['fill'], edgecolor=a['stroke'],
                linewidth=f('stroke-width'), joinstyle='miter', **common))
        elif kind == 'line':
            ax.plot([f('x1'), f('x2')], [f('y1'), f('y2')], color=a['stroke'],
                    linewidth=f('stroke-width'), solid_capstyle='butt', **common)
        elif kind == 'polyline':
            points = [tuple(map(float, pair.split(','))) for pair in a['points'].split()]
            ax.plot(*zip(*points), color=a['stroke'], linewidth=f('stroke-width'),
                    solid_capstyle='butt', solid_joinstyle='miter', **common)
        elif kind == 'text':
            if list(node):
                raise ValueError('Nested SVG text is not supported')
            label = node.text or ''
            text_labels.append(label)
            ax.text(f('x'), f('y'), label,
                    fontproperties=FontProperties(fname=str(font_path), size=f('font-size')),
                    ha={'start': 'left', 'middle': 'center', 'end': 'right'}[a['text-anchor']],
                    va='baseline', color=a['fill'], **common)
        else:
            raise ValueError('Unsupported SVG element: ' + kind)
    save_figure(fig, stem)
    plt.close(fig)
    return dict(svg_sha256=sha(source), width_pt=width, height_pt=height,
                text_labels=text_labels, vector_coordinates_and_labels_from_svg=True)
