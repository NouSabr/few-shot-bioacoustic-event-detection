#!/usr/bin/env python3
"""Render saved recall summaries without recomputing metrics.

SVG and vector PDF use only the Python standard library. Optional PNG previews
use an already installed Pillow. Shapes and data are shared across formats.
"""
from __future__ import annotations

import argparse
import csv
from html import escape
from pathlib import Path

BLUE = '#1764a0'
ORANGE = '#c15312'
DARK = '#20252b'
GRID = '#d8dfe5'
CONFIGURATIONS = ('baseline', 'cife', 'phase1_random', 'slot_uniform',
                  'phase2_energy', 'phase3_activation', 'afri_direct', 'afri_inverse')
LABELS = ('Baseline', 'CIFE-inspired', 'Random', 'Slot-uniform',
          'Energy', 'Activation', 'AFRI direct', 'AFRI inverse')


def rgb(color):
    return tuple(int(color[i:i+2], 16)/255 for i in (1, 3, 5))


def number(value):
    return f'{value:.5f}'.rstrip('0').rstrip('.') or '0'


class Canvas:
    def __init__(self, width, height, title):
        self.width, self.height, self.title = width, height, title
        self.shapes = []
        self.rect(0, 0, width, height, '#ffffff')

    def line(self, x1, y1, x2, y2, color=DARK, width=1):
        self.shapes.append(('line', x1, y1, x2, y2, color, width))

    def rect(self, x, y, width, height, fill, stroke=None, line_width=1):
        self.shapes.append(('rect', x, y, width, height, fill, stroke, line_width))

    def polyline(self, points, color, width=2):
        self.shapes.append(('polyline', points, color, width))

    def text(self, x, y, text, size=16, anchor='start', color=DARK):
        self.shapes.append(('text', x, y, str(text), size, anchor, color))

    def save(self, stem, png=False):
        for suffix in ('.svg', '.pdf') + (('.png',) if png else ()):
            if stem.with_suffix(suffix).exists():
                raise ValueError(f'Refusing to overwrite figure: {stem.with_suffix(suffix)}')
        self.save_svg(stem.with_suffix('.svg'))
        self.save_pdf(stem.with_suffix('.pdf'))
        if png:
            self.save_png(stem.with_suffix('.png'))

    def save_svg(self, path):
        lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}" role="img">',
                 f'<title>{escape(self.title)}</title>',
                 '<g font-family="Arial, Helvetica, sans-serif">']
        for shape in self.shapes:
            kind, *s = shape
            if kind == 'line':
                x1,y1,x2,y2,color,width = s
                lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>')
            elif kind == 'rect':
                x,y,w,h,fill,stroke,lw = s
                lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill or "none"}" stroke="{stroke or "none"}" stroke-width="{lw}"/>')
            elif kind == 'polyline':
                points,color,width = s
                points_text = ' '.join(f'{number(x)},{number(y)}' for x,y in points)
                lines.append(f'<polyline points="{points_text}" fill="none" stroke="{color}" stroke-width="{width}"/>')
            elif kind == 'text':
                x,y,text,size,anchor,color = s
                lines.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}">{escape(text)}</text>')
        lines.extend(['</g>', '</svg>', ''])
        path.write_text('\n'.join(lines), encoding='utf-8')

    def save_pdf(self, path):
        def color_command(color, operator):
            return ' '.join(number(x) for x in rgb(color)) + ' ' + operator
        lines = []
        for shape in self.shapes:
            kind, *s = shape
            if kind == 'line':
                x1,y1,x2,y2,color,width = s
                lines += [color_command(color, 'RG'), f'{number(width)} w',
                          f'{number(x1)} {number(self.height-y1)} m {number(x2)} {number(self.height-y2)} l S']
            elif kind == 'rect':
                x,y,w,h,fill,stroke,lw = s
                if fill: lines.append(color_command(fill, 'rg'))
                if stroke: lines += [color_command(stroke, 'RG'), f'{number(lw)} w']
                operation = 'B' if fill and stroke else 'f' if fill else 'S'
                lines.append(f'{number(x)} {number(self.height-y-h)} {number(w)} {number(h)} re {operation}')
            elif kind == 'polyline':
                points,color,width = s
                lines += [color_command(color, 'RG'), f'{number(width)} w']
                lines.append(' '.join(f'{number(x)} {number(self.height-y)} {"m" if i==0 else "l"}' for i,(x,y) in enumerate(points)) + ' S')
            elif kind == 'text':
                x,y,text,size,anchor,color = s
                # Helvetica's exact widths for the glyphs used in these figures.
                widths = {' ':278, '%':889, '-':333, '.':278, ',':278, ':':278, ';':278,
                          '/':278, '(':333, ')':333, '=':584, '+':584}
                widths.update(dict(zip('0123456789', [556]*10)))
                widths.update(dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                    [667,667,722,722,667,611,778,722,278,500,667,556,833,722,778,667,778,722,667,611,722,667,944,667,667,611])))
                widths.update(dict(zip('abcdefghijklmnopqrstuvwxyz',
                    [556,556,500,556,556,278,556,556,222,222,500,222,833,556,556,556,556,333,500,278,556,500,722,500,500,500])))
                length = sum(widths.get(c,556) for c in text) * size/1000
                if anchor == 'middle': x -= length/2
                elif anchor == 'end': x -= length
                safe = text.replace('\\','\\\\').replace('(','\\(').replace(')','\\)')
                lines += [color_command(color,'rg'), f'BT /F1 {size} Tf {number(x)} {number(self.height-y)} Td ({safe}) Tj ET']
        content = ('\n'.join(lines)+'\n').encode('ascii')
        objects = [b'<< /Type /Catalog /Pages 2 0 R >>',
                   b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
                   f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>'.encode(),
                   b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
                   f'<< /Length {len(content)} >>\nstream\n'.encode()+content+b'endstream']
        document = bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
        offsets = [0]
        for i, obj in enumerate(objects,1):
            offsets.append(len(document))
            document.extend(f'{i} 0 obj\n'.encode()+obj+b'\nendobj\n')
        position = len(document)
        document.extend(f'xref\n0 {len(offsets)}\n0000000000 65535 f \n'.encode())
        for offset in offsets[1:]: document.extend(f'{offset:010d} 00000 n \n'.encode())
        document.extend(f'trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{position}\n%%EOF\n'.encode())
        path.write_bytes(document)

    def save_png(self, path):
        from PIL import Image, ImageDraw, ImageFont
        image = Image.new('RGB',(self.width,self.height),'white')
        draw = ImageDraw.Draw(image)
        for shape in self.shapes:
            kind, *s = shape
            if kind == 'line':
                x1,y1,x2,y2,color,width = s
                draw.line((x1,y1,x2,y2),fill=color,width=max(1,round(width)))
            elif kind == 'rect':
                x,y,w,h,fill,stroke,lw = s
                draw.rectangle((x,y,x+w,y+h),fill=fill,outline=stroke,width=max(1,round(lw)))
            elif kind == 'polyline':
                points,color,width = s
                draw.line(points,fill=color,width=max(1,round(width)))
            elif kind == 'text':
                x,y,text,size,anchor,color = s
                try:
                    font = ImageFont.truetype('DejaVuSans.ttf',size)
                except OSError:
                    font = ImageFont.load_default(size=size)
                draw.text((x,y),text,font=font,fill=color,
                          anchor={'start':'ls','middle':'ms','end':'rs'}[anchor])
        image.save(path)


def rows(path):
    with path.open(newline='',encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def render_figures(input_dir, output_dir, png=False):
    ecdf = rows(input_dir/'recall_ecdf.csv')
    fig = Canvas(960,620,'Recording recall before and after label-informed deletion')
    fig.text(480,35,fig.title,23,'middle')
    fig.text(480,64,'24 selected outputs; 42 distinct recording names per output',16,'middle')
    left,right,top,bottom = 100,910,135,505
    for tick in range(0,101,20):
        x = left+(right-left)*tick/100
        y = bottom-(bottom-top)*tick/100
        fig.line(x,top,x,bottom,GRID)
        fig.line(left,y,right,y,GRID)
        fig.text(x,bottom+25,str(tick),16,'middle')
        fig.text(left-12,y+5,str(tick),16,'end')
    fig.text(left,top-16,'Cumulative share (%)',16)
    fig.text((left+right)/2,bottom+58,'Recording recall (%)',18,'middle')
    for stage,color,label,legend_x in [('original',BLUE,'Original predictions',120),
                                        ('deleted',ORANGE,'Label-informed deletion',475)]:
        group = [r for r in ecdf if r['stage']==stage]
        points = [(left,bottom)]
        previous = 0.0
        for row in group:
            x = left+(right-left)*float(row['recall_percent'])/100
            y = bottom-(bottom-top)*float(row['cumulative_percent'])/100
            points += [(x,bottom-(bottom-top)*previous/100),(x,y)]
            previous = float(row['cumulative_percent'])
        fig.polyline(points,color,3)
        fig.line(legend_x,93,legend_x+35,93,color,3)
        fig.text(legend_x+45,99,label,17)
    fig.line(left,top,left,bottom,DARK,1.5)
    fig.line(left,bottom,right,bottom,DARK,1.5)
    fig.text(480,598,'Equal-weight descriptive mixture of 1,008 recording-output pairs; not independent samples.',14,'middle')
    fig.save(output_dir/'recall_ecdf',png)

    data = {(r['configuration'],int(r['seed']),r['stage']):r
            for r in rows(input_dir/'recall_quantiles_by_output.csv')}
    fig = Canvas(1120,740,'Recording-level recall before and after deletion')
    fig.text(560,35,fig.title,23,'middle')
    for x,color,label in [(260,BLUE,'Original'),(540,ORANGE,'Label-informed deletion')]:
        fig.rect(x,85,18,14,color)
        fig.text(x+28,99,label,17)
    for index,(configuration,label) in enumerate(zip(CONFIGURATIONS,LABELS)):
        panel_x = 65+(index%4)*275
        top = 155+(index//4)*285
        bottom = top+190
        width = 225
        fig.text(panel_x+width/2,top-18,label,18,'middle')
        for tick in (0,25,50,75,100):
            y = bottom-190*tick/100
            fig.line(panel_x,y,panel_x+width,y,GRID)
            fig.text(panel_x-10,y+5,str(tick),14,'end')
        fig.line(panel_x,top,panel_x,bottom,DARK)
        fig.line(panel_x,bottom,panel_x+width,bottom,DARK)
        for seed_index,seed in enumerate((1234,2026,3009)):
            center = panel_x+40+seed_index*75
            fig.text(center,bottom+25,str(seed),14,'middle')
            for stage,color,offset in [('original',BLUE,-10),('deleted',ORANGE,10)]:
                row = data[configuration,seed,stage]
                values = {name:bottom-190*float(row[name+'_recall_percent'])/100
                          for name in ('minimum','q1','median','q3','maximum')}
                x = center+offset
                fig.line(x,values['maximum'],x,values['minimum'],color,1.5)
                for endpoint in ('minimum','maximum'):
                    fig.line(x-5,values[endpoint],x+5,values[endpoint],color,1.5)
                fig.rect(x-7,values['q3'],14,values['q1']-values['q3'],'#ffffff',color,2)
                fig.line(x-7,values['median'],x+7,values['median'],color,3)
    fig.text(20,122,'Recall (%)',16)
    fig.text(560,712,'Training seed',17,'middle')
    fig.save(output_dir/'recall_boxplots',png)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True,help='NEW directory; never overwritten')
    parser.add_argument('--png',action='store_true')
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.exists():
        parser.error('Use a new output directory; no figures are overwritten')
    destination.mkdir(parents=True)
    render_figures(args.input_dir.resolve(),destination,png=args.png)


if __name__ == '__main__':
    main()
