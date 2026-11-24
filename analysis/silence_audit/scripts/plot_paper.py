#!/usr/bin/env python3
"""Portable rendering adapter; drawing bodies preserved from approved snapshots."""
import argparse,csv,hashlib,json,math,statistics,sys
from pathlib import Path
import plot_results as drawing
from plot_results import Canvas,BLUE,ORANGE,DARK,GRID,CONFIGURATIONS,LABELS,rows

def require(value,message):
    if not value: raise ValueError(message)

def paired(pairs,means,out,png):
    BLUE,ORANGE,INK,PAIR='#226a9f','#c36a25','#263442','#d5dde3'
    canvas=drawing.Canvas(1000,560,'Before and after label-informed deletion: paired-dot panels')
    canvas.text(500,31,'Before and after label-informed deletion',25,'middle',INK)
    canvas.text(500,60,'24 selected configuration-seed outputs',18,'middle',INK)

    def marker(x,y,shape,color,radius=4.5,width=1.8):
        if shape=='circle':
            points=[(x+radius*math.cos(2*math.pi*i/36),y+radius*math.sin(2*math.pi*i/36)) for i in range(37)]
        else:
            points=[(x,y-radius),(x+radius,y),(x,y+radius),(x-radius,y),(x,y-radius)]
        canvas.polyline(points,color,width)

    # Stage is directly labelled on each x axis; the shared legend explains the lines/means.
    canvas.line(285,91,330,91,PAIR,1.5)
    marker(285,91,'circle','#8493a0',3.5)
    marker(330,91,'circle','#8493a0',3.5)
    canvas.text(342,97,'Individual output',18,'start',INK)
    canvas.line(565,91,610,91,INK,3)
    marker(565,91,'diamond',INK,5,2.2)
    marker(610,91,'diamond',INK,5,2.2)
    canvas.text(622,97,'Mean across outputs',18,'start',INK)

    top,bottom=155,425
    panels=[dict(metric='official',title='(a) Official score',left=70,right=475,
                 lower=14,upper=28,ticks=range(14,29,2),suffix='',change_suffix='points'),
            dict(metric='coverage',title='(b) Recording coverage (%)',left=575,right=980,
                 lower=50,upper=100,ticks=range(50,101,10),suffix='%',change_suffix='percentage points')]
    for panel in panels:
        metric=panel['metric']
        left,right=panel['left'],panel['right']
        before_x,after_x=left+80,right-80
        def y(value):
            require(panel['lower']<=value<=panel['upper'],'Endpoint outside panel limits')
            return bottom-(bottom-top)*(value-panel['lower'])/(panel['upper']-panel['lower'])
        canvas.text(left,135,panel['title'],22,'start',INK)
        for tick in panel['ticks']:
            yy=y(tick)
            canvas.line(left,yy,right,yy,'#e6ebef',1)
            canvas.text(left-12,yy+6,str(tick),18,'end',INK)
        canvas.line(left,top,left,bottom,'#82909b',1.2)
        canvas.line(left,bottom,right,bottom,'#82909b',1.2)
        for row in pairs:
            canvas.line(before_x,y(row['original_'+metric]),after_x,y(row['deleted_'+metric]),PAIR,1.2)
        for row in pairs:
            marker(before_x,y(row['original_'+metric]),'circle',BLUE,4.3)
            marker(after_x,y(row['deleted_'+metric]),'circle',ORANGE,4.3)
        original_mean,deleted_mean=means['original_'+metric],means['deleted_'+metric]
        original_y,deleted_y=y(original_mean),y(deleted_mean)
        canvas.line(before_x,original_y,after_x,deleted_y,INK,3)
        for x,yy in ((before_x,original_y),(after_x,deleted_y)):
            marker(x,yy,'diamond','#ffffff',6,5)
            marker(x,yy,'diamond',INK,6,2.5)
        canvas.text(before_x-15,original_y+6,f'{original_mean:.2f}'+panel['suffix'],17,'end',INK)
        canvas.text(after_x+15,deleted_y+6,f'{deleted_mean:.2f}'+panel['suffix'],17,'start',INK)
        canvas.text(before_x,bottom+32,'Original',20,'middle',BLUE)
        canvas.text(after_x,bottom+32,'After deletion',20,'middle',ORANGE)
        change=deleted_mean-original_mean
        canvas.text((left+right)/2,498,f"Mean change: {change:+.2f} {panel['change_suffix']}",18,'middle',INK)
    canvas.text(500,540,'Lines connect the same output; the two panels use separate vertical scales.',17,'middle',INK)
    canvas.save(out/'score_coverage_paired_panels',png=png)

def boxes(input_dir,output_dir,png):
    data = {(r['configuration'],int(r['seed']),r['stage']):r
            for r in rows(input_dir/'recall_quantiles_by_output.csv')}
    fig = Canvas(1120,800,'Per-recording recall distributions for each selected output')
    fig.text(560,35,fig.title,23,'middle')
    fig.text(560,65,'Each box summarizes the same 42 recording names; deleted recordings remain as zeros',16,'middle')
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
    fig.text(560,712,'Training seed (one complete selected output per configuration and seed)',17,'middle')
    fig.text(560,747,'Boxes: middle 50%; center line: median; whiskers: minimum and maximum; n = 42 per box.',15,'middle')
    fig.text(560,777,'Descriptive distributions, not confidence intervals. Shared PB/PB24 name counted once per box.',14,'middle')
    fig.save(output_dir/'recall_boxplots',png)

def workflow(out,png):
    canvas = drawing.Canvas(500, 540, 'Single-column deletion audit workflow: preview')
    ink, line, border = '#243344', '#7b8b9b', '#cbd4dd'
    blue, orange = '#38678f', '#a26a2c'
    text_regions = {}

    def box(x, y, w, h, title, body, fill='#ffffff', accent=None):
        canvas.rect(x, y, w, h, fill, border, 1.1)
        if accent:
            canvas.rect(x, y, w, 3, accent)
        canvas.text(x+w/2, y+23, title, 20, 'middle', ink)
        text_regions[title] = (x, y, w, h)
        for i, label in enumerate(body):
            canvas.text(x+w/2, y+44+i*20, label, 18, 'middle', ink)
            text_regions[label] = (x, y, w, h)

    def path(points, color=line, arrow=False):
        canvas.polyline(points, color, 1.6)
        if arrow:
            x0, y0 = points[-2]
            x, y = points[-1]
            theta = math.atan2(y-y0, x-x0)
            for offset in (-0.48, 0.48):
                canvas.line(x, y, x-7*math.cos(theta+offset), y-7*math.sin(theta+offset), color, 1.6)

    box(52, 8, 396, 56, 'Predictions + reference labels',
        ['24 outputs; settings fixed'], '#f5f8fb', blue)
    path([(250, 64), (250, 86)], arrow=True)
    box(52, 86, 396, 76, 'Match and aggregate',
        ['Recording TP / FP / FN', 'Subset micro-F1; harmonic mean'], '#f7f9fb')
    path([(250, 162), (250, 184)], arrow=True)

    canvas.rect(38, 184, 424, 230, '#ffffff', border, 1.1)
    canvas.text(250, 207, 'Separate deletion analyses', 17, 'middle', line)
    box(54, 220, 392, 78, 'Single-recording deletion',
        ['One recording at a time', 'Official + support-consistent'], accent=blue)
    box(54, 318, 392, 78, 'Joint-deletion bound',
        ['Label-informed selection', 'Maximize official score'], '#fff9f2', orange)

    # Ground-truth labels also enter joint selection directly.
    path([(52, 36), (18, 36), (18, 357), (54, 357)], orange, arrow=True)
    canvas.text(26, 78, 'Labels', 14, 'start', orange)
    path([(250, 414), (250, 436)], arrow=True)
    box(52, 436, 396, 96, 'Before/after comparison',
        ['Single: score changes and gain rate',
         'Joint: official score, recording-macro F1,',
         'coverage and recall'], '#f7f9fb')
    canvas.save(out/'audit_workflow',png=png)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--png',action='store_true')
    args=parser.parse_args()
    require(not args.output_dir.exists(),'Use a new figure directory')
    data=rows(args.audit_dir/'metrics_by_output.csv')
    require(len(data)==24,'Expected 24 calculated outputs')
    pairs=[{field:float(r[field]) for field in ['original_official','deleted_official','original_coverage','deleted_coverage']} for r in data]
    means={field:statistics.mean(r[field] for r in pairs) for field in pairs[0]}
    args.output_dir.mkdir(parents=True)
    paired(pairs,means,args.output_dir,args.png)
    boxes(args.audit_dir,args.output_dir,args.png)
    workflow(args.output_dir,args.png)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output_dir.iterdir()) if p.is_file()}
    (args.output_dir/'figure_manifest.json').write_text(json.dumps(hashes,indent=2,sort_keys=True)+'\n')
    print('Rendered paired panels, recall boxplots, and workflow; no ECDF or manuscript edits.')

if __name__=='__main__':
    try: main()
    except (ValueError,OSError,KeyError,ImportError) as error: sys.exit(f'ERROR: {error}')
