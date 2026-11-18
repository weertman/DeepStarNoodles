#!/usr/bin/env python3
"""Render Figure 3: two annotation datasets and held-out arm traces."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'deepstarnoodles-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=HERE/'figure_03_data'
sys.path.insert(0,str(HERE))
import published_models as PM
W,H=1800,1950
INK,MUTED,LINE='#20303c','#586976','#d7dfe3'
TEAL,BLUE,AMBER='#237b76','#35668f','#93651e'
TRACE='#00e4ff'
GROUND_TRUTH='#ed55b4'
FAMILIES=[('starseg','StarSeg color','#4C78A8'),
          ('starseglab','StarSegLab monochrome','#F58518'),
          ('rigs','Sleepy Stars + other rigs','#E45756')]
MOSAIC_COLUMNS={'starseg':4,'starseglab':4,'rigs':2}
STARSEG_LABELS={'sunflower_star':'Sunflower star','other_sea_star':'Other sea stars',
                'live_prey':'Live prey','dead_prey':'Dead prey'}
STARSEG_INVENTORY_LABELS={'sunflower_star':'Sunflower stars','live_prey':'Live prey','dead_prey':'Dead prey'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    record=json.loads((DATA/'prepared.json').read_text())
    assert digest(DATA/'example_images.npz')==record['example_images_sha256']
    for source in record['sources']:
        assert digest(source['path'])==source['sha256'],source['path']
    for sample in record['source_examples']:
        assert digest(ROOT/sample['image_path'])==sample['image_sha256']
        assert digest(ROOT/sample['shapes_path'])==sample['shapes_sha256']
    for sample in record['starseg_examples']:
        assert digest(ROOT/sample['image_path'])==sample['image_sha256']
        assert digest(ROOT/sample['label_path'])==sample['label_sha256']
    images=np.load(DATA/'example_images.npz')
    # C: locked seed-0 test readout of the published dense model and tracer.
    rep=record['locked_readout']
    assert digest(ROOT/rep['rows_path'])==rep['rows_sha256'],'Changed locked readout'
    tracer=record['tracer']
    assert tracer['released'] and tracer['sha256']==digest(PM.TRACER)
    assert record['dense']['sha256']==PM.HYDRA_SHA256==rep['request']['hydra_sha256']
    assert rep['request']['gru_sha256']==tracer['source_checkpoint_sha256']
    assert tracer['invalid_seed_threshold']==rep['request']['invalid_seed_threshold']==0.6
    stored=[json.loads(row) for row in (ROOT/rep['rows_path']).read_text().splitlines() if row.strip()]
    def pooled(rows):
        tp,fp,fn=(sum(row[key] for row in rows) for key in ('tp','fp','fn'))
        return {'images':len(rows),'groups':len({row['group_id'] for row in rows}),'tp':tp,'fp':fp,'fn':fn}
    # Published test score of the pipeline (S1 Text).
    assert pooled(stored)=={'images':553,'groups':pooled(stored)['groups'],'tp':4534,'fp':989,'fn':1151}
    sleepy=rep['sleepy_stars']
    assert pooled([row for row in stored if row['source_family']=='sleepy_stars'])=={
        key:sleepy[key] for key in ('images','groups','tp','fp','fn')}
    by_uid={row['uid']:row for row in stored}
    for example in record['examples']:
        row=by_uid[example['uid']]
        assert row['source_family']=='sleepy_stars' and row['split']=='test'
        assert (row['tp'],row['fp'],row['fn'])==(example['tp'],example['fp'],example['fn'])
        assert row['predictions']==example['tracer_paths'] and row['ground_truth']==example['ground_truth_paths']
    # B: frozen arm-annotation dataset.
    arm=record['arm_dataset']
    arm_total=arm['images']
    group_counts=arm['group_image_counts']
    assert sum(group_counts.values())==arm_total==sum(arm['split_images'].values())
    assert sum(arm['source_family_image_counts'].values())==arm_total
    assert [row['family'] for row in record['source_examples']]==[
        family for family,_,_ in FAMILIES for _ in range(3*MOSAIC_COLUMNS[family])]
    inventory=arm['annotation_inventory']
    assert all(sum(inventory['by_split'][split][key] for split in ('train','validation','test'))==value
               for key,value in inventory['total'].items())
    # A: frozen detector dataset.
    starseg=record['starseg_inventory']
    display=record['starseg_display']
    STARSEG_CLASSES=[(group,STARSEG_LABELS[group],color)
                     for group,color in zip(display['groups'],display['colors'])]
    STARSEG_COLUMNS=tuple(display['columns'])
    assert sum(starseg['split_images'].values())==starseg['images']
    assert sum(starseg['coarse_mask_counts'].values())==starseg['masks']==sum(starseg['fine_mask_counts'].values())
    assert starseg['fine_classes']==37
    assert all(sum(n for name,n in starseg['fine_mask_counts'].items()
                   if starseg['fine_to_coarse'][name]==group)==starseg['coarse_mask_counts'][group]
               for group in display['groups'])
    assert STARSEG_COLUMNS==(4,2,2,2)
    assert [sample['class'] for sample in record['starseg_examples']]==[
        cls for (cls,_,_),ncols in zip(STARSEG_CLASSES,STARSEG_COLUMNS)
        for _ in range(3*ncols)]
    printed=[]   # every number drawn in the figure, with its definition and source
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'svg.fonttype':'none',
                         'pdf.fonttype':42,'svg.hashsalt':'deepstarnoodles-figure-03-v2'})
    fig=plt.figure(figsize=(7.48,7.48*H/W),facecolor='white')
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')

    def text(x,y,value,size=7,color=INK,weight='normal',ha='center',**kw):
        return ax.text(x,y,str(value),fontsize=size,color=color,fontweight=weight,
                       ha=ha,va='center',linespacing=1.3,**kw)

    def line(x1,y1,x2,y2,color=LINE,width=.7):
        ax.plot([x1,x2],[y1,y2],color=color,lw=width,solid_capstyle='round')

    def panel(letter,title,x,y,color):
        text(x,y,letter,10.4,INK,'bold',ha='left')
        text(x+37,y+1,title,7.4,INK,ha='left')   # plain ink; colour only encodes data

    def image_ax(x,y,width,image,gamma=1.0):
        a=fig.add_axes([x/W,1-(y+width)/H,width/W,width/H])
        a.imshow(np.clip(image.astype(float)/255,0,1)**gamma,interpolation='nearest')
        a.set_xticks([]);a.set_yticks([])
        for spine in a.spines.values():spine.set(color=LINE,linewidth=.55)
        return a

    # A: StarSeg images shown with their instance annotations.
    panel('A','StarSeg annotated images',37,39,BLUE)
    starseg_tile,starseg_gap,starseg_group_gap=157,5,15
    starseg_x=77
    starseg_display=[]
    offset=0
    for group,((class_name,label,color),ncols) in enumerate(zip(STARSEG_CLASSES,STARSEG_COLUMNS)):
        group_width=ncols*starseg_tile+(ncols-1)*starseg_gap
        text(starseg_x,96,label,6.3,color,'bold',ha='left')
        text(starseg_x,119,f'n = {starseg["coarse_image_counts"][class_name]:,} images',
             5.8,MUTED,ha='left')
        printed.append({'panel':'A','label':f'{label} · n = {starseg["coarse_image_counts"][class_name]:,} images',
                        'value':starseg['coarse_image_counts'][class_name],
                        'definition':f'Images of the frozen detector dataset (train, val and test) with at least one {class_name} mask; an image with masks of several groups is counted in each.',
                        'source':starseg['dataset']+'/arms/F/labels and manifest.csv'})
        ax.add_patch(Rectangle((starseg_x,131),group_width,3,fc=color,ec='none'))
        for j,sample in enumerate(record['starseg_examples'][offset:offset+3*ncols]):
            row,col=divmod(j,ncols)
            shown=image_ax(starseg_x+col*(starseg_tile+starseg_gap),
                           143+row*(starseg_tile+starseg_gap),starseg_tile,
                           images[sample['mask_array_key']])
            for spine in shown.spines.values():spine.set(color=color,linewidth=.8)
            starseg_display.append({key:sample[key] for key in
                                    ('class','selection','split','image_path','image_sha256','label_path',
                                     'label_sha256','polygon_count','fine_counts','coarse_counts')})
        offset+=3*ncols
        starseg_x+=group_width+starseg_group_gap
    assert offset==30 and starseg_x-starseg_group_gap==1722
    text(51,653,'StarSeg annotation inventory',6.7,INK,'bold',ha='left')
    starseg_metrics=[(f'{starseg["coarse_mask_counts"][group]:,}',
                      f'Other sea stars · {starseg["named_other_sea_star_species"]} named species'
                      if group=='other_sea_star' else STARSEG_INVENTORY_LABELS[group])
                     for group in display['groups']]
    for i,(value,label) in enumerate(starseg_metrics):
        left=52+i*430
        text(left,698,value,9.1,INK,'bold',ha='left')
        text(left,730,label,6.0,MUTED,ha='left')
        printed.append({'panel':'A','label':f'{value} {label}','value':starseg['coarse_mask_counts'][display['groups'][i]],
                        'definition':f'Instance masks (polygons) of the {display["groups"][i]} group in all {starseg["images"]:,} images of the frozen detector dataset.',
                        'source':starseg['dataset']+'/arms/F/labels with label_maps.json fine_to_coarse; total equals summary.json'})
    printed.append({'panel':'A','label':f'{starseg["named_other_sea_star_species"]} named species','value':starseg['named_other_sea_star_species'],
                    'definition':'Fine classes of the other-sea-star group with at least one mask, excluding Unknown_species.',
                    'source':starseg['dataset']+'/arms/F/labels and label_maps.json'})

    # B: DeepStarNoodles examples grouped by source appearance.
    panel('B','DeepStarNoodles annotated images',37,798,TEAL)
    sample_display=[]
    x=77
    tile,gap,group_gap=157,5,20
    for family,label,color in FAMILIES:
        ncols=MOSAIC_COLUMNS[family]
        group_width=ncols*tile+(ncols-1)*gap
        n=group_counts[family]
        header_text=text(x,855,label,6.3,color,'bold',ha='left')
        assert header_text.get_window_extent(fig.canvas.get_renderer()).width*W/fig.bbox.width<=group_width,label
        count_label=f'n = {n:,} ({n/arm_total:.1%})'
        printed.append({'panel':'B','label':f'{label} · {count_label}','value':n,'percent_of':arm_total,
                        'definition':'Images of the frozen arm-annotation dataset (train, validation and test) in this display group: '
                                     +json.dumps(arm['group_source_families'][family])+'. '+arm['classification']['rule'],
                        'source':arm['dataset']+'/splits/{train,validation,test}_index.csv'})
        text(x,878,count_label,5.8,MUTED,ha='left')
        ax.add_patch(Rectangle((x,890),group_width,3,fc=color,ec='none'))
        samples=[s for s in record['source_examples'] if s['family']==family]
        for j,sample in enumerate(samples):
            row,col=divmod(j,ncols)
            shown=image_ax(x+col*(tile+gap),902+row*(tile+gap),tile,
                           images[sample['image_array_key']])
            for spine in shown.spines.values():spine.set(color=color,linewidth=.8)
            sample_display.append({'family':family,'image_path':sample['image_path'],
                                   'image_sha256':sample['image_sha256'],
                                   'shapes_path':sample['shapes_path'],
                                   'shapes_sha256':sample['shapes_sha256'],
                                   'split':sample['split'],
                                   'source_family':sample['source_family'],
                                   'split_group_id':sample['split_group_id'],
                                   'selection':sample['selection']})
        x+=group_width+group_gap
    assert x-group_gap==1722
    text(51,1405,'DeepStarNoodles annotation inventory',6.7,INK,'bold',ha='left')
    metrics=[('Arm polylines','arm_polylines'),('Mouth points','mouth_points'),
             ('Madreporite points','madreporite_points'),('Anus points','anus_points')]
    for i,(label,key) in enumerate(metrics):
        left=52+i*430
        text(left,1450,f'{inventory["total"][key]:,}',9.1,INK,'bold',ha='left')
        text(left,1485,label,6.0,MUTED,ha='left')
        printed.append({'panel':'B','label':f'{inventory["total"][key]:,} {label}','value':inventory['total'][key],
                        'definition':f'Sum of the {inventory["columns"][key]} column over all {arm_total:,} images of the frozen arm-annotation dataset.',
                        'source':arm['dataset']+'/splits/{train,validation,test}_index.csv'})

    # C: six examples in a single row, each with ground truth and GRU paths.
    panel('C','Held-out arm traces',37,1553,AMBER)
    metric=sleepy
    tp,fp,fn=(metric[k] for k in ('tp','fp','fn'))
    pooled_f1=2*tp/(2*tp+fp+fn)
    assert abs(pooled_f1-metric['pooled_f1'])<1e-12 and tp+fn==metric['annotated_arms']
    header=f'Sleepy Stars test set · n = {metric["images"]} images, {tp+fn} arms · pooled F1 = {pooled_f1:.3f}'
    text(51,1593,header,6.5,MUTED,ha='left')
    for label,value,definition in (
            (f'n = {metric["images"]} images',metric['images'],'Sleepy Stars images of the frozen test split.'),
            (f'{tp+fn} arms',tp+fn,'Annotated arms in those images (TP + FN).'),
            (f'pooled F1 = {pooled_f1:.3f}',pooled_f1,
             f'2TP / (2TP + FP + FN) with TP {tp}, FP {fp}, FN {fn} summed over those images; '+rep['match_rule']+
             '. Published dense model and tracer, seed 0, invalid-seed threshold 0.6.')):
        printed.append({'panel':'C','label':label,'value':value,'definition':definition,'source':rep['rows_path']})
    # Legend right-aligned to the image row's right edge (x = 1716), entries laid out right to left.
    renderer=fig.canvas.get_renderer()
    units_per_px=W/fig.bbox.width
    right=74+5*275+267
    for color,label in ((TRACE,'Noodle Tracer'),(GROUND_TRUTH,'Ground truth')):
        label_text=text(right,1593,label,5.8,INK,ha='right')
        left=right-label_text.get_window_extent(renderer).width*units_per_px
        line(left-46,1593,left-9,1593,color=color,width=2.5)
        right=left-46-40
    example_display=[]
    group_titles=('Spread','Curled at corner','Along side wall')
    for group,title in enumerate(group_titles):
        text(345+group*550,1634,title,6.8,INK,'bold')
    for i,example in enumerate(record['examples']):
        x=74+i*275
        y=1653
        shown=image_ax(x,y,267,images[example['image_array_key']],gamma=.55)
        for path in example['ground_truth_paths']:
            xy=np.asarray(path)
            shown.plot(xy[:,0],xy[:,1],color='#17212a',lw=2.4,
                       solid_capstyle='round',zorder=2)
            shown.plot(xy[:,0],xy[:,1],color=GROUND_TRUTH,lw=1.5,
                       solid_capstyle='round',zorder=3)
        for path in example['tracer_paths']:
            xy=np.asarray(path)
            shown.plot(xy[:,0],xy[:,1],color='#17212a',lw=1.6,
                       solid_capstyle='round',zorder=4)
            shown.plot(xy[:,0],xy[:,1],color=TRACE,lw=0.9,
                       solid_capstyle='round',zorder=5)
        example_display.append({'role':example['role'],'uid':example['uid'],'image_path':example['image_path'],
                                'tp':example['tp'],'fp':example['fp'],'fn':example['fn'],
                                'image_f1':example['image_f1'],
                                'split_group_id':example['split_group_id']})

    base=HERE/'figure_03_training_and_validation'
    fig.savefig(base.with_suffix('.pdf'),dpi=300,metadata={
        'Title':'StarSeg and DeepStarNoodles datasets with held-out arm traces',
        'Creator':'build_figure_03.py',
        'Subject':'Frozen detector and arm-annotation datasets with locked Sleepy Stars test traces of the published models'})
    fig.savefig(base.with_suffix('.svg'),dpi=300)
    fig.savefig(base.with_suffix('.png'),dpi=300)
    plt.close(fig)
    provenance={
        'figure':3,'status':'published models and frozen datasets',
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'prepared_record':str((DATA/'prepared.json').relative_to(ROOT)),
        'prepared_record_sha256':digest(DATA/'prepared.json'),
        'build_script_sha256':digest(__file__),
        'size_inches':[7.48,7.48*H/W],
        'display':{'source':'A shows images of the frozen detector dataset with human masks coloured by four display groups, and the mask count per group (the published detector is trained on two classes, sunflower star and not sunflower); B shows the frozen arm-annotation dataset grouped by source appearance; C shows six held-out Sleepy Stars test crops with the arm paths stored in the locked test readout of the published dense model and tracer.',
                   'starseg_inventory':starseg,
                   'starseg_display':display,
                   'starseg_selection_rule':record['starseg_selection_rule'],
                   'starseg_examples':starseg_display,
                   'arm_dataset':arm,
                   'source_selection_rule':record['source_selection_rule'],
                   'source_examples':sample_display,
                   'tracer':tracer,
                   'dense':record['dense'],
                   'locked_test':{'rows':rep['rows_path'],'rows_sha256':rep['rows_sha256'],
                                  'request':rep['request'],'all_test':rep['all_test'],
                                  'sleepy_stars':sleepy,'match_rule':rep['match_rule']},
                   'examples':example_display,'example_rgb_gamma':.55,
                   'traces':'Six held-out test RGB crops with every ground-truth path in magenta and every stored tracer path in cyan.'},
        'outputs':{kind:{'path':str(base.with_suffix('.'+kind).relative_to(ROOT)),
                          'sha256':digest(base.with_suffix('.'+kind))}
                   for kind in ('pdf','svg','png')},
    }
    numbers={
        'figure':3,
        'generated_at_utc':provenance['generated_at_utc'],
        'models':{'dense_model':record['dense'],
                  'tracer':{'path':tracer['path'],'sha256':tracer['sha256'],
                            'invalid_seed_threshold':tracer['invalid_seed_threshold']},
                  'detector':'not run for this figure; panel A shows human annotations only'},
        'datasets':{'detector':starseg['dataset'],'arm_annotations':arm['dataset'],'locked_test_rows':rep['rows_path']},
        'printed':printed,
        'not_printed':{
            'detector_dataset_images':starseg['images'],'detector_dataset_masks':starseg['masks'],
            'detector_dataset_images_without_masks':starseg['images_without_masks'],
            'arm_dataset_images':arm_total,'arm_dataset_split_images':arm['split_images'],
            'arm_dataset_source_family_images':arm['source_family_image_counts'],
            'sleepy_stars_test':sleepy,'all_test':rep['all_test'],
            'panel_c_examples':[{key:example[key] for key in ('role','uid','tp','fp','fn')}
                                for example in record['examples']]},
    }
    (HERE/'figure_03_numbers.json').write_text(json.dumps(numbers,indent=2)+'\n')
    base.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Built Figure 3 PDF, SVG, PNG, provenance and numbers.')
    for item in printed:print(' ',item['panel'],item['label'])


if __name__=='__main__':build()
