#!/usr/bin/env python3
"""Render Figure 4: full-rate geometry, held-out states and righting times.

Version 15 (2026-11-16): every panel is drawn from the published detector, dense model and
tracer (``published_models.py``; traces through ``trace_sources.py``). A-B read
``figure_04_data/prepared.json``; C-D read the posture readout
(``figure_04_data/readout/readout.json``, ``prepare_figure_04_readout.py``: dense-map summary
features, balanced logistic regression, hidden Markov model); E reads its 24 bouts from
``figure_04_data/aligned_bouts.json`` (``prepare_figure_04_bouts.py``, human labels).
Also writes ``figure_04_numbers.json``: every number printed in the figure or used by its caption.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'deepstarnoodles-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

HERE=Path(__file__).resolve().parent
import sys
sys.path.insert(0,str(HERE))
import published_models as PM   # noqa: E402
DATA=HERE/'figure_04_data'
READOUT=DATA/'readout'/'readout.json'   # prepare_figure_04_readout.py (C-D)
BOUTS=DATA/'aligned_bouts.json'          # prepare_figure_04_bouts.py (E)
INK,MUTED,GRID='#21313b','#60717b','#d8e0e4'
BLUE,AMBER,PURPLE='#287ba9','#d28d26','#81549b'
TEAL,GOLD='#208f88','#d2a73b'
STATES={'Right Side Up':BLUE,'Being flipped':AMBER,'Upside Down':PURPLE}
W_IN,H_IN=7.48,9.40   # one inch grid for every panel


def draw_kinematics(fig,ax_in,text_in,top):
    """Panel E: six flip-aligned measures for the 24 righting bouts, every recorded frame (2 Hz)."""
    details=json.loads((DATA/'kinematics_aligned.json').read_text())
    bouts=json.loads(BOUTS.read_text())
    events=bouts['examples']
    assert digest(DATA/'kinematics_aligned.npz')==details['kinematics_npz_sha256']
    assert len(events)==24
    data=dict(np.load(DATA/'kinematics_aligned.npz'))
    landmark_details=json.loads((DATA/'landmarks_aligned.json').read_text())
    assert digest(DATA/'landmarks_aligned.npz')==landmark_details['landmarks_npz_sha256']
    landmarks=np.load(DATA/'landmarks_aligned.npz')
    np.testing.assert_array_equal(landmarks['event_id'],data['event_id'])
    np.testing.assert_allclose(landmarks['relative_time_s'],data['relative_time_s'])
    events_check={'kinematics':check_event_windows(events,data,details,'kinematics'),
                  'landmarks':check_event_windows(events,dict(landmarks),landmark_details,'landmarks')}
    data['side_index']=landmarks['side_index']
    event_id,time,phase=data['event_id'],data['relative_time_s'],data['phase']
    class_colors=np.asarray((BLUE,AMBER,PURPLE))
    flip_to_up=np.median([e['upside_down_start_s']-e['flip_start_s'] for e in events])
    flip_to_right=np.median([e['right_side_up_start_s']-e['flip_start_s'] for e in events])
    panels=[('side_index','Side facing camera','p(anus) − p(mouth)','side'),
            ('arm_deformation','Arm deformation','R₀ / s, log scale','log'),
            ('fold_asymmetry','Fold asymmetry','tip centroid offset, R₀','linear'),
            ('arm_length_spread','Arm-length spread','IQR of lengths, R₀','linear'),
            ('arm_curvature','Arm curvature','median, rad / R₀','linear'),
            ('support_area','Support area','relative to pre-flip','linear')]
    grid=np.arange(-5,50.5,.5)
    width,gap,x0=.93,.272,.42
    medians={}
    for k,(key,title,unit,scale) in enumerate(panels):
        ax=ax_in(x0+k*(width+gap),top+.32,width,1.0)
        for start,stop,color in ((-5,0,BLUE),(0,flip_to_up,AMBER),
                                 (flip_to_up,flip_to_right,PURPLE),(flip_to_right,50,BLUE)):
            ax.axvspan(start,stop,color=color,alpha=.075,lw=0)
        ax.axvline(0,color=AMBER,lw=.8,ls='--')
        values=data[key]
        matrix=np.full((24,len(grid)),np.nan)
        for index in range(24):
            keep=event_id==index
            t,v,ph=time[keep],values[keep],phase[keep]
            points=np.column_stack((t,v))
            segments=np.stack((points[:-1],points[1:]),axis=1)
            ax.add_collection(LineCollection(segments,colors=class_colors[ph[:-1]],lw=.55,alpha=.3))
            positions=np.round((t+5)*2).astype(int)
            valid=(positions>=0)&(positions<len(grid))
            matrix[index,positions[valid]]=v[valid]
        available=np.isfinite(matrix).sum(axis=0)>=12
        median=np.nanmedian(matrix[:,available],axis=0)
        ax.plot(grid[available],median,color=INK,lw=1.15)
        medians[key]={'time_s':grid[available].tolist(),'median':median.tolist()}
        ax.set_yscale('log' if scale=='log' else 'linear')
        if scale=='side':
            ax.axhline(0,color=MUTED,lw=.5,zorder=1)
            ax.set_ylim(-1.05,1.05);ax.set_yticks([-1,0,1])
            ax.text(.98,.93,'aboral up',transform=ax.transAxes,ha='right',va='top',fontsize=5.8,color=MUTED)
            ax.text(.98,.07,'oral up',transform=ax.transAxes,ha='right',va='bottom',fontsize=5.8,color=MUTED)
        elif scale=='log':
            ax.set_ylim(.004,1);ax.set_yticks([.01,.1,1],['0.01','0.1','1'])
        else:
            low,high=np.nanpercentile(values,[.5,99.5]);pad=(high-low)*.08
            ax.set_ylim(low-pad,high+pad)
        ax.set_xlim(-5,50);ax.set_xticks([0,20,40])
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color(GRID)
        ax.tick_params(length=2.2,labelsize=6.0,pad=1.5)
        ax.set_title(title,loc='left',fontsize=6.8,color=INK,fontweight='bold',pad=10)
        ax.text(0,1.02,unit,transform=ax.transAxes,ha='left',va='bottom',fontsize=5.9,color=MUTED)
        if k==2:ax.set_xlabel('Time from flip onset (s)',fontsize=6.3,labelpad=1)
        ax.xaxis.set_label_coords(1.12 if k==2 else .5,-.22)
    handles=[Line2D([0],[0],color=color,lw=1.6,label=label)
             for color,label in zip(class_colors,('Right side up','Being flipped','Upside down'))]
    handles.append(Line2D([0],[0],color=INK,lw=1.2,label='Median across bouts'))
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.42/W_IN,1-(top+1.62)/H_IN),ncol=4,
               frameon=False,fontsize=6.0,handletextpad=.4,columnspacing=1.2,borderaxespad=0,handlelength=1.5)
    phase_summary={}
    for key,title,unit,_ in panels:
        phase_summary[key]={'title':title,'unit':unit}
        for code,name in enumerate(('right_side_up','being_flipped','upside_down')):
            values=data[key][phase==code]
            finite=np.isfinite(values)
            phase_summary[key][name]={'median':float(np.nanmedian(values)),'n_frames':int(finite.sum()),
                                      'n_frames_missing':int((~finite).sum()),
                                      'n_bouts':int(len(np.unique(event_id[phase==code][finite])))}
    return {'kinematics':details,'landmarks':landmark_details,'medians':medians,
            'n_events':len(events),'n_frames':int(len(event_id)),'phase_summary':phase_summary,
            'righting_time_range_s':bouts['righting_time_range_s'],
            'events_file':{'path':str(BOUTS),'sha256':digest(BOUTS),'selection':bouts['selection']},
            'events_check':events_check,
            'phase_shading_s':{'upside_down_median':float(flip_to_up),'right_side_up_median':float(flip_to_right)}}


def check_event_windows(events,data,details,name):
    """E data must cover exactly each bout's window at 2 Hz, with phases from its human onsets."""
    if details.get('events_json_sha256')==digest(BOUTS):
        status='prepared from aligned_bouts.json'
    else:
        status='prepared from an earlier event file; windows and phases verified identical to aligned_bouts.json'
    for index,event in enumerate(events):
        keep=data['event_id']==index
        t=data['relative_time_s'][keep]
        expected=np.arange(event['first_time_s'],event['last_time_s']+.25,.5)-event['flip_start_s']
        assert len(t)==len(expected) and np.allclose(t,expected),(name,index)
        absolute=t+event['flip_start_s']
        phase=np.where((absolute>=event['flip_start_s'])&(absolute<event['upside_down_start_s']),1,
                       np.where((absolute>=event['upside_down_start_s'])&(absolute<event['right_side_up_start_s']),2,0))
        np.testing.assert_array_equal(data['phase'][keep],phase)
    return status


# Same order as panel E. Colour-bar labels: (low, high) end values.
ATLAS_CHANNELS=[('side_index','Side facing','camera','RdBu_r',(-1,1),('−1','1')),
                ('log_deformation','Arm','deformation','plasma',(-2,-.6),('0.01','0.25')),
                ('fold_asymmetry','Fold','asymmetry','viridis',(0,.5),('0','0.5')),
                ('arm_length_spread','Arm-length','spread','cividis',(0,.9),('0','0.9')),
                ('arm_curvature','Arm','curvature','magma',(0,1),('0','1')),
                ('support_area','Support','area','PuOr',(.6,1.4),('0.6','1.4'))]


def draw_atlas(fig,ax_in,text_in,x0,top,width,height,gap):
    """Every human-labelled righting bout (rows, sorted by duration) in real time, one heatmap per measure."""
    tree=DATA/'tree'
    record=json.loads((tree/'features.json').read_text())
    assert digest(tree/'features.npz')==record['features_npz_sha256']
    data=np.load(tree/'features.npz')
    meta=record['bouts']
    human=json.loads(READOUT.read_text())['durations']['human']
    assert [(m['video_id'],m['start_s'],m['stop_s']) for m in meta]==[
        (h['video_id'],h['start_s'],h['stop_s']) for h in human],'atlas bouts differ from the readout human bouts'
    duration=np.array([m['duration_s'] for m in meta])
    order=np.argsort(duration,kind='stable')
    dt,t0=.5,-record['context_s']
    t1=float(duration.max())+record['context_s']
    cols=int(round((t1-t0)/dt))+1
    for j,(key,line1,line2,cmap,limits,ticks) in enumerate(ATLAS_CHANNELS):
        grid=np.full((len(meta),cols),np.nan)
        for i in range(len(meta)):
            keep=data['bout_id']==i
            c=np.round((data['time_from_start_s'][keep]-t0)/dt).astype(int)
            ok=(c>=0)&(c<cols)
            grid[i,c[ok]]=data[key][keep][ok]
        x=x0+j*(width+gap)
        ax=ax_in(x,top,width,height)
        image=ax.imshow(grid[order],aspect='auto',cmap=cmap,vmin=limits[0],vmax=limits[1],
                        interpolation='nearest',extent=(t0-dt/2,t1+dt/2,len(meta),0),rasterized=True)
        ax.set_xlim(t0-dt/2,t1+dt/2);ax.set_xticks([0,50,100])
        ax.set_yticks([])
        ax.tick_params(axis='x',labelsize=5.9,length=2,pad=1.5,colors=MUTED)
        for spine in ax.spines.values():spine.set(color=GRID,linewidth=.6)
        text_in(x,top-.17,line1,6.0,INK,'bold')
        text_in(x,top-.07,line2,6.0,INK,'bold')
        cax=ax_in(x+.03,top+height+.33,width-.06,.035)
        bar=fig.colorbar(image,cax=cax,orientation='horizontal',ticks=[])
        bar.outline.set_linewidth(.3)
        # End labels sit inside the bar's span so neighbouring channels never collide.
        text_in(x+.03,top+height+.45,ticks[0],5.8,MUTED,ha='left')
        text_in(x+width-.03,top+height+.45,ticks[1],5.8,MUTED,ha='right')
    text_in(x0-.06,top+height/2,f'{len(meta)} bouts, by duration',6.2,INK,ha='right',rotation=90)
    right=x0+6*width+5*gap
    text_in((x0+right)/2,top+height+.21,'Time from upside-down onset (s)',6.3,INK,ha='center')
    return {'tree_features_sha256':record['features_npz_sha256'],'n_bouts':len(meta),
            'n_frames':record['n_frames'],'missing':{k:record['scalars'][k] for k,*_ in ATLAS_CHANNELS},
            'bouts_match_readout_human':True,
            'rows':'sorted by human righting duration (upside-down onset to right-side-up onset)',
            'x_range_s':[t0,t1],'clipped':'none; full time range of every bout window',
            'channels':[{'key':k,'cmap':c,'limits':list(l)} for k,_,_,c,l,_ in ATLAS_CHANNELS]}


def models_record()->dict:
    """The three published models; nothing else enters this figure except the analysis classifier of C-D."""
    return {'detector':{'path':str(PM.DETECTOR),'sha256':PM.DETECTOR_SHA256},
            'dense':{'path':str(PM.HYDRA),'sha256':PM.HYDRA_SHA256,'weights':'EMA (stored under "model")'},
            'tracer':{'path':str(PM.TRACER),'sha256':PM.TRACER_SHA256,'invalid_seed_threshold':PM.INVALID_SEED_THRESHOLD},
            'analysis_classifier':'C-D posture readout: logistic regression and hidden Markov model fitted for this figure '
                                  '(prepare_figure_04_readout.py); an analysis classifier, not a released model'}


def numbers_record(record,readout,primary,scores,durations,analysis,human,machine,atlas,kinematics)->dict:
    """Every number printed in the figure or used by its caption, each with a definition."""
    rel=lambda path:str(Path(path).relative_to(HERE.parents[2])) if str(path).startswith(str(HERE.parents[2])) else str(path)
    src_prepared=rel(DATA/'prepared.json');src_readout=rel(READOUT)
    src_bouts=rel(BOUTS);src_kin=rel(DATA/'kinematics_aligned.npz');src_land=rel(DATA/'landmarks_aligned.npz')
    src_tree=rel(DATA/'tree'/'features.npz')
    items=[]
    def add(panel,label,value,definition,source):
        items.append({'panel':panel,'label':label,'value':value,'definition':definition,'source':source})
    stat=lambda key:{'mean':float(np.mean([row[key] for row in scores])),
                     'sd':float(np.std([row[key] for row in scores],ddof=1)),
                     'by_seed':{str(row['seed']):float(row[key]) for row in scores}}
    # A
    selected=record['selected_frames']
    add('A','frame times',[row['time_s'] for row in selected],
        'time from clip start (frame / 2 frames per second) of the six displayed frames; frames '+
        ', '.join(str(row['frame']) for row in selected),src_prepared)
    add('A','frame states',[row['state'] for row in selected],'human posture label of each displayed frame',src_prepared)
    add('A','traced arms per displayed frame',[row['n_arms'] for row in selected],
        'number of arm paths returned by the published tracer on each displayed crop',src_prepared)
    # B
    timeline=record['timeline'];paths=record['skeleton']
    path_frames=sorted({p['frame'] for p in paths})
    with_star=[row for row in timeline if row['has_star']]
    add('B','arm-path slices',{'frames_with_paths':len(path_frames),'first_time_s':min(p['time_s'] for p in paths),
                               'last_time_s':max(p['time_s'] for p in paths),'arm_paths':len(paths)},
        'recorded frames (2 per second) whose traced arm paths are drawn, in body radii about the body origin of the traces',src_prepared)
    add('B','time series window',{'frames':len(timeline),'first_time_s':timeline[0]['time_s'],'last_time_s':timeline[-1]['time_s'],
                                  'frames_with_star':len(with_star),
                                  'times_without_star_s':[row['time_s'] for row in timeline if not row['has_star']]},
        'recorded frames in the four time series; a frame without a star is one in which the published detector found no star (drawn as a gap)',src_prepared)
    add('B','shaded interval',[370.0,375.0],'human Being flipped annotation of the example flip, seconds from clip start',src_prepared)
    add('B','support area range',[min(r['support_area_model_px'] for r in with_star),max(r['support_area_model_px'] for r in with_star)],
        'arm-support area (support probability > 0.5) in 512 x 512 model pixels, minimum and maximum over the window',src_prepared)
    add('B','recovered arms range',[min(r['n_arms'] for r in with_star),max(r['n_arms'] for r in with_star)],
        'number of traced arms per frame, minimum and maximum over the frames with a star',src_prepared)
    # C
    anchors=readout['anchors']
    add('C','clips · seeds',{'clips':anchors['clips'],'seeds':len(scores)},
        'self-righting clips scored and classifier seeds',src_readout)
    add('C','one-second anchors',{'n':anchors['n'],'animals':anchors['animals'],'sessions':anchors['sessions'],
                                  'by_human_label':anchors['label_counts']},
        'scored time points: '+anchors['rule'],src_readout)
    add('C','folds',[{'test_session':f['test_session'],'train_sessions':f['train_sessions'],
                      'left_out_unused_session':f['left_out_unused_session']} for f in readout['folds']],
        readout['method']['folds'],src_readout)
    add('C','anchors per fold',[{'fold':f['fold'],'train_anchors':f['train_anchors'],'test_anchors':f['test_anchors'],
                                 'balanced_training_rows':f['balanced_training_rows']}
                                for f in readout['fits'] if f['seed']==scores[0]['seed']],
        'training anchors available, anchors used in the balanced fit and test anchors of each fold (same for every seed)',src_readout)
    add('C','Flip episodes F1',stat('event_f1'),
        'episode F1: Being-flipped intervals matched one-to-one within five anchors, F1 per clip, mean over clips; mean and SD (n-1) over seeds; the bar label is the mean',src_readout)
    add('C','Posture states F1',stat('macro_f1'),
        'macro-F1: unweighted mean of the three per-anchor class F1 values over all anchors; mean and SD (n-1) over seeds; the bar label is the mean',src_readout)
    add('C','per-class F1',{label:{'mean':float(np.mean([row['class_f1'][label] for row in scores])),
                                   'sd':float(np.std([row['class_f1'][label] for row in scores],ddof=1)),
                                   'by_seed':{str(row['seed']):row['class_f1'][label] for row in scores}}
                            for label in scores[0]['class_f1']},
        'per-anchor F1 of each posture class over all anchors; mean and SD (n-1) over seeds',src_readout)
    unsmoothed=readout['scores']['session_oof'][primary.replace('_hmm','_none')]
    add('C','without the hidden Markov model',{'event_f1':{k:unsmoothed['event_f1'][k] for k in ('mean','sd')},
                                               'macro_f1':{k:unsmoothed['macro_f1'][k] for k in ('mean','sd')}},
        'same classifier, arg-max of the posteriors without smoothing; not drawn',src_readout)
    example=readout['example']
    add('C','example clip',{'video_id':example['video_id'],'seed':example['seed'],'anchors':len(example['rows']),
                            'first_anchor_s':example['rows'][0]['timestamp'],'last_anchor_s':example['rows'][-1]['timestamp']},
        'clip of the human and model state bars; the bars span the first anchor to one second after the last',src_readout)
    # D
    quart=lambda v:{'median':float(np.median(v)),'q1':float(np.percentile(v,25)),'q3':float(np.percentile(v,75)),
                    'min':float(np.min(v)),'max':float(np.max(v)),'n':int(len(v))}
    add('D','Human n',quart(human),'complete human-labelled self-righting bouts and their righting time (s): '+durations['definition'],src_readout)
    add('D','Model n',quart(machine),f"complete self-righting bouts in the model states, seed {durations['display_seed']}, and their righting time (s)",src_readout)
    add('D','Matched bouts n',{'n_matched':analysis['n_matched'],'animals':analysis['n_animals'],
                               'human_unmatched':analysis['n_human_unmatched'],'model_unmatched':analysis['n_machine_unmatched']},
        analysis['matching']+f" Seed {durations['display_seed']}.",src_readout)
    add('D','matched righting-time error',{'mean_absolute_error_s':analysis['mean_absolute_error_s'],
                                           'bias_model_minus_human_s':analysis['mean_difference_s'],
                                           'bias_ci95_s':analysis['mean_difference_ci95_s'],
                                           'median_difference_s':analysis['median_difference_s']},
        'model minus human righting time over matched pairs; 95% interval from 10,000 animal-cluster bootstrap resamples; display seed',src_readout)
    add('D','over seeds',{key:{k:value[k] for k in ('mean','sd','values')} for key,value in durations['summary_over_seeds'].items()},
        'mean and SD (n-1) over the three classifier seeds of the bout counts and matched errors',src_readout)
    add('D','atlas',{'bouts':atlas['n_bouts'],'frames':atlas['n_frames'],'x_range_s':atlas['x_range_s'],
                     'missing_frames_per_measure':atlas['missing']},
        'all human-labelled bouts at every recorded frame from 2 s before upside-down onset to 2 s after right-side-up onset; rows sorted by righting time',src_tree)
    add('D','atlas colour ranges',{c['key']:c['limits'] for c in atlas['channels']},
        'colour limits of the six heat maps; arm deformation limits are log10 of R0 per second (0.01 to 0.25)',rel(Path(__file__)))
    # data used by all panels
    hydra=json.loads((DATA/'hydra_pass.json').read_text());agree=hydra['agreement_with_inference_records']
    add('all','clips and frames',hydra['totals'],
        'the 44 self-righting clips: recorded frames, frames in which the published traces hold a star, stars detected by the '
        'published detector, frames with no detection and frames with more than one detection',rel(DATA/'hydra_pass.json'))
    add('all','dense-model pass against the inference records',
        {'frames':agree['chosen_star_per_frame']['mouth_peak_probability']['n'],
         'landmark_peak_probability_max_abs_difference':max(agree['chosen_star_per_frame'][f'{name}_peak_probability']['max']
                                                          for name in ('mouth','madreporite','anus')),
         'support_area_px':agree['chosen_star_per_frame']['support_area_px'],
         'gpu_against_numpy_reference_max_relative_error':hydra['geometry70']['gpu_against_numpy_reference_max_relative_error']},
        agree['definition']+'; star of the published traces in each frame',rel(DATA/'hydra_pass.json'))
    # E
    shading=kinematics['phase_shading_s']
    add('E','candidate bouts',json.loads(BOUTS.read_text())['candidates_all_clips'],
        'complete bouts in all clips that satisfy the selection rule before one is chosen per clip',src_bouts)
    add('E','aligned flips',{'bouts':kinematics['n_events'],'frames':kinematics['n_frames'],
                             'missing_frames_per_measure':{**kinematics['kinematics']['n_missing'],
                                                           'side_index':kinematics['landmarks']['n_frames_without_star']},
                             'righting_time_range_s':kinematics['righting_time_range_s']},
        'bouts chosen by prepare_figure_04_bouts.py: '+kinematics['events_file']['selection'],src_bouts)
    add('E','phase shading',{'flip_to_upside_down_median_s':shading['upside_down_median'],
                             'flip_to_right_side_up_median_s':shading['right_side_up_median']},
        'median over the bouts of the human upside-down onset and right-side-up onset, seconds from flip onset',src_bouts)
    for key,row in kinematics['phase_summary'].items():
        add('E',f"{row['title']} ({row['unit']})",{name:row[name] for name in ('right_side_up','being_flipped','upside_down')},
            'median over all frames of the bouts in each human-labelled phase, with the number of frames with a value, frames without one and bouts',
            src_land if key=='side_index' else src_kin)
    return {'figure':4,'generated_at_utc':datetime.now(timezone.utc).isoformat(),
            'models':models_record(),
            'example':{'clip':record['example_clip'],
                       'video':next(s['path'] for s in record['source_hashes'] if s['path'].endswith('.mp4')),
                       'animal':'juvenile i2, session R2_W1_12C','health':'healthy (confirmed by the authors, 2026-11-17)'},
            'analysis_classifier':{'statement':'The posture readout of C-D is an analysis classifier fitted for this figure, not a released model.',
                                   **readout['method']},
            'printed':items}


def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def machine_intervals(rows:list[dict])->list[tuple[float,float,str]]:
    """Run-length encode scored labels without bridging missing anchors."""
    spans=[]
    if not rows:return spans
    start=float(rows[0]['timestamp']);previous=float(rows[0]['timestamp'])
    state=rows[0]['prediction']
    for row in rows[1:]:
        current=float(row['timestamp'])
        if row['prediction']!=state or current-previous>1.01:
            spans.append((start,previous+1,state))
            start=current;state=row['prediction']
        previous=current
    spans.append((start,previous+1,state))
    return spans


def build(preview:bool=False)->None:
    record=json.loads((DATA/'prepared.json').read_text())
    assert record['version']==7 and digest(DATA/'images.npz')==record['images_sha256']
    assert record['checkpoint_hashes']=={'detector':PM.DETECTOR_SHA256,'dense':PM.HYDRA_SHA256,'recurrent':PM.TRACER_SHA256}
    for source in record['source_hashes']:
        assert digest(Path(source['path']))==source['sha256'],source['path']
    readout=json.loads(READOUT.read_text())
    assert digest(DATA/'readout'/'oof_predictions.parquet')==readout['oof_predictions_parquet_sha256']
    assert readout['models']['dense_sha256']==PM.HYDRA_SHA256 and readout['models']['detector_sha256']==PM.DETECTOR_SHA256
    primary=readout['primary_variant']
    arrays=np.load(DATA/'images.npz')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
                         'svg.fonttype':'none','pdf.fonttype':42,
                         'axes.labelcolor':INK,'xtick.color':MUTED,
                         'ytick.color':MUTED})
    fig=plt.figure(figsize=(W_IN,H_IN),facecolor='white')

    def ax_in(x,y,w,h,**kw):
        """Axes from inches, y measured down from the top of the page."""
        return fig.add_axes([x/W_IN,1-(y+h)/H_IN,w/W_IN,h/H_IN],**kw)

    def text_in(x,y,value,size=7,color=INK,weight='normal',ha='left',**kw):
        return fig.text(x/W_IN,1-y/H_IN,value,fontsize=size,color=color,fontweight=weight,
                        ha=ha,va='center',**kw)

    def heading(letter:str,title:str,y:float,color:str|None=None)->None:
        # Titles are plain ink; colour is reserved for data encodings (posture states).
        text_in(.08,y,letter,10.5,INK,'bold')
        text_in(.30,y-.005,title,7.4,INK)

    def simple_axes(ax,axis:str='y'):
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color(GRID)
        ax.tick_params(length=2.4,labelsize=6.2,pad=2)
        ax.grid(axis=axis,color=GRID,lw=.55,zorder=0)

    # A: image row with recurrent paths only.
    heading('A','One flip, frame by frame',.15,BLUE)
    tile,tile_gap=.98,.21
    for i,row in enumerate(record['selected_frames']):
        x=.30+i*(tile+tile_gap)
        ax=ax_in(x,.42,tile,tile)
        ax.imshow(arrays[f"overlay_{row['frame']}"],interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set(color=STATES[row['state']],linewidth=1)
        state={'Right Side Up':'Right side up','Being flipped':'Being flipped',
               'Upside Down':'Upside down'}[row['state']]
        text_in(x+tile/2,.33,f"{row['time_s']:.1f} s",7.0,STATES[row['state']],'bold',ha='center')
        text_in(x+tile/2,.42+tile+.10,state,6.2,STATES[row['state']],ha='center')

    # B: 2-D arm geometry is in a vertical plane; time runs left to right.
    b_top=1.72
    heading('B','Arm geometry through the flip',b_top,TEAL)
    ax3=ax_in(.05,b_top+.08,3.95,1.95,projection='3d')
    ax3.set_proj_type('ortho')
    paths=record['skeleton']
    state_by_time={row['time_s']:row['state'] for row in record['timeline']}
    tmin=min(p['time_s'] for p in paths);tmax=max(p['time_s'] for p in paths)
    for path in paths:
        xy=np.asarray(path['xy_body_radii'])
        color=STATES[state_by_time[path['time_s']]]
        ax3.plot(np.full(len(xy),path['time_s']),xy[:,0],xy[:,1],
                 color=color,lw=.88,alpha=.72)
    ax3.set_xlim(tmin,tmax);ax3.set_ylim(-3.0,3.0);ax3.set_zlim(-3.0,3.0)
    ax3.set_box_aspect((2.5,1,1),zoom=1.22)
    ax3.view_init(elev=19,azim=-124)
    ax3.set_xlabel('Time (s)',fontsize=6.3,labelpad=-1)
    ax3.set_ylabel('Body x (r)',fontsize=6.3,labelpad=-1)
    ax3.set_zlabel('Body y (r)',fontsize=6.3,labelpad=-1)
    ax3.set_xticks([368,370,372,374,376]);ax3.tick_params(labelsize=5.9,pad=0)
    for axis in (ax3.xaxis,ax3.yaxis,ax3.zaxis):
        axis.pane.set_alpha(0)
        axis._axinfo['grid']['color']=GRID
        axis._axinfo['grid']['linewidth']=.35
    for x,state,label in ((1.05,'Right Side Up','Right side up'),
                          (2.00,'Being flipped','Being flipped'),
                          (2.95,'Upside Down','Upside down')):
        fig.add_artist(Line2D([x/W_IN,(x+.11)/W_IN],[1-(b_top+2.05)/H_IN]*2,transform=fig.transFigure,
                              color=STATES[state],lw=2))
        text_in(x+.15,b_top+2.05,label,6.0,INK)

    # Four compact, unfiltered measurements at the recorded half-second cadence.
    timeline=record['timeline'];times=np.array([r['time_s'] for r in timeline])
    configs=[
        ('Support area','10³ px',INK,
         np.array([r['support_area_model_px']/1000 if r['has_star'] else np.nan for r in timeline]),None),
        ('Recovered arms','count',INK,
         np.array([r['n_arms'] if r['has_star'] else np.nan for r in timeline]),None),
        ('Arm length','body radii',MUTED,None,'arm_lengths_body_radii'),
        ('Arm curvature','rad / body radius',MUTED,None,
         'arm_curvatures_rad_per_body_radius'),
    ]
    ts_x,ts_w,ts_h,ts_step=4.30,3.05,.32,.47
    for k,(title,unit,color,values,arm_key) in enumerate(configs):
        ax=ax_in(ts_x,b_top+.28+k*ts_step,ts_w,ts_h)
        ax.axvspan(370,375,color=AMBER,alpha=.10,lw=0)
        if arm_key is None:
            ax.plot(times,values,color=color,lw=1.05,marker='o',markersize=2.0)
        else:
            px=[];py=[]
            for row in timeline:
                px.extend([row['time_s']]*len(row[arm_key]));py.extend(row[arm_key])
            ax.scatter(px,py,s=5,color=color,alpha=.45,linewidths=0)
            medians=[np.median(row[arm_key]) if row[arm_key] else np.nan for row in timeline]
            ax.plot(times,medians,color=INK,lw=.9,marker='.',markersize=2.5)
        ax.set_xlim(365,379);ax.set_xticks([366,372,378])
        simple_axes(ax)
        if title=='Recovered arms':
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3,integer=True))
        if k<3:ax.tick_params(labelbottom=False)
        else:ax.set_xlabel('Time (s)',fontsize=6.3,labelpad=1)
        ax.set_title(title,loc='left',fontsize=6.8,color=INK,fontweight='bold',pad=2)
        ax.text(1,1.13,unit,transform=ax.transAxes,ha='right',fontsize=5.9,color=MUTED)

    # C: human and model states on their common scored interval.
    c_top=4.20
    heading('C','Held-out classification',c_top,AMBER)
    example=readout['example']
    assert example['video_id']==record['example_clip']
    scored=example['rows']
    scored_start=float(scored[0]['timestamp'])
    scored_stop=float(scored[-1]['timestamp'])+1
    for i,(state,label) in enumerate([('Right Side Up','Right side up'),
                                      ('Being flipped','Being flipped'),
                                      ('Upside Down','Upside down')]):
        x=.75+i*1.2
        fig.add_artist(Rectangle((x/W_IN,1-(c_top+.27)/H_IN-.0045),.10/W_IN,.009,transform=fig.transFigure,
                                 fc=STATES[state],ec='none'))
        text_in(x+.14,c_top+.27,label,6.3,INK)
    ax=ax_in(.75,c_top+.38,3.95,.52)
    ax.set_xlim(scored_start,scored_stop);ax.set_ylim(0,2.3)
    for row in example['human_intervals']:
        start=max(float(row['start_s']),scored_start)
        stop=min(float(row['stop_s']),scored_stop)
        if stop>start:
            ax.add_patch(Rectangle((start,1.25),stop-start,.74,
                                   fc=STATES[row['state']],ec='none'))
    for start,stop,state in machine_intervals(scored):
        ax.add_patch(Rectangle((start,.17),stop-start,.74,
                               fc=STATES[state],ec='none'))
    ax.set_yticks([1.62,.54],['Human','Model'])
    ax.set_xticks(np.arange(60,661,120),[str(i) for i in range(1,12,2)])
    ax.set_xlabel('Time in trial (min)',fontsize=6.5,labelpad=1)
    ax.tick_params(axis='y',length=0,labelsize=6.5,pad=4)
    ax.tick_params(axis='x',labelsize=6.2,length=2,pad=2)
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color(GRID)

    scores=readout['scores']['session_oof'][primary]['by_seed']
    n_anchors,n_clips=readout['anchors']['n'],readout['anchors']['clips']
    assert len(scores)==3 and all(row['anchors']==n_anchors and row['videos']==n_clips for row in scores)
    score_ax=ax_in(5.55,c_top+.38,1.80,.56)
    score_ax.set_xlim(0,1.03);score_ax.set_ylim(-.5,1.5)
    for ypos,key,color in [(1,'event_f1',INK),(0,'macro_f1',INK)]:
        vals=np.array([row[key] for row in scores]);mean=float(vals.mean());sd=float(vals.std(ddof=1))
        score_ax.barh(ypos,mean,height=.43,color=GRID,edgecolor=MUTED,lw=.6)
        score_ax.errorbar(mean,ypos,xerr=sd,fmt='o',color=color,ms=3.5,capsize=2)
        score_ax.scatter(vals,[ypos]*3,s=12,facecolors='white',edgecolors=color,lw=.7,zorder=4)
        score_ax.text(.99,ypos+.27,f'{mean:.3f}',ha='right',va='bottom',fontsize=6.2,color=INK)
    score_ax.set_yticks([1,0],['Flip episodes','Posture states'])
    score_ax.set_xticks([0,.5,1.0]);score_ax.set_xlabel('F1',fontsize=6.5,labelpad=1)
    score_ax.tick_params(axis='y',length=0,labelsize=6.2,pad=2)
    score_ax.tick_params(axis='x',labelsize=6.2,length=2)
    score_ax.spines[['top','right','left']].set_visible(False)
    score_ax.spines['bottom'].set_color(GRID)
    score_ax.grid(axis='x',color=GRID,lw=.5)
    text_in(5.55,c_top+.27,f'{n_clips} clips · {len(scores)} seeds',6.3,MUTED)

    # D: self-righting latency, from first upside-down to first right-side-up anchor.
    d_top=5.58
    heading('D','Self-righting time across all bouts',d_top,PURPLE)
    durations=readout['durations']
    shown=durations['by_seed'][str(durations['display_seed'])]
    human=np.array([e['duration_s'] for e in durations['human']])
    machine=np.array([e['duration_s'] for e in shown['machine']])
    bx=ax_in(.62,d_top+.32,1.22,1.05)
    boxes=bx.boxplot([human,machine],positions=[1,2],widths=.47,patch_artist=True,
                     showfliers=True,flierprops={'marker':'.','markersize':2.1,'alpha':.42},
                     medianprops={'color':INK,'linewidth':1.2},
                     whiskerprops={'color':MUTED,'linewidth':.8},
                     capprops={'color':MUTED,'linewidth':.8})
    for patch in boxes['boxes']:
        patch.set(facecolor=GRID,edgecolor=MUTED,linewidth=.9)
    bx.set_xticks([1,2],[f'Human\nn={len(human)}',f'Model\nn={len(machine)}'])
    bx.set_ylabel('Righting time (s)',fontsize=6.5)
    bx.set_ylim(0,130);simple_axes(bx)

    analysis=shown['matched_analysis']
    pairs=analysis['pairs']
    fx=ax_in(2.30,d_top+.32,1.0,1.0)
    fx.plot([0,130],[0,130],color=MUTED,lw=.75,ls='--',zorder=1)
    fx.scatter([pair['human_duration_s'] for pair in pairs],
               [pair['machine_duration_s'] for pair in pairs],
               s=8,color=INK,alpha=.4,linewidths=0,rasterized=True,zorder=2)
    fx.set_xlim(0,130);fx.set_ylim(0,130)
    fx.set_xticks([0,40,80,120]);fx.set_yticks([0,40,80,120])
    fx.set_xlabel('Human (s)',fontsize=6.5,labelpad=1)
    fx.set_ylabel('Model (s)',fontsize=6.5)
    simple_axes(fx)
    text_in(2.30,d_top+.20,f"Matched bouts (n={analysis['n_matched']})",6.2,INK,'bold')
    atlas=draw_atlas(fig,ax_in,text_in,3.80,d_top+.32,.55,1.05,.068)

    e_top=7.55
    kinematics=draw_kinematics(fig,ax_in,text_in,e_top)
    heading('E',f"Righting kinematics across {kinematics['n_events']} aligned flips",e_top,TEAL)

    base=HERE/'figure_04_flipping_analysis'
    summary={'figure':4,'version':15,'status':'published models','size_inches':[W_IN,H_IN],
             'generated_at_utc':datetime.now(timezone.utc).isoformat(),
             'models':models_record(),
             'hydra_pass_sha256':digest(DATA/'hydra_pass.json'),
             'prepared_sha256':digest(DATA/'prepared.json'),
             'readout_sha256':digest(READOUT),'readout_primary_variant':primary,
             'readout_prepare_script_sha256':digest(HERE/'prepare_figure_04_readout.py'),
             'aligned_bouts_sha256':digest(BOUTS),
             'bouts_prepare_script_sha256':digest(HERE/'prepare_figure_04_bouts.py'),
             'build_script_sha256':digest(Path(__file__)),
             'kinematics_prepare_script_sha256':digest(HERE/'prepare_figure_04_kinematics.py'),
             'landmarks_prepare_script_sha256':digest(HERE/'prepare_figure_04_landmarks.py'),
             'tree_features_prepare_script_sha256':digest(HERE/'prepare_figure_04_tree_features.py'),
             'counts':{'skeleton_frames':len({p['frame'] for p in paths}),
                       'human_complete_events':len(human),
                       'machine_complete_events':len(machine),
                       'matched_events':analysis['n_matched'],
                       'duration_display_seed':durations['display_seed']},
             'scores':{key:{'mean':float(np.mean([row[key] for row in scores])),
                            'sd':float(np.std([row[key] for row in scores],ddof=1))}
                       for key in ('event_f1','macro_f1')},
             'kinematics':kinematics,
             'atlas':atlas}
    if preview:
        full=HERE/'figure_04_flipping_analysis_readout_preview.png'
        fig.savefig(full,dpi=300)
        plt.close(fig)
        from PIL import Image
        with Image.open(full) as image:
            top=int(round((c_top-.17)/H_IN*image.height))
            image.crop((0,top,image.width,image.height)).save(HERE/'figure_04_panels_cde_readout_preview.png')
        summary['outputs']={name:{'path':str(HERE/name),'sha256':digest(HERE/name)} for name in
                            ('figure_04_flipping_analysis_readout_preview.png','figure_04_panels_cde_readout_preview.png')}
        summary['status']='preview'
        (HERE/'figure_04_readout_preview.provenance.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('Rendered Figure 4 v15 preview PNGs (full page and C-E).')
        return
    fig.savefig(base.with_suffix('.pdf'),dpi=300,
                metadata={'Title':'Flipping analysis from DeepStarNoodles outputs',
                          'Creator':'build_figure_04.py'})
    fig.savefig(base.with_suffix('.svg'),dpi=300)
    fig.savefig(base.with_suffix('.png'),dpi=300)
    plt.close(fig)
    summary['outputs']={ext:{'path':str(base.with_suffix('.'+ext)),
                             'sha256':digest(base.with_suffix('.'+ext))}
                        for ext in ('pdf','svg','png')}
    base.with_suffix('.provenance.json').write_text(json.dumps(summary,indent=2)+'\n')
    numbers=numbers_record(record,readout,primary,scores,durations,analysis,human,machine,atlas,kinematics)
    (HERE/'figure_04_numbers.json').write_text(json.dumps(numbers,indent=2,ensure_ascii=False)+'\n')
    print('Built Figure 4 v15 PDF, SVG, PNG, provenance and numbers.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--preview',action='store_true',help='write PNG previews only; leave the figure files untouched')
    build(parser.parse_args().preview)
