"""Reproduce the baseline bridge, paired baseline, LVIS support and nested AP analyses."""
from pathlib import Path
import sys, json, argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import revision_core as core
from portable_data import load_dataset
from portable_cli import install_read_guard
REV=ROOT/'source_data'
METRICS=['precision_recall_unknown_balanced_score','known_precision','known_recall','unknown_false_accept_objects','background_false_accept_count']

def ci_table(replicates,comparisons,metrics):
    rows=[]
    for a,b in comparisons:
        for metric in metrics:
            pivot=replicates.pivot(index="replicate",columns="method",values=metric)
            values=(pivot[a]-pivot[b]).to_numpy()
            rows.append(dict(comparison=f"{a} minus {b}",metric=metric,n_boot=len(values),mean_difference=float(values.mean()),
                median_difference=float(np.median(values)),CI_low=float(np.quantile(values,.025)),CI_high=float(np.quantile(values,.975)),
                positive_fraction=float((values>0).mean()),negative_fraction=float((values<0).mean()),zero_fraction=float((values==0).mean())))
    return pd.DataFrame(rows)

def selection_key(metrics):
    return (metrics["precision_recall_unknown_balanced_score"],metrics["unknown_reject_rate_object_level"],
            metrics["known_precision"],metrics["known_recall"],-metrics["background_false_accept_count"],
            -metrics["num_accepted_detections"])

def output_integrity(out, extra):
    core.write_json(out / "integrity.json", dict(extra, complete=True,
        script_sha256=core.sha256_file(__file__), outputs={str(p.relative_to(out)):core.sha256_file(p)
        for p in out.rglob("*") if p.is_file() and p.name != "integrity.json"}))

def run_reassessment(ds, out):
    """Additional reviewer-requested analyses using existing fixed policies."""
    import re
    from sklearn.compose import ColumnTransformer
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    final=REV/'final_results/E0/coco_yolo'
    alignment=pd.read_csv(final/'score_alignment.csv')
    assert alignment.det_id.astype(str).tolist()==ds.candidates.det_id.astype(str).tolist()
    pp=pd.read_csv(final/'selected_policies.csv')
    with np.load(ROOT/'artifacts/coco_yolo/frozen_selected_scores.npz')as a:fs={row.score_col:a[row.method].copy()for row in pp.itertuples()}
    policies=pd.read_csv(final/'selected_policies.csv').set_index('method')
    rfrow=policies.loc['RF'];rf=fs[rfrow.score_col];rfthr=float(rfrow.threshold)
    bp=REV/'external_baselines'
    th=json.loads((bp/'models/raw_per_class_thresholds.json').read_text())
    raw=ds.candidates.raw_score.to_numpy(float);limits=np.array([th['thresholds'].get(x,th['fallback_unseen_class_threshold'])for x in ds.candidates.pred_label])
    pl=json.loads((bp/'baseline_definitions.json').read_text())['parameters']['Ordinary Platt/logistic']
    plscores=expit((ds.candidates.score_logit.to_numpy(float)-pl['scaler_mean'][0])/pl['scaler_scale'][0]*pl['coefficient_standardized'][0][0]+pl['intercept_standardized'][0])
    bs={'raw_per_class_accepted_stream':np.where(raw>=limits,raw,-np.inf),'Ordinary Platt/logistic':plscores}
    classscore=bs['raw_per_class_accepted_stream']
    ev=core.ExactEvaluator(ds.candidates,ds.gt,ds.test_ids,ds.known_classes)
    cal=core.ExactEvaluator(ds.candidates,ds.gt,ds.cal_ids,ds.known_classes)
    legacy=core.ExactEvaluator(ds.candidates,ds.gt,ds.test_ids,ds.known_classes,taxonomy='legacy')
    paired={'RF':(rf,rfthr),'Raw per-class threshold':(classscore,0)}
    reps=core.paired_image_bootstrap(ev,paired,n_boot=2000,seed=20261001)
    reps.to_csv(out/'RF_vs_perclass_bootstrap_replicates.csv',index=False)
    ci=ci_table(reps,[('RF','Raw per-class threshold')],METRICS)
    point={m:ev.evaluate(s,t)for m,(s,t)in paired.items()}
    ci['point_difference']=[point['RF'][m]-point['Raw per-class threshold'][m]for m in ci.metric]
    ci.to_csv(out/'RF_vs_perclass_paired_CI.csv',index=False)
    pd.DataFrame([dict(method=m,**v)for m,v in point.items()]).to_csv(out/'RF_vs_perclass_metrics.csv',index=False)
    apci=pd.read_csv(REV/'final_results/E1/exact_AP_bootstrap/paired_CI.csv')
    apci=apci[apci.comparison.eq('RF minus Raw')].copy();apci['comparison']='RF minus Raw per-class threshold'
    apci['justification']='Per-class threshold full-list AP uses the exact same raw ranking as Raw; existing paired raw-ranking draws apply unchanged.'
    apci.to_csv(out/'RF_vs_perclass_full_list_AP_CI.csv',index=False)
    print('Reassessment paired RF/per-class comparison complete',flush=True)

    # Restore the precise submitted score-only recipe rather than relabel a new fit.
    cache=pd.read_csv(ds.audit['cache'],usecols=['image_id','det_id','split','score_logit','risk_label_tp'])
    cache.image_id=cache.image_id.map(core.normalize_image_id)
    assert cache.det_id.astype(str).tolist()==ds.candidates.det_id.astype(str).tolist()
    fitmask=cache.image_id.isin(ds.cal_ids);oldy=cache.loc[fitmask,'risk_label_tp'].to_numpy(int)
    newlabels=core.label_candidates(ds.candidates,ds.gt,ds.cal_ids,ds.known_classes)
    assert newlabels.det_id.astype(str).tolist()==cache.loc[fitmask,'det_id'].astype(str).tolist()
    def old_recipe(y):
        model=Pipeline([('pre',ColumnTransformer([('num',StandardScaler(),['score_logit'])],remainder='drop')),
                        ('clf',LogisticRegression(C=.5,class_weight='balanced',max_iter=1000,solver='liblinear',random_state=2026))])
        model.fit(cache.loc[fitmask],y)
        return model,model.predict_proba(cache)[:,1]
    old_model,oldscores=old_recipe(oldy);new_model,newscores=old_recipe(newlabels.risk_label_tp.to_numpy(int))
    bridge=[]
    def emit(name,values,threshold,evaluator,note):
        bridge.append(dict(stage=name,threshold=threshold,**evaluator.evaluate(values,threshold),**evaluator.ap(values),note=note))
    emit('Submitted score-only fit and threshold',oldscores,.85,legacy,'Single score logit; C=0.5; class-balanced; no risk-type sample weights; original cached labels and legacy evaluator.')
    original=pd.read_csv(ROOT/'data/coco_yolo/submitted_Platt_reference.csv')
    oldrow=original[original.method.eq('Score-only Platt/logistic')].iloc[0]
    for key,col in [('precision_recall_unknown_balanced_score','B'),('unknown_false_accept_objects','UFA'),('background_false_accept_count','BG_FP'),('num_accepted_detections','accepted_detection_count')]:
        assert abs(bridge[-1][key]-float(oldrow[col]))<1e-10,(key,bridge[-1][key],oldrow[col])
    emit('Same fit and threshold; corrected evaluator',oldscores,.85,ev,'Only the evaluator changes.')
    oldgrid=core.build_policy_grid(cal,{'score':oldscores},[dict(score_col='score',mode_family='direct')]);oldpol=core.select_policy(oldgrid)
    emit('Same fit; recalibrated threshold',oldscores,float(oldpol['threshold']),ev,'Corrected evaluator and calibration-only threshold reselection; same historical training labels.')
    emit('Same recipe; corrected labels; original threshold',newscores,.85,ev,'Corrected known-TP labels; otherwise original score-only class-balanced C=0.5 recipe.')
    newgrid=core.build_policy_grid(cal,{'score':newscores},[dict(score_col='score',mode_family='direct')]);newpol=core.select_policy(newgrid)
    emit('Original score-only recipe recalibrated',newscores,float(newpol['threshold']),ev,'Corrected labels and evaluator; calibration-only threshold reselection; no risk-type weights.')
    # Preserve the submitted threshold search as well as the fitting recipe.
    # Original source: gorc_round2_strong_baselines.BASE_THRESHOLDS/make_thresholds.
    submitted_anchors=[.001,.002,.003,.005,.008,.01,.015,.02,.03,.05,.08,.1,.15,.2,.25,.3,.4,.5,.65,.75,.85,.95]
    grid_checks=[]
    for label,values in [('submitted_fit',oldscores),('corrected_label_fit',newscores)]:
        cal_values=pd.Series(values[fitmask.to_numpy()])
        candidates=sorted(set(submitted_anchors+[float(cal_values.quantile(q))for q in np.linspace(.02,.98,25)]+[float(cal_values.min()),float(cal_values.max())]))
        grid=[]
        for threshold in candidates:grid.append(dict(threshold=threshold,**cal.evaluate(values,threshold)))
        best=max(grid,key=selection_key)
        pd.DataFrame(grid).to_csv(out/f'{label}_submitted_threshold_grid_cal.csv',index=False)
        stage='Same submitted fit and original grid; corrected evaluator' if label=='submitted_fit' else 'Original recipe with submitted threshold grid'
        emit(stage,values,float(best['threshold']),ev,'Submitted grid: 22 fixed anchors, 25 quantiles from 0.02 to 0.98, and fitting-score extrema. Corrected calibration evaluation chooses the threshold.')
        grid_checks.append(dict(fit=label,threshold=best['threshold'],calibration_B=best['precision_recall_unknown_balanced_score'],candidate_count=len(candidates)))
    cal_legacy=core.ExactEvaluator(ds.candidates,ds.gt,ds.cal_ids,ds.known_classes,taxonomy='legacy')
    old_cal_values=pd.Series(oldscores[fitmask.to_numpy()])
    legacy_thresholds=sorted(set(submitted_anchors+[float(old_cal_values.quantile(q))for q in np.linspace(.02,.98,25)]+[float(old_cal_values.min()),float(old_cal_values.max())]))
    legacy_grid=[dict(threshold=t,**cal_legacy.evaluate(oldscores,t))for t in legacy_thresholds]
    legacy_selected=max(legacy_grid,key=selection_key)
    assert legacy_selected['threshold']==.85,legacy_selected
    pd.DataFrame(legacy_grid).to_csv(out/'submitted_fit_original_evaluator_grid_cal.csv',index=False)
    core.write_json(out/'submitted_threshold_grid_verification.json',dict(original_legacy_selection_reproduced=True,original_selected_threshold=.85,anchors=submitted_anchors,quantiles=np.linspace(.02,.98,25).tolist(),corrected_evaluation_selections=grid_checks))
    bpol=pd.read_csv(REV/'external_baselines/selected_policies.csv').set_index('method')
    emit('Ordinary unweighted Platt',bs['Ordinary Platt/logistic'],float(bpol.loc['Ordinary Platt/logistic','threshold']),ev,'One score logit; C=0.5; no class balancing or risk-type weighting; separate probability-calibration baseline.')
    risk=pd.read_csv(REV/'final_results/baselines_matched_score_C01/metrics.csv');risk=risk[risk.split.eq('test')].iloc[0].to_dict()
    bridge.append(dict(stage='Risk-weighted score-only control',**risk,note='Raw score and logit; C=0.1; class balancing and risk-type weights. This is a different control, not the submitted Platt row.'))
    pd.DataFrame(bridge).to_csv(out/'submitted_Platt_bridge.csv',index=False)
    oldgrid.to_csv(out/'submitted_fit_reselection_grid_cal.csv',index=False);newgrid.to_csv(out/'original_recipe_recalibration_grid_cal.csv',index=False)
    joblib.dump(new_model,out/'original_recipe_recalibrated.joblib')
    newlabels.to_csv(out/'original_recipe_training_labels.csv',index=False)
    core.write_json(out/'original_recipe_selected_policy.json',newpol)
    print('Reassessment submitted Platt row exactly reproduced; bridge complete',flush=True)

    # Cross-annotation support, not semantic reannotation or replacement UFA.
    lvispath=ROOT/'data/lvis_coco_val_bbox_annotations.json'
    lv=json.loads(lvispath.read_text(encoding='utf-8'))
    image_map={};image_records={}
    for im in lv['images']:
        url=im.get('coco_url',im.get('file_name',''))
        match=re.search(r'(\d{12})\.(?:jpg|png)',url)
        if match:
            name=core.normalize_image_id(match.group(1));image_map[im['id']]=name;image_records[name]=im
    anns={};cats={a['id']:a['name']for a in lv['categories']};all_protocol_ids=set(ds.split.image_id)
    for ann in lv['annotations']:
        name=image_map.get(ann['image_id'])
        if name in all_protocol_ids:anns.setdefault(name,[]).append(ann)
    candidate=ds.candidates.set_index('det_id',drop=False);gtgroups={k:g for k,g in ds.gt.groupby('image_id')}
    covered=set(ds.test_ids)&set(image_records)
    splitdims=ds.split.set_index('image_id')
    for image_id in covered:
        for dim in ['width','height']:assert int(image_records[image_id][dim])==int(splitdims.loc[image_id,dim])
    support=[];crosssummary=[]
    allpol={'Raw':(fs[policies.loc['Raw','score_col']],float(policies.loc['Raw','threshold'])),**paired}
    for method,(scores,threshold)in allpol.items():
        evaluated=ev.evaluate(scores,threshold,return_types=True)
        bg=ev.candidates.loc[np.asarray(evaluated['error_types'])=='background_false_accept'].copy()
        assert len(bg)==evaluated['background_false_accept_count']
        for image_id,dets in bg.groupby('image_id',sort=False):
            aa=anns.get(image_id,[]);boxes=np.array([[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]]for a in aa],float).reshape(-1,4)
            db=dets[['x1','y1','x2','y2']].to_numpy(float)
            iou=core._iou_matrix(db,boxes) if len(boxes) else np.zeros((len(db),0))
            cg=gtgroups.get(image_id);gb=cg[['x1','y1','x2','y2']].to_numpy(float)if cg is not None else np.empty((0,4))
            counterpart=core._iou_matrix(boxes,gb).max(axis=1)if len(boxes)and len(gb)else np.zeros(len(boxes))
            for j,row in enumerate(dets.itertuples()):
                k=int(iou[j].argmax())if len(boxes)else None;maximum=float(iou[j,k])if k is not None else 0.
                hit=aa[k]if k is not None else {}
                support.append(dict(method=method,image_id=image_id,det_id=row.det_id,LVIS_image_covered=image_id in covered,max_LVIS_IoU=maximum,
                    matched_LVIS_annotation_id=hit.get('id'),matched_LVIS_category=cats.get(hit.get('category_id'),''),
                    LVIS_match_at_50=maximum>=.5,matched_LVIS_box_max_COCO_IoU=float(counterpart[k])if k is not None else np.nan,
                    support_without_COCO_counterpart=bool(maximum>=.5 and counterpart[k]<.5)))
        local=pd.DataFrame([x for x in support if x['method']==method]);eligible=local[local.LVIS_image_covered]
        image_counts=pd.DataFrame(index=ds.test_ids,columns=['covered_BG','matched_BG','matched_no_COCO_counterpart'],data=0)
        for image_id,g in eligible.groupby('image_id'):
            image_counts.loc[image_id]=[len(g),int(g.LVIS_match_at_50.sum()),int(g.support_without_COCO_counterpart.sum())]
        rng=np.random.default_rng(20261002);ratios=[];arr=image_counts.to_numpy(float)
        for b in range(2000):
            draw=rng.integers(0,len(arr),len(arr));sums=arr[draw].sum(axis=0);ratios.append(sums[1]/sums[0]if sums[0]else np.nan)
        crosssummary.append(dict(method=method,total_BG=len(local),covered_BG=len(eligible),uncovered_BG=len(local)-len(eligible),LVIS_matched_BG=int(eligible.LVIS_match_at_50.sum()),
            supported_fraction_covered=float(eligible.LVIS_match_at_50.mean()),supported_fraction_all=float(local.LVIS_match_at_50.mean()),
            support_fraction_CI_low=float(np.nanquantile(ratios,.025)),support_fraction_CI_high=float(np.nanquantile(ratios,.975)),
            support_without_COCO_counterpart=int(eligible.support_without_COCO_counterpart.sum())))
    pd.DataFrame(support).to_csv(out/'LVIS_BG_cross_annotation_detections.csv',index=False)
    pd.DataFrame(crosssummary).to_csv(out/'LVIS_BG_cross_annotation_summary.csv',index=False)
    core.write_json(out/'LVIS_cross_annotation_protocol.json',dict(lvis_path=str(lvispath),lvis_sha256=core.sha256_file(lvispath),
        coco_5000_images_covered=len(set(ds.split.image_id)&set(image_records)),coco_5000_LVIS_annotations=sum(len(x)for x in anns.values()),
        test_images=4000,test_images_LVIS_covered=len(covered),category_count=len(cats),IoU=.5,bootstrap_unit='image',bootstrap_repeats=2000,seed=20261002,
        limitations='Additional annotation support is measurable. It does not prove that a detection corresponds to a COCO-missing distinct object: annotation vocabulary and box extents can differ. Unmatched boxes remain unresolved because LVIS federated labels are incomplete. UFA/BG metric definitions and original policies remain unchanged.',
        primary_source='https://openaccess.thecvf.com/content_CVPR_2019/papers/Gupta_LVIS_A_Dataset_for_Large_Vocabulary_Instance_Segmentation_CVPR_2019_paper.pdf'))
    nested=pd.read_csv(REV/'final_results/E5/nested_fit_selection_test.csv')
    nrows=[]
    for (seed,setting),g in nested[nested.split.eq('test')].groupby(['inner_seed','selection_setting']):
        g=g.set_index('method');raw=g.loc['Raw']
        for method in ['RF','AP-C']:
            x=g.loc[method];nrows.append(dict(seed=int(seed),selection_setting=setting,method=method,ranker_family=x.ranker_family,C=float(x.C),
                B=float(x.precision_recall_unknown_balanced_score),delta_B=float(x.precision_recall_unknown_balanced_score-raw.precision_recall_unknown_balanced_score),AP=float(x.AP),delta_AP=float(x.AP-raw.AP)))
    pd.DataFrame(nrows).to_csv(out/'nested_AP_gate_comparison.csv',index=False)
    output_integrity(out,dict(no_new_detector_inference=True,no_new_main_policy_selection=True,original_Platt_row_exactly_reproduced=True,
        paired_bootstrap_repeats=2000,paired_seed=20261001,all_test_images_including_empty=True,LVIS_annotation_support_not_semantic_reannotation=True))
    print('Reassessment LVIS crosscheck and nested AP extraction complete',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'run_outputs/reassessment');parser.add_argument('--deny-path',action='append',default=[])
    args=parser.parse_args();install_read_guard(args.deny_path)
    ds=load_dataset('coco_yolo');ds.audit['cache']=str(ROOT/'data/coco_yolo/submitted_label_cache.csv')
    run_reassessment(ds,args.out)
