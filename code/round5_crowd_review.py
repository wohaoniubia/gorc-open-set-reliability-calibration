"""Reproduce crowd-box overlap and author-name summaries without changing metrics."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def calculate(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    paths={
      'candidates':root/'data/coco_yolo/candidates.csv',
      'support':root/'source_data/final_results/E7/policy_candidate_annotation_support_test.csv',
      'coco':root/'data/instances_val2017.json',
      'ratings':root/'source_data/manual_review_20261006/ratings_with_sample_ids.csv',
      'gt':root/'data/coco_yolo/ground_truth.csv'}
    cand=pd.read_csv(paths['candidates'],float_precision='round_trip')
    support=pd.read_csv(paths['support'],float_precision='round_trip')
    coco=json.loads(paths['coco'].read_text(encoding='utf-8'))
    ratings=pd.read_csv(paths['ratings'],keep_default_na=False)
    gt=pd.read_csv(paths['gt']);crowd={};crowd_ids=set()
    for a in coco['annotations']:
        if a.get('iscrowd',0):
            x,y,w,h=a['bbox'];crowd.setdefault(int(a['image_id']),[]).append((a['id'],a['category_id'],x,y,x+w,y+h));crowd_ids.add(a['id'])
    def overlap(row):
        image=int(str(row.image_id).split('_')[-1]);area=max(row.x2-row.x1,0)*max(row.y2-row.y1,0)
        values=[]
        for aid,cat,x1,y1,x2,y2 in crowd.get(image,[]):
            inter=max(0,min(row.x2,x2)-max(row.x1,x1))*max(0,min(row.y2,y2)-max(row.y1,y1))
            values.append((inter/area if area else 0,aid,cat))
        return max(values,default=(0,None,None))
    cols=['det_id','x1','y1','x2','y2','pred_label']
    all_rows=[];summary=[]
    for m in ['Raw','SC','SCG','RF']:
        s=support[support.method.eq(m)&support.type_at_50.eq('background_false_accept')].merge(cand[cols],on='det_id',validate='one_to_one')
        result=[overlap(row)for row in s.itertuples()]
        s['max_crowd_box_coverage']=[x[0]for x in result];s['best_crowd_annotation_id']=[x[1]for x in result];s['best_crowd_category_id']=[x[2]for x in result]
        s['crowd_box_coverage_ge_50']=s.max_crowd_box_coverage.ge(.5)
        n=int(s.crowd_box_coverage_ge_50.sum())
        summary.append(dict(method=m,BG_FP=len(s),inside_crowd_box=n,fraction=n/len(s),outside_crowd_box=len(s)-n))
        all_rows.append(s)
    details=pd.concat(all_rows,ignore_index=True);summary=pd.DataFrame(summary)
    assert summary.BG_FP.tolist()==[2875,2833,2848,2828]
    rf=details[details.method.eq('RF')]
    sample=ratings.merge(rf[['det_id','max_crowd_box_coverage','crowd_box_coverage_ge_50']],on='det_id',validate='one_to_one')
    assert len(sample)==150 and sample.image_id.nunique()==125
    known=set(gt.loc[gt.gt_is_known.astype(str).str.lower().eq('true'),'gt_label'].str.replace('_',' ').str.lower())
    assert len(known)==20
    sample['recorded_name_normalized']=sample.object_name.str.strip().str.lower().str.replace('_',' ')
    sample['recorded_name_matches_known']=sample.recorded_name_normalized.isin(known)
    recognized=sample[sample.category.eq('可辨认物体')]
    name_summary=dict(recognizable=len(recognized),known_name_matches=int(recognized.recorded_name_matches_known.sum()),person_name=int(recognized.recorded_name_normalized.eq('person').sum()),well_fitted_known_name=int((recognized.recorded_name_matches_known&recognized.box_fit.eq('较完整贴合物体')).sum()),outside_known_names=int((~recognized.recorded_name_matches_known).sum()),outside_name_counts=recognized.loc[~recognized.recorded_name_matches_known,'recorded_name_normalized'].value_counts().to_dict(),sample_inside_crowd_box=int(sample.crowd_box_coverage_ge_50.sum()),sample_images=sample.image_id.nunique(),vocabulary_relation_used=False)
    report=dict(definition='max over iscrowd=1 boxes of area(detection intersect crowd box)/area(detection) >= 0.5; any category; bounding boxes, not segmentation masks',diagnostic_only=True,original_metrics_changed=False,source_hashes={k:digest(p)for k,p in paths.items()},crowd_annotations_in_coco=len(crowd_ids),crowd_annotations_in_project_gt=len(crowd_ids & set(gt.gt_id.astype(int))),manual_name_summary=name_summary,scope_note='Name matching is literal after case/space normalization, not semantic reannotation. Nonmatching names include generic signs and body parts; these are not nine verified distinct unknown objects.')
    details.to_csv(out/'crowd_BG_detections.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(out/'crowd_BG_summary.csv',index=False,encoding='utf-8-sig')
    sample.to_csv(out/'manual_name_and_crowd_details.csv',index=False,encoding='utf-8-sig')
    (out/'crowd_and_name_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=lambda x:int(x)if isinstance(x,np.integer)else float(x)),encoding='utf-8')
    print(summary.to_string(index=False));print(json.dumps(name_summary,ensure_ascii=True,indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--out',type=Path)
    args=parser.parse_args();calculate(args.root,args.out or args.root/'run_outputs/crowd_review')
