"""Portable GORC cache refit, calibration-only selection, and fixed-policy replay."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
CODE=Path(__file__).resolve().parent
ROOT=CODE.parent
sys.path.insert(0,str(CODE))

def install_read_guard(paths):
    denied=[os.path.normcase(os.path.abspath(p)).rstrip(os.sep) for p in paths]
    def guard(event,args):
        if event!="open" or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
        if any(path==d or path.startswith(d+os.sep) for d in denied):raise PermissionError("Clean reproduction blocked a private source-root read")
    sys.addaudithook(guard)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["replay","refit","smoke","summarize","configure","verify-online"])
    parser.add_argument("--dataset",default="coco_yolo")
    parser.add_argument("--out",type=Path,default=ROOT/"run_outputs/current")
    parser.add_argument("--images",type=Path);parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--deny-path",action="append",default=[])
    args=parser.parse_args();install_read_guard(args.deny_path)
    if args.deny_path:
        try:
            with open(Path(args.deny_path[0])/"__guard_probe_must_not_be_read__","rb"):pass
        except PermissionError:pass
        else:raise AssertionError("The source-root audit guard did not block the explicit probe")
    import numpy as np
    import pandas as pd
    import joblib
    import revision_core as core
    import enhanced_rankers as enhanced
    from portable_data import load_dataset
    from compiled_scorer import compile_model,predict_compiled,raw_numeric_features
    from threadpoolctl import threadpool_limits
    configs=json.loads((ROOT/"config/datasets.json").read_text())
    cfg=configs[args.dataset];out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.command=="configure":
        for path in [args.images,args.checkpoint]:
            if path is not None and not path.exists():raise FileNotFoundError(path)
        core.write_json(out/"optional_detector_paths.json",dict(images=str(args.images) if args.images else None,checkpoint=str(args.checkpoint) if args.checkpoint else None,
            scope="Optional detector regeneration configuration; cached-output reproduction does not read images or checkpoints"))
        return
    ds=load_dataset(args.dataset)
    if args.command=="verify-online":
        import online_runtime as online
        modelbase=ROOT/cfg["artifact_folder"];model=joblib.load(modelbase/"models/SCG.joblib")
        compiled=online.compile_class_interactions(model);inp=online.raw_input_arrays(ds,np.arange(len(ds.candidates)))
        policies=pd.read_csv(modelbase/"selected_policies.csv").set_index("method");prior=json.loads((modelbase/"class_prior.json").read_text())
        rows=[]
        with np.load(modelbase/"frozen_selected_scores.npz") as frozen:
            for implementation in ["dense_reference","compiled_class_coefficients"]:
                for method in ["SCG","RF","AP-C"]:
                    policy=policies.loc[method].to_dict();boxes,labels,score=online.apply_online(inp,model,compiled,policy,prior,implementation)
                    mask=frozen[method]>=policy["threshold"]
                    assert np.array_equal(boxes,inp["boxes"][mask]) and np.array_equal(labels,inp["labels"][mask])
                    error=float(np.abs(score-frozen[method][mask]).max());assert error<=1e-10
                    rows.append(dict(method=method,implementation=implementation,accepted_count=len(score),max_score_error=error,mask_identical=True))
        core.write_json(out/"verification.json",dict(complete=True,command="verify-online",dataset=args.dataset,private_source_read_guard_active=bool(args.deny_path),
            original_timing_helpers_match_frozen_acceptance=True,rows=rows))
        print("Original timing helpers verified from raw outputs against all frozen selected scores and masks.");return
    if args.command=="summarize":
        core.write_json(out/"source_summary.json",dict(dataset=ds.audit,known_classes=ds.known_classes,n_known_GT=int(ds.gt.gt_is_known.sum()),n_unknown_GT=int((~ds.gt.gt_is_known).sum()),
            calibration_images=len(ds.cal_ids),test_images=len(ds.test_ids),method_lock=json.loads((ROOT/"config/final_method_lock.json").read_text())))
        print(json.dumps(ds.audit));return
    smoke=args.command=="smoke"
    if smoke:
        # Retain explicit empty images when available; split populations never come from candidate.unique().
        cal=ds.cal_ids[:16];test=ds.test_ids[:16]
        empty=set(ds.split.image_id)-set(ds.candidates.image_id)
        for ids,kind in [(cal,"calibration"),(test,"test")]:
            options=ds.split.loc[ds.split.split.eq(kind)&ds.split.image_id.isin(empty),"image_id"].tolist()
            if options and options[0] not in ids:ids.append(options[0])
        kept=set(cal+test)
        ds=core.Dataset(ds.name,ds.candidates.loc[ds.candidates.image_id.isin(kept)].reset_index(drop=True),ds.gt.loc[ds.gt.image_id.isin(kept)].reset_index(drop=True),
            ds.split.loc[ds.split.image_id.isin(kept)].reset_index(drop=True),ds.known_classes,ds.audit)
    models={};scores={};modelbase=ROOT/cfg["artifact_folder"]
    with threadpool_limits(limits=1):
        for variant in cfg["variants"]:
            if args.command=="replay":model=joblib.load(modelbase/f"models/{variant}.joblib")
            elif variant=="SG":model=core.fit_ranker(ds.candidates,ds.cal_ids,variant,C=.1,gt=ds.gt)
            else:model=enhanced.fit_enhanced(ds.candidates,ds.cal_ids,ds.gt,family="class_interactions",variant=variant,C=.1)
            models[variant]=model;scores[variant]=model.predict_proba(ds.candidates)[:,1]
            if args.command!="replay":
                if variant=="SG":core.save_ranker(model,out/f"models/{variant}.joblib")
                else:enhanced.save_enhanced(model,out/f"models/{variant}.joblib")
        prior=json.loads((modelbase/"class_prior.json").read_text()) if args.command=="replay" else core.class_prior(ds.candidates,ds.cal_ids,gt=ds.gt)
        streams,specs=core.make_streams(ds.candidates,scores["SCG"],prior)
        evs={s:core.ExactEvaluator(ds.candidates,ds.gt,ids,ds.known_classes) for s,ids in [("calibration",ds.cal_ids),("test",ds.test_ids)]}
        if args.command=="replay":policies=pd.read_csv(modelbase/"selected_policies.csv").set_index("method").to_dict("index")
        else:
            grid=core.build_policy_grid(evs["calibration"],streams,specs);grid.to_csv(out/"policy_grid_cal.csv",index=False)
            policies={"Raw":core.select_policy(grid.loc[grid.score_col.eq("raw_score")]),"RF":core.select_policy(grid),
                "AP-C":core.select_policy(grid,"AP-C",cfg["AP_tolerance"],cfg["AP_tolerance"])}
            for variant in scores:
                grid=core.build_policy_grid(evs["calibration"],{variant:scores[variant]},[dict(score_col=variant,mode_family="direct")])
                policies[variant]=core.select_policy(grid)
        rows=[];allvalues={}
        for method,policy in policies.items():
            col=policy["score_col"];value=scores[col] if col in scores else streams[col];allvalues[method]=value
            for split,ev in evs.items():rows.append(dict(method=method,split=split,score_col=col,threshold=policy["threshold"],**ev.evaluate(value,policy["threshold"]),**ev.ap(value)))
        metrics=pd.DataFrame(rows);metrics.to_csv(out/"metrics.csv",index=False)
        pd.DataFrame([dict(method=m,**p) for m,p in policies.items()]).to_csv(out/"selected_policies.csv",index=False)
        compiled=predict_compiled(ds.candidates,compile_model(models["SCG"]))
        fold_error=float(np.abs(compiled-scores["SCG"]).max());assert fold_error<=1e-10
        if args.command=="replay":
            with np.load(modelbase/"frozen_selected_scores.npz") as z:
                score_errors={m:float(np.abs(v-z[m]).max()) for m,v in allvalues.items()}
            assert max(score_errors.values())<=1e-10,score_errors
        else:score_errors={}
    expected=pd.read_csv(modelbase/"main_metrics.csv",float_precision="round_trip")
    expected=expected.loc[expected.method.isin(policies)]
    comparisons=[]
    if not smoke:
        assert len(expected)==len(metrics)
        numeric=[c for c in metrics if c not in ["method","split","score_col","threshold"]]
        for _,row in metrics.iterrows():
            reference=expected.loc[expected.method.eq(row.method)&expected.split.eq(row.split)].iloc[0]
            delta=max(abs(float(row[c])-float(reference[c])) for c in numeric)
            same=delta<=1e-10
            comparisons.append(dict(method=row.method,split=row.split,max_metric_absolute_error=delta,agrees_with_archived_result=same))
        pd.DataFrame(comparisons).to_csv(out/"reconciliation.csv",index=False)
        assert all(r["agrees_with_archived_result"] for r in comparisons),comparisons
    sizes=ds.split.set_index("image_id")
    width=ds.candidates.image_id.map(sizes.width).to_numpy(float);height=ds.candidates.image_id.map(sizes.height).to_numpy(float)
    features=raw_numeric_features(ds.candidates[["x1","y1","x2","y2"]].to_numpy(float),ds.candidates.raw_score.to_numpy(float),width,height)
    feature_error=float(np.abs(features-ds.candidates[core.SCORES+core.GEOMETRY].to_numpy(float)).max());assert feature_error<=1e-10
    report=dict(complete=True,command=args.command,dataset=args.dataset,metrics_rows=len(metrics),calibration_images=len(ds.cal_ids),test_images=len(ds.test_ids),
        test_candidate_count=int(ds.candidates.image_id.isin(ds.test_ids).sum()),empty_images=len(set(ds.split.image_id)-set(ds.candidates.image_id)),
        full_result_reconciled=not smoke,max_compiled_score_absolute_error=fold_error,max_reconstructed_feature_absolute_error=feature_error,
        fixed_score_errors=score_errors,private_source_read_guard_active=bool(args.deny_path),
        protocol="explicit complete split population; revised predicted-class TP/all-overlap UFA/GT-independent BG; custom full-candidate101-point AP before acceptance threshold")
    core.write_json(out/"verification.json",report)
    print(json.dumps(report))

if __name__=="__main__":main()
