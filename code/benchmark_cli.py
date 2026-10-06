"""Time the complete output-side paths copied from the measured final implementation."""
import argparse,json,sys
from pathlib import Path
CODE=Path(__file__).resolve().parent;ROOT=CODE.parent;sys.path.insert(0,str(CODE))
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from portable_data import load_dataset
import online_runtime as online

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--dataset",default="coco_yolo");parser.add_argument("--out",type=Path,default=ROOT/"run_outputs/benchmark")
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/"config/datasets.json").read_text())[args.dataset];base=ROOT/config["artifact_folder"]
    ds=load_dataset(args.dataset);model=joblib.load(base/"models/SCG.joblib");compiled=online.compile_class_interactions(model)
    policies=pd.read_csv(base/"selected_policies.csv").set_index("method");prior=json.loads((base/"class_prior.json").read_text())
    all_input=online.raw_input_arrays(ds,np.arange(len(ds.candidates)))
    rank=online.compiled_scores(all_input,compiled)
    with np.load(base/"frozen_selected_scores.npz") as frozen:
        for method in ["SCG","RF","AP-C"]:
            p=policies.loc[method].to_dict();values=online.transform_selected(all_input["raw"],rank,all_input["labels"],p,prior)
            assert np.abs(values-frozen[method]).max()<=1e-10
            assert np.array_equal(values>=p["threshold"],frozen[method]>=p["threshold"])
    testidx=np.flatnonzero(ds.candidates.image_id.isin(ds.test_ids).to_numpy());batch=online.raw_input_arrays(ds,testidx)
    counts=ds.candidates.iloc[testidx].groupby("image_id").size().reindex(ds.test_ids,fill_value=0);representatives=[]
    for lo,hi in [(0,0),(1,5),(6,10),(11,20),(21,50),(51,100),(101,int(max(101,counts.max())))]:
        pool=counts.loc[counts.between(lo,hi)]
        if not len(pool):continue
        median=float(pool.median());img=sorted(pool.index,key=lambda x:(abs(int(pool.loc[x])-median),x))[0]
        index=np.flatnonzero(ds.candidates.image_id.eq(img).to_numpy())
        representatives.append((img,int(pool.loc[img]),online.raw_input_arrays(ds,index)))
    rows=[];repeats=[]
    with threadpool_limits(limits=1):
        for implementation in ["dense_reference","compiled_class_coefficients"]:
            for method in ["SCG","RF","AP-C"]:
                policy=policies.loc[method].to_dict()
                summary,samples=online.timing_repeated(lambda:online.apply_online(batch,model,compiled,policy,prior,implementation))
                rows.append(dict(method=method,implementation=implementation,scope="batch",n_images=len(ds.test_ids),n_candidates=len(testidx),**summary,
                    batch_median_ms_per_image=summary["median_ms"]/len(ds.test_ids),batch_p95_ms_per_image=summary["p95_ms"]/len(ds.test_ids)))
                repeats.extend(dict(method=method,implementation=implementation,scope="batch",image_id="",repeat=i,elapsed_ms=v) for i,v in enumerate(samples))
                for img,n,inp in representatives:
                    summary,samples=online.timing_repeated(lambda:online.apply_online(inp,model,compiled,policy,prior,implementation))
                    rows.append(dict(method=method,implementation=implementation,scope="single-image representative",image_id=img,n_images=1,n_candidates=n,**summary))
                    repeats.extend(dict(method=method,implementation=implementation,scope="single-image",image_id=img,repeat=i,elapsed_ms=v) for i,v in enumerate(samples))
    pd.DataFrame(rows).to_csv(args.out/"online_latency_summary.csv",index=False);pd.DataFrame(repeats).to_csv(args.out/"100_repeats.csv",index=False)
    print("Complete raw-output timing finished: 10 warmups, 100 repetitions, one CPU thread; batch means and representative-image measurements are separate.")

if __name__=="__main__":main()
