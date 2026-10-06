"""Pure helper functions copied from the measured final output-side implementation."""
import time
import numpy as np
import pandas as pd
from scipy.special import expit
EPS=1e-6

def raw_input_arrays(ds, indices):
    """Prepare detector-output interface inputs, outside timed online operations."""
    c=ds.candidates.iloc[indices]
    sizes=ds.split.set_index("image_id")
    return dict(boxes=c[["x1","y1","x2","y2"]].to_numpy(float),labels=c.pred_label.to_numpy(str),
                raw=c.raw_score.to_numpy(float),width=c.image_id.map(sizes.width).to_numpy(float),height=c.image_id.map(sizes.height).to_numpy(float))


def build_online_features(inp):
    """Reconstruct ten numeric terms and class identity from raw outputs on each call."""
    x1,y1,x2,y2=inp["boxes"].T; w,h=inp["width"],inp["height"]
    bw,bh=np.maximum(x2-x1,0),np.maximum(y2-y1,0)
    gap=np.column_stack([np.maximum(x1,0),np.maximum(y1,0),np.maximum(w-x2,0),np.maximum(h-y2,0)])
    score=np.clip(inp["raw"],EPS,1-EPS)
    return pd.DataFrame(dict(raw_score=inp["raw"],score_logit=np.log(score/(1-score)),
        box_area_norm=bw*bh/(w*h),box_aspect_log=np.log(np.maximum(bw,EPS)/np.maximum(bh,EPS)),
        box_width_norm=bw/w,box_height_norm=bh/h,center_x_norm=(x1+x2)*.5/w,center_y_norm=(y1+y2)*.5/h,
        edge_min_dist_norm=gap.min(axis=1)/np.maximum(np.minimum(w,h),1),
        edge_contact_count=(gap <= (.015*np.maximum(np.maximum(w,h),1))[:,None]).sum(axis=1),pred_label=inp["labels"]))


def transform_selected(raw, rank, labels, policy, prior):
    family=policy["mode_family"]
    if family=="direct" or family=="geometry":return rank
    if family=="raw":return np.clip(raw,0,1)
    if family=="alpha_blend":return float(policy["alpha"])*rank+(1-float(policy["alpha"]))*np.clip(raw,0,1)
    if family=="class_scaled_raw":
        weights=np.array([prior.get(c,1.) for c in labels])
        return np.clip(raw,0,1)*weights**float(policy["gamma"])
    if family=="light_geometry_gate":return np.clip(raw,0,1)*np.clip((1-float(policy["beta"]))+float(policy["beta"])*rank,0,1)
    raise ValueError(family)


def timing_repeated(fn, warmup=10, repeats=100):
    for _ in range(warmup):fn()
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter_ns();out=fn();end=time.perf_counter_ns()
        samples.append((end-start)/1e6)
        if len(out)!=3:raise AssertionError("Unexpected online output")
    samples=np.asarray(samples)
    return dict(median_ms=float(np.median(samples)),p95_ms=float(np.quantile(samples,.95)),mean_ms=float(samples.mean()),min_ms=float(samples.min())),samples


def compile_class_interactions(model):
    features=model.named_steps["features"];clf=model.named_steps["clf"]
    assert features.family=="class_interactions" and features.variant=="SCG"
    assert features.use_interactions_ and not features.use_profile_
    d=len(features.numeric_);k=len(features.categories_);weight=clf.coef_[0].copy()
    assert len(weight)==d+k+k*d
    global_numeric=weight[:d];class_offset=weight[d:d+k];interactions=weight[d+k:].reshape(k,d)
    # The final extra row exactly implements handle_unknown='ignore'.
    coefficients=np.vstack([interactions+global_numeric,global_numeric])
    offsets=np.r_[class_offset+clf.intercept_[0],clf.intercept_[0]]
    return dict(numeric_features=features.numeric_,categories=features.categories_,
        mean=features.scaler_.mean_.copy(),scale=features.scaler_.scale_.copy(),coefficients=coefficients,offsets=offsets,
        dense_feature_dimension=len(weight),trained_logistic_parameters=len(weight)+1,C=float(clf.C),
        formula="z=(numeric-mean)/scale; logit=(global_numeric+class_interaction[c])路z + intercept+class_offset[c]",
        unknown_label_rule="global_numeric coefficients and intercept only; exactly corresponds to ignored unknown one-hot class")


def numeric_from_raw(inp):
    """Raw boxes/scores/dimensions to ten original numeric terms; no cached features."""
    x1,y1,x2,y2=inp["boxes"].T;w,h=inp["width"],inp["height"]
    bw,bh=np.maximum(x2-x1,0),np.maximum(y2-y1,0)
    gaps=np.column_stack([np.maximum(x1,0),np.maximum(y1,0),np.maximum(w-x2,0),np.maximum(h-y2,0)])
    score=np.clip(inp["raw"],1e-6,1-1e-6)
    return np.column_stack([inp["raw"],np.log(score/(1-score)),bw*bh/(w*h),
        np.log(np.maximum(bw,1e-6)/np.maximum(bh,1e-6)),bw/w,bh/h,(x1+x2)*.5/w,(y1+y2)*.5/h,
        gaps.min(axis=1)/np.maximum(np.minimum(w,h),1),
        (gaps<=(.015*np.maximum(np.maximum(w,h),1))[:,None]).sum(axis=1)])


def compiled_scores(inp,compiled):
    numeric=numeric_from_raw(inp)
    numeric-=compiled["mean"];numeric/=compiled["scale"]
    classes=pd.Categorical(inp["labels"],categories=compiled["categories"]).codes.astype(np.int64)
    classes[classes<0]=len(compiled["categories"])
    logit=np.einsum("ij,ij->i",numeric,compiled["coefficients"][classes])+compiled["offsets"][classes]
    return expit(logit)


def apply_online(inp,model,compiled,policy,class_prior,implementation):
    if implementation=="dense_reference":
        frame=build_online_features(inp)
        rank=model.predict_proba(frame)[:,1] if len(frame) else np.empty(0)
    elif implementation=="compiled_class_coefficients":rank=compiled_scores(inp,compiled)
    else:raise ValueError(implementation)
    score=transform_selected(inp["raw"],rank,inp["labels"],policy,class_prior)
    keep=score>=float(policy["threshold"])
    return inp["boxes"][keep],inp["labels"][keep],score[keep]


