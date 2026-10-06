"""Locked, self-contained revision experiment core (2026-09-30).

This implements corrected manuscript accepted-set definitions and the project's
101-point AP; it is not pycocotools COCOeval. Selection uses explicit IDs only.
The legacy matcher is retained for exact step8j reconciliation. Revised matching
uses predicted-class known GT and separates unknown objects from BG detections.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT = Path(__file__).resolve().parent.parent
REVISION = PROJECT / "run_outputs"
if (REVISION / "runtime_deps").exists():
    sys.path.insert(0,str(REVISION / "runtime_deps"))
try:
    from numba import njit
except ImportError:
    def njit(*args,**kwargs):
        return lambda f:f
GEOMETRY = ["box_area_norm", "box_aspect_log", "box_width_norm", "box_height_norm",
            "center_x_norm", "center_y_norm", "edge_min_dist_norm", "edge_contact_count"]
SCORES = ["raw_score", "score_logit"]
ANCHORS = [.05, .10, .15, .20, .25, .30, .35, .40, .50, .60, .70, .80, .90]
QUANTILES = [.25, .50, .70, .80, .90, .95]
ALPHAS = [.05, .10, .20, .40, .60, .80, 1.0]
GAMMAS = [.25, .50, .75, 1.0]
BETAS = [.05, .10, .20]
LOCKED_RECIPE = dict(C=.5, class_weight="balanced", solver="liblinear", max_iter=1000,
                     random_state=12, unknown_weight=2., background_weight=1.25,
                     feature_order=SCORES + GEOMETRY + ["pred_label"])
DATA_PATHS = {name: {} for name in ['coco_yolo','coco_gdino','lvis_yolo']}

def normalize_image_id(value):
    s = Path(str(value).replace("\\", "/")).stem
    m = re.search(r"(\d+)$", s)
    return f"lvis_{int(m.group(1)):012d}" if m else s

def safe_class_name(value):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]+", "_", re.sub(r"\s+", "_", str(value).strip().lower()))).strip("_")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def ids_hash(ids):
    return hashlib.sha256("\n".join(sorted(map(normalize_image_id, ids))).encode()).hexdigest()

def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")

def geometry_features(candidates, split):
    """Same formulas as step12c; recompute only absent fields (LVIS cache)."""
    out = candidates.copy()
    sizes = split.set_index("image_id")[["width", "height"]]
    w = out.image_id.map(sizes.width).to_numpy(float)
    h = out.image_id.map(sizes.height).to_numpy(float)
    x1, y1, x2, y2 = (out[c].to_numpy(float) for c in ["x1", "y1", "x2", "y2"])
    bw, bh = np.maximum(x2-x1, 0), np.maximum(y2-y1, 0)
    gaps = np.column_stack([np.maximum(x1, 0), np.maximum(y1, 0), np.maximum(w-x2, 0), np.maximum(h-y2, 0)])
    values = dict(box_area_norm=bw*bh/(w*h), box_aspect_log=np.log(np.maximum(bw, 1e-6)/np.maximum(bh, 1e-6)),
                  box_width_norm=bw/w, box_height_norm=bh/h, center_x_norm=(x1+x2)*.5/w,
                  center_y_norm=(y1+y2)*.5/h, edge_min_dist_norm=gaps.min(axis=1)/np.maximum(np.minimum(w,h),1),
                  edge_contact_count=(gaps <= (.015*np.maximum(np.maximum(w,h),1))[:, None]).sum(axis=1))
    for c, val in values.items():
        if c not in out:
            out[c] = val
    return out

@dataclass
class Dataset:
    name: str
    candidates: pd.DataFrame
    gt: pd.DataFrame
    split: pd.DataFrame
    known_classes: list[str]
    audit: dict
    @property
    def cal_ids(self): return self.split.loc[self.split.split.eq("calibration"), "image_id"].tolist()
    @property
    def test_ids(self): return self.split.loc[self.split.split.eq("test"), "image_id"].tolist()

def load_dataset(name="coco_yolo", unknown_access=True, paths=None):
    if paths is not None: raise ValueError("Use package data/config paths")
    from portable_data import load_dataset as portable_load
    return portable_load(name, unknown_access=unknown_access)

def _iou_matrix(a, b):
    if not len(b): return np.zeros((len(a),0))
    wh = np.maximum(np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]),0)
    inter = wh[:,:,0]*wh[:,:,1]
    aa = np.maximum(a[:,2]-a[:,0],0)*np.maximum(a[:,3]-a[:,1],0)
    bb = np.maximum(b[:,2]-b[:,0],0)*np.maximum(b[:,3]-b[:,1],0)
    den = aa[:,None]+bb[None,:]-inter
    return np.divide(inter,den,out=np.zeros_like(inter),where=(inter>0)&(den>0))

def known_tp_labels(candidates, known_gt, iou=.5):
    """Historical training labels: best known box first, then unique raw TP key."""
    neighbors = [None]*len(candidates)
    gtgroups = {img:g for img,g in known_gt.groupby("image_id",sort=False)}
    labels = np.zeros(len(candidates),dtype=np.int8)
    for img, indices in candidates.groupby("image_id",sort=False).indices.items():
        g = gtgroups.get(img)
        if g is None: continue
        mat = _iou_matrix(candidates.iloc[indices][["x1","y1","x2","y2"]].to_numpy(float),g[["x1","y1","x2","y2"]].to_numpy(float))
        if not mat.shape[1]: continue
        best = mat.argmax(axis=1)
        for j, cidx in enumerate(indices):
            if mat[j,best[j]]>=iou and candidates.iloc[cidx].pred_label == g.iloc[best[j]].gt_label:
                neighbors[cidx] = (img,str(g.iloc[best[j]].gt_id))
    seen = set()
    for idx in pd.Series(candidates.raw_score.to_numpy()).sort_values(ascending=False).index.to_numpy():
        key = neighbors[idx]
        if key is not None and key not in seen:
            labels[idx]=1; seen.add(key)
    return labels

def label_candidates(candidates,gt,fit_ids,known_classes=None):
    """Revised training labels and taxonomy using only the explicit fit images."""
    ids=set(map(normalize_image_id,fit_ids))
    train=candidates.loc[candidates.image_id.isin(ids)].copy()
    subset_gt=gt.loc[gt.image_id.isin(ids)].copy()
    classes=known_classes or sorted(subset_gt.loc[subset_gt.gt_is_known,"gt_label"].unique())
    ev=ExactEvaluator(train,subset_gt,sorted(ids),classes,taxonomy="revised")
    result=ev.evaluate(train.raw_score.to_numpy(),0,return_types=True)
    train["risk_error_type"]=result["error_types"]
    train["risk_label_tp"]=(train.risk_error_type.eq("known_tp")).astype(int)
    return train

def fit_ranker(candidates, fit_ids, variant="SCG", supervision="full", C=.5, random_state=12,gt=None):
    if gt is None: raise ValueError("Explicit fit GT is required: pass gt=ds.gt; cached legacy labels are not used.")
    fit_ids = set(map(normalize_image_id,fit_ids))
    if supervision=="known_only" and bool((~gt.loc[gt.image_id.isin(fit_ids),"gt_is_known"]).any()):
        raise ValueError("Known-only fit must receive known-only GT from load_dataset(unknown_access=False).")
    train = label_candidates(candidates,gt,fit_ids)
    numeric = SCORES + (GEOMETRY if variant in ["SCG","SG"] else [])
    transformers = [("num",StandardScaler(),numeric)]
    if variant in ["SCG","SC"]: transformers.append(("cat",OneHotEncoder(handle_unknown="ignore"),["pred_label"]))
    model = Pipeline([("pre",ColumnTransformer(transformers,remainder="drop")),
                      ("clf",LogisticRegression(C=C,class_weight="balanced",solver="liblinear",max_iter=1000,random_state=random_state))])
    y = train.risk_label_tp.astype(int).to_numpy()
    sw = np.ones(len(train))
    if supervision=="full":
        sw[train.risk_error_type.eq("unknown_false_accept").to_numpy()] = 2.
        sw[train.risk_error_type.eq("background_false_accept").to_numpy()] = 1.25
    elif supervision!="known_only": raise ValueError(supervision)
    model.fit(train,y,clf__sample_weight=sw)
    model.revision_config_ = dict(LOCKED_RECIPE, C=C, random_state=random_state, variant=variant, supervision=supervision,
                                  numeric=numeric, with_class=variant in ["SCG","SC"], fit_ids_sha256=ids_hash(fit_ids),
                                  taxonomy="revised predicted-class TP, all-overlap UFA, GT-independent BG", n_fit_images=len(fit_ids), n_fit_candidates=len(train), positive=int(y.sum()),
                                  negative=int((1-y).sum()), risk_weight_sum=float(sw.sum()),
                                  feature_names=model.named_steps["pre"].get_feature_names_out().tolist())
    model.revision_training_labels_ = train[["image_id","det_id","risk_label_tp","risk_error_type"]].copy()
    return model

def save_ranker(model,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,path)
    model.revision_training_labels_.to_csv(path.with_suffix(".training_labels.csv"),index=False)
    pre=model.named_steps["pre"]; clf=model.named_steps["clf"]
    write_json(path.with_suffix(".json"),dict(model.revision_config_,coef=clf.coef_.tolist(),intercept=clf.intercept_.tolist(),
               scaler_mean=pre.named_transformers_["num"].mean_.tolist(),scaler_scale=pre.named_transformers_["num"].scale_.tolist(),
               class_encoder_categories=[x.tolist() for x in pre.named_transformers_["cat"].categories_] if "cat" in pre.named_transformers_ else []))

def class_prior(candidates,fit_ids,mode="full",gt=None):
    if gt is None: raise ValueError("Explicit fit GT is required: pass gt=ds.gt.")
    if mode=="known_only" and bool((~gt.loc[gt.image_id.isin(set(map(normalize_image_id,fit_ids))),"gt_is_known"]).any()):
        raise ValueError("Known-only prior requires known-only GT.")
    train=label_candidates(candidates,gt,fit_ids)
    weights={}
    for label,sub in train.groupby("pred_label",sort=True):
        n=len(sub); tp=float(sub.risk_label_tp.sum()); p=(tp+1)/(n+2)
        if mode=="full":
            bg=int(sub.risk_error_type.eq("background_false_accept").sum())
            unk=int(sub.risk_error_type.eq("unknown_false_accept").sum())
            weights[label]=(p*((n-bg+1)/(n+2))*((n-unk+1)/(n+2)))**(1/3)
        elif mode=="known_only": weights[label]=p
        else: raise ValueError(mode)
    vmax=max(weights.values(),default=1.)
    return {c:float(np.clip(v/max(vmax,1e-9),.2,1.)) for c,v in weights.items()}

def make_streams(candidates,ranker_scores,prior):
    raw=np.clip(candidates.raw_score.to_numpy(float),0,1); rank=np.clip(np.asarray(ranker_scores,float),0,1)
    streams={"raw_score":raw,"geometry_risk_score":rank}
    specs=[dict(score_col="raw_score",mode_family="raw",ap_rank_preserving_by_design=True),dict(score_col="geometry_risk_score",mode_family="geometry",ap_rank_preserving_by_design=False)]
    for alpha in ALPHAS:
        col=f"geometry_alpha_{alpha:.3f}".replace(".","p")
        streams[col]=alpha*rank+(1-alpha)*raw
        specs.append(dict(score_col=col,mode_family="alpha_blend",alpha=alpha,ap_rank_preserving_by_design=alpha==0))
    base=candidates.pred_label.map(prior).fillna(1).to_numpy(float)
    for gamma in GAMMAS:
        col=f"class_reliability_gamma_{gamma:.3f}".replace(".","p")
        streams[col]=raw*np.power(base,gamma)
        specs.append(dict(score_col=col,mode_family="class_scaled_raw",gamma=gamma,ap_rank_preserving_by_design=True))
    for beta in BETAS:
        col=f"raw_light_geom_gate_{beta:.3f}".replace(".","p")
        streams[col]=raw*np.clip((1-beta)+beta*rank,0,1)
        specs.append(dict(score_col=col,mode_family="light_geometry_gate",beta=beta,ap_rank_preserving_by_design=False))
    return streams,specs

def make_thresholds(scores):
    scores=np.asarray(scores,float); scores=scores[np.isfinite(scores)]
    values=set(ANCHORS)
    if len(scores): values.update(np.quantile(scores,QUANTILES).tolist()); values.add(float(scores.max()))
    return sorted(x for x in values if np.isfinite(x) and x>=0)

def metrics_from_counts(counts):
    nk,nu,nd,tp,ufa,bg=map(float,counts)
    fp=nd-tp; p=tp/nd if nd else 0.; r=tp/nk if nk else 0.; urr=1-ufa/nu if nu else 1.
    B=3/(1/p+1/r+1/urr) if min(p,r,urr)>0 else 0.
    h=2*r*urr/(r+urr) if r+urr else 0.
    return dict(num_known_gt=int(nk),num_unknown_gt=int(nu),num_accepted_detections=int(nd),tp_known=int(tp),fp_known=int(fp),fn_known=int(nk-tp),
                known_precision=p,known_recall=r,unknown_false_accept_objects=int(ufa),unknown_reject_rate_object_level=urr,
                open_set_hscore_object_level=h,precision_recall_unknown_balanced_score=B,background_false_accept_count=int(bg))

@njit(cache=True)
def _match_kernel(order,pred_labels,gt_labels,det_images,base,kptr,kids,kov,aptr,aids,aov,uptr,uids,uov,n_gt,iou,revised):
    matched=np.zeros(n_gt,dtype=np.bool_); unknown=np.zeros(n_gt,dtype=np.bool_)
    counts=base.copy(); types=np.zeros(len(pred_labels),dtype=np.int8)
    for d in order:
        image=det_images[d]; counts[image,2]+=1
        best=-1
        if revised:
            for j in range(aptr[d],aptr[d+1]):
                if aov[j]<iou: break
                g=aids[j]
                if not matched[g]: best=g; break
        else:
            for j in range(kptr[d],kptr[d+1]):
                if kov[j]<iou: break
                g=kids[j]
                if not matched[g]: best=g; break
        if best>=0 and pred_labels[d]==gt_labels[best]:
            matched[best]=True; counts[image,3]+=1; types[d]=1
            continue
        known_overlap=False
        if revised:
            known_overlap=(kptr[d]<kptr[d+1] and kov[kptr[d]]>=iou)
        else: known_overlap=best>=0
        has_unknown=False
        for j in range(uptr[d],uptr[d+1]):
            if uov[j]<iou: break
            g=uids[j]; has_unknown=True
            if not unknown[g]: unknown[g]=True; counts[image,4]+=1
            if not revised: break
        if has_unknown: types[d]=2
        elif known_overlap: types[d]=3
        else: types[d]=4; counts[image,5]+=1
    return counts,types

@njit(cache=True)
def _ap_match_kernel(order,aptr,aids,aov,n_gt,iou):
    matched=np.zeros(n_gt,dtype=np.bool_); tp=np.zeros(len(order))
    for j in range(len(order)):
        d=order[j]
        for k in range(aptr[d],aptr[d+1]):
            if aov[k]<iou: break
            g=aids[k]
            if not matched[g]: matched[g]=True; tp[j]=1.; break
    return tp

def _as_csr(neighbors):
    ptr=np.zeros(len(neighbors)+1,dtype=np.int64)
    for i,(idx,ov) in enumerate(neighbors): ptr[i+1]=ptr[i]+len(idx)
    ids=np.concatenate([x[0] for x in neighbors]).astype(np.int64) if neighbors else np.empty(0,dtype=np.int64)
    ovs=np.concatenate([x[1] for x in neighbors]).astype(float) if neighbors else np.empty(0)
    return ptr,ids,ovs

class ExactEvaluator:
    """Precomputed IoU; revised default and exactly reproduced legacy taxonomy."""
    def __init__(self,candidates,gt,image_ids,known_classes,taxonomy="revised"):
        if taxonomy not in ["revised","legacy"]: raise ValueError(taxonomy)
        self.taxonomy=taxonomy
        self.image_ids=list(map(normalize_image_id,image_ids)); self.image_map={v:i for i,v in enumerate(self.image_ids)}
        mask=candidates.image_id.isin(self.image_map)
        self.indices=np.flatnonzero(mask.to_numpy()); self.candidates=candidates.loc[mask].reset_index(drop=True)
        self.gt=gt.loc[gt.image_id.isin(self.image_map)].reset_index(drop=True)
        self.known_classes=list(known_classes)
        all_labels=sorted(set(self.gt.gt_label)|set(self.candidates.pred_label)); lm={v:i for i,v in enumerate(all_labels)}
        self.pred_labels=self.candidates.pred_label.map(lm).to_numpy(int)
        self.gt_labels=self.gt.gt_label.map(lm).to_numpy(int)
        self.gt_known=self.gt.gt_is_known.to_numpy(bool)
        self.det_images=self.candidates.image_id.map(self.image_map).to_numpy(int)
        self.gt_images=self.gt.image_id.map(self.image_map).to_numpy(int)
        self.counts_base=np.zeros((len(self.image_ids),6),dtype=np.int64)
        np.add.at(self.counts_base[:,0],self.gt_images[self.gt_known],1)
        np.add.at(self.counts_base[:,1],self.gt_images[~self.gt_known],1)
        self.known_neighbors=[None]*len(self.candidates); self.unknown_best=np.full(len(self.candidates),-1,dtype=int); self.unknown_iou=np.zeros(len(self.candidates))
        self.ap_neighbors=[None]*len(self.candidates)
        self.unknown_neighbors=[None]*len(self.candidates)
        gg=self.gt.groupby("image_id",sort=False).indices
        boxes=self.candidates[["x1","y1","x2","y2"]].to_numpy(float); gboxes=self.gt[["x1","y1","x2","y2"]].to_numpy(float)
        for img, di in self.candidates.groupby("image_id",sort=False).indices.items():
            gi=np.asarray(gg.get(img,[]),dtype=int); ki=gi[self.gt_known[gi]]; ui=gi[~self.gt_known[gi]]
            km=_iou_matrix(boxes[di],gboxes[ki]); um=_iou_matrix(boxes[di],gboxes[ui])
            for j,d in enumerate(di):
                if len(ki):
                    order=np.argsort(-km[j],kind="stable"); keep=order[km[j,order]>=.30]
                    self.known_neighbors[d]=(ki[keep],km[j,keep])
                    same=keep[self.gt_labels[ki[keep]]==self.pred_labels[d]]
                    self.ap_neighbors[d]=(ki[same],km[j,same])
                else:
                    self.known_neighbors[d]=(np.empty(0,dtype=int),np.empty(0)); self.ap_neighbors[d]=(np.empty(0,dtype=int),np.empty(0))
                if len(ui):
                    u=int(np.argmax(um[j])); self.unknown_best[d]=ui[u]; self.unknown_iou[d]=um[j,u]
                    uorder=np.argsort(-um[j],kind="stable"); keepu=uorder[um[j,uorder]>=.30]
                    self.unknown_neighbors[d]=(ui[keepu],um[j,keepu])
                else: self.unknown_neighbors[d]=(np.empty(0,dtype=int),np.empty(0))
        self.kptr,self.kids,self.kov=_as_csr(self.known_neighbors)
        self.aptr,self.aids,self.aov=_as_csr(self.ap_neighbors)
        self.uptr,self.uids,self.uov=_as_csr(self.unknown_neighbors)
        self._ap_cache={}

    def local_scores(self,scores):
        scores=np.asarray(scores,float)
        if len(scores)==len(self.candidates): return scores
        return scores[self.indices]

    def evaluate(self,scores,threshold=.0,iou=.5,per_image=False,return_types=False,taxonomy=None):
        scores=self.local_scores(scores)
        accepted=np.flatnonzero(scores>=float(threshold))
        # Pandas quicksort on precisely the threshold-filtered original row order.
        order=accepted[pd.Series(scores[accepted]).sort_values(ascending=False).index.to_numpy()]
        counts,codes=_match_kernel(order.astype(np.int64),self.pred_labels,self.gt_labels,self.det_images,self.counts_base,
                    self.kptr,self.kids,self.kov,self.aptr,self.aids,self.aov,self.uptr,self.uids,self.uov,len(self.gt),float(iou),(taxonomy or self.taxonomy)=="revised")
        result=metrics_from_counts(counts.sum(axis=0))
        if per_image: result["per_image_counts"]=counts
        if return_types: result["error_types"]=np.array(["rejected","known_tp","unknown_false_accept","wrong_known_class_or_duplicate","background_false_accept"],dtype=object)[codes]
        return result

    def ap(self,scores,return_per_class=False):
        scores=self.local_scores(scores); key=hashlib.sha256(scores.tobytes()).hexdigest()
        if key in self._ap_cache:
            summary,table=self._ap_cache[key]; return (summary.copy(),table.copy()) if return_per_class else summary.copy()
        rows=[]; ious=[round(x,2) for x in np.arange(.5,.96,.05)]
        for label in self.known_classes:
            di=np.flatnonzero(self.candidates.pred_label.eq(label).to_numpy())
            npos=int((self.gt.gt_is_known & self.gt.gt_label.eq(label)).sum())
            order=di[pd.Series(scores[di]).sort_values(ascending=False).index.to_numpy()]
            for iou in ious:
                if not npos: ap=float("nan")
                elif not len(di): ap=0.
                else:
                    tp=_ap_match_kernel(order.astype(np.int64),self.aptr,self.aids,self.aov,len(self.gt),float(iou))
                    tc=np.cumsum(tp); rec=tc/npos; prec=tc/np.arange(1,len(order)+1)
                    envelope=np.maximum.accumulate(prec[::-1])[::-1]
                    ix=np.searchsorted(rec,np.linspace(0,1,101),side="left")
                    ap=float(envelope[ix[ix<len(order)]].sum()/101.)
                rows.append(dict(class_name=label,iou_threshold=iou,AP=ap,num_gt=npos,num_det=len(di)))
        table=pd.DataFrame(rows)
        summary={f"AP{round(iou*100)}":float(table.loc[table.iou_threshold.eq(iou),"AP"].mean()) for iou in ious}
        summary["AP"]=float(np.mean(list(summary.values())))
        self._ap_cache[key]=(summary,table)
        return (summary.copy(),table.copy()) if return_per_class else summary.copy()

def build_policy_grid(evaluator,streams,specs,with_ap=True):
    rows=[]; masks_seen={}; streams_seen={}; duplicate_streams={}
    for spec in specs:
        col=spec["score_col"]; local=evaluator.local_scores(streams[col]); h=hashlib.sha256(local.tobytes()).hexdigest()
        duplicate_streams[col]=streams_seen.get(h,col); streams_seen[h]=duplicate_streams[col]
        ap=evaluator.ap(streams[col]) if with_ap else {}
        for threshold in make_thresholds(local):
            # A policy is score ordering plus acceptance mask. Different order can
            # change TP even for the same mask, so retain stream hash in identity.
            maskhash=hashlib.sha256(np.packbits(local>=threshold).tobytes()).hexdigest()
            identity=(h,maskhash)
            if identity not in masks_seen: masks_seen[identity]=evaluator.evaluate(streams[col],threshold)
            m=masks_seen[identity]
            rows.append(dict(spec,threshold=threshold,cal_balanced=m["precision_recall_unknown_balanced_score"],cal_precision=m["known_precision"],
                cal_recall=m["known_recall"],cal_unknown_reject=m["unknown_reject_rate_object_level"],cal_num_accepted=m["num_accepted_detections"],
                cal_unknown_false_accept=m["unknown_false_accept_objects"],cal_background_false_accepts=m["background_false_accept_count"],
                cal_tp_known=m["tp_known"],cal_AP50=ap.get("AP50",np.nan),cal_AP75=ap.get("AP75",np.nan),cal_AP=ap.get("AP",np.nan),
                stream_hash=h,acceptance_mask_hash=maskhash,canonical_stream=duplicate_streams[col]))
    grid=pd.DataFrame(rows)
    grid.attrs["capacity"]=dict(named_streams=len(specs),unique_score_streams=len(streams_seen),named_policies=len(rows),unique_ordered_acceptance_policies=len(masks_seen),
         rules=dict(anchors=ANCHORS,cal_quantiles=QUANTILES,include_cal_max=True),duplicate_streams=duplicate_streams)
    return grid

def select_policy(grid,mode="B",ap50_tol=0.,ap_tol=0.):
    work=grid.copy()
    if mode=="AP-C":
        raw=work.loc[work.score_col.eq("raw_score")].iloc[0]
        work=work.loc[(work.cal_AP50>=raw.cal_AP50-ap50_tol-1e-12)&(work.cal_AP>=raw.cal_AP-ap_tol-1e-12)]
    elif mode not in ["B","F1"]: raise ValueError(mode)
    if work.empty: raise ValueError("No eligible policies")
    if mode=="F1":
        work["selection_objective"]=2*work.cal_precision*work.cal_recall/(work.cal_precision+work.cal_recall).clip(lower=1e-12)
        keys=["selection_objective","cal_precision","cal_recall","cal_num_accepted"]
        ascending=[False,False,False,True]
    else: work["selection_objective"]=work.cal_balanced
    if mode!="F1":
        keys=["selection_objective","cal_unknown_reject","cal_precision","cal_recall","cal_background_false_accepts","cal_num_accepted"]
        ascending=[False,False,False,False,True,True]
    return work.sort_values(keys,ascending=ascending,kind="stable").iloc[0].to_dict()

def paired_image_bootstrap(evaluator,policies,n_boot=2000,seed=20260930,iou=.5):
    """Fixed-policy paired image bootstrap, includes zero-candidate images.
    policies maps method -> (scores, fixed threshold). Returns replicate metrics.
    Matching is independent within images, so replicate totals from exact image
    counts equal evaluation on cloned images. It does not refit/reselect models.
    """
    counts={k:evaluator.evaluate(s,t,iou,per_image=True)["per_image_counts"] for k,(s,t) in policies.items()}
    rng=np.random.default_rng(seed); n=len(evaluator.image_ids); rows=[]
    for b in range(n_boot):
        multiplicity=np.bincount(rng.integers(0,n,n),minlength=n)
        for method,arr in counts.items(): rows.append(dict(replicate=b,method=method,**metrics_from_counts(multiplicity @ arr)))
    return pd.DataFrame(rows)

def run_e0(dataset_name="coco_yolo",outdir=None):
    outdir=Path(outdir or REVISION / "results/E0" / dataset_name); outdir.mkdir(parents=True,exist_ok=True)
    started=time.time(); ds=load_dataset(dataset_name)
    ap_tolerance=.005 if dataset_name=="lvis_yolo" else 0.
    write_json(outdir/"locked_config.json",dict(recipe=LOCKED_RECIPE,dataset=ds.audit,cal_ids_sha256=ids_hash(ds.cal_ids),test_ids_sha256=ids_hash(ds.test_ids),selection="formal project B on calibration only",taxonomy="revised predicted-class TP, all-overlap UFA, GT-independent BG",AP="project custom 101-point, IoU .50:.95, not official COCOeval",AP50_tolerance=ap_tolerance,AP_tolerance=ap_tolerance,tie_break=["higher URR","higher precision","higher recall","lower BG","fewer accepted"]))
    evaluators={"calibration":ExactEvaluator(ds.candidates,ds.gt,ds.cal_ids,ds.known_classes),"test":ExactEvaluator(ds.candidates,ds.gt,ds.test_ids,ds.known_classes)}
    scores={}; models={}
    for variant in ["SC","SG","SCG"]:
        model=fit_ranker(ds.candidates,ds.cal_ids,variant,gt=ds.gt); models[variant]=model
        scores[variant]=model.predict_proba(ds.candidates)[:,1]
        save_ranker(model,outdir/f"models/{variant}.joblib")
        print(f"{dataset_name}: fitted {variant}, dim={len(model.revision_config_['feature_names'])}",flush=True)
    prior=class_prior(ds.candidates,ds.cal_ids,gt=ds.gt); write_json(outdir/"class_prior.json",prior)
    streams,specs=make_streams(ds.candidates,scores["SCG"],prior)
    grid=build_policy_grid(evaluators["calibration"],streams,specs)
    grid.to_csv(outdir/"policy_grid_cal.csv",index=False)
    write_json(outdir/"policy_capacity.json",grid.attrs["capacity"])
    selected={"Raw":select_policy(grid.loc[grid.score_col.eq("raw_score")]),"RF":select_policy(grid),"AP-C":select_policy(grid,"AP-C",ap50_tol=ap_tolerance,ap_tol=ap_tolerance)}
    eligibility=[]
    for tol50 in [0.,.005,.01,.02]:
        for tol in [0.,.005,.01,.02]:
            p=select_policy(grid,"AP-C",tol50,tol)
            raw=grid.loc[grid.score_col.eq("raw_score")].iloc[0]
            eligible=grid.loc[(grid.cal_AP50>=raw.cal_AP50-tol50-1e-12)&(grid.cal_AP>=raw.cal_AP-tol-1e-12)]
            eligibility.append(dict(AP50_tolerance=tol50,AP_tolerance=tol,eligible_named_streams=eligible.score_col.nunique(),eligible_named_policies=len(eligible),selected_score_col=p["score_col"],selected_threshold=p["threshold"],selected_cal_B=p["cal_balanced"],main_apc_rule=(tol50==ap_tolerance and tol==ap_tolerance)))
    pd.DataFrame(eligibility).to_csv(outdir/"ap_eligibility_sensitivity_cal.csv",index=False)
    for variant in ["SC","SG","SCG"]:
        g=build_policy_grid(evaluators["calibration"],{variant:scores[variant]},[dict(score_col=variant,mode_family="direct")])
        g.to_csv(outdir/f"direct_{variant}_policy_grid_cal.csv",index=False)
        selected[variant]=select_policy(g)
    rows=[]
    for method,policy in selected.items():
        col=policy["score_col"]; values=scores[col] if col in scores else streams[col]
        for split,ev in evaluators.items():
            m=ev.evaluate(values,policy["threshold"]); ap=ev.ap(values)
            rows.append(dict(method=method,split=split,score_col=col,threshold=policy["threshold"],**m,**ap))
            print(f"{method}/{split}: B={m['precision_recall_unknown_balanced_score']:.9f}, UFA={m['unknown_false_accept_objects']}, BG={m['background_false_accept_count']}",flush=True)
    pd.DataFrame(rows).to_csv(outdir/"main_metrics.csv",index=False)
    pd.DataFrame([dict(method=k,**v) for k,v in selected.items()]).to_csv(outdir/"selected_policies.csv",index=False)
    np.savez_compressed(outdir/"scores.npz",**scores,**streams)
    ds.candidates[["image_id","det_id"]].to_csv(outdir/"score_alignment.csv",index=False)
    write_json(outdir/"integrity.json",dict(dataset=ds.audit,recipe=LOCKED_RECIPE,policy_capacity=grid.attrs["capacity"],elapsed_seconds=time.time()-started,complete=True,
                 outputs={str(p.relative_to(outdir)):sha256_file(p) for p in outdir.rglob("*") if p.is_file() and p.name!="integrity.json"}))
    return ds,evaluators,models,streams,scores,selected,pd.DataFrame(rows)

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset",default="coco_yolo",choices=list(DATA_PATHS)); parser.add_argument("--outdir",type=Path)
    args=parser.parse_args(); run_e0(args.dataset,args.outdir)
