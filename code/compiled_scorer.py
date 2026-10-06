"""Exact algebraic class-interaction fold; no fit or policy selection here."""
import numpy as np
import pandas as pd
from scipy.special import expit

def compile_model(model):
    f=model.named_steps["features"];clf=model.named_steps["clf"]
    if f.family!="class_interactions" or f.use_profile_:raise ValueError("Class interactions without profile terms required")
    d=len(f.numeric_);k=len(f.categories_);w=clf.coef_[0]
    return dict(numeric=f.numeric_,categories=f.categories_,mean=f.scaler_.mean_,scale=f.scaler_.scale_,
        coefficients=np.vstack([w[d+k:].reshape(k,d)+w[:d],w[:d]]),offsets=np.r_[w[d:d+k]+clf.intercept_[0],clf.intercept_[0]])

def predict_compiled(frame,compiled):
    x=frame[compiled["numeric"]].to_numpy(float).copy()
    x-=compiled["mean"];x/=compiled["scale"]
    codes=pd.Index(compiled["categories"]).get_indexer(frame.pred_label).astype(np.int64)
    codes[codes<0]=len(compiled["categories"])
    return expit(np.einsum("ij,ij->i",x,compiled["coefficients"][codes])+compiled["offsets"][codes])

def raw_numeric_features(boxes,raw,width,height):
    x1,y1,x2,y2=np.asarray(boxes,float).T;w=np.asarray(width,float);h=np.asarray(height,float)
    bw,bh=np.maximum(x2-x1,0),np.maximum(y2-y1,0)
    gaps=np.column_stack([np.maximum(x1,0),np.maximum(y1,0),np.maximum(w-x2,0),np.maximum(h-y2,0)])
    q=np.clip(raw,1e-6,1-1e-6)
    return np.column_stack([raw,np.log(q/(1-q)),bw*bh/(w*h),np.log(np.maximum(bw,1e-6)/np.maximum(bh,1e-6)),bw/w,bh/h,
        (x1+x2)*.5/w,(y1+y2)*.5/h,gaps.min(axis=1)/np.maximum(np.minimum(w,h),1),
        (gaps <= (.015*np.maximum(np.maximum(w,h),1))[:,None]).sum(axis=1)])
