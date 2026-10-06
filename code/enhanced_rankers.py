"""Small output-side rankers proposed in the 2026-10-01 calibration amendment.

No image features or detector updates. All scalers, class profiles, encoders,
labels, and weights are learned on explicitly supplied fit image IDs only.
"""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from revision_core import GEOMETRY,SCORES,LOCKED_RECIPE,label_candidates,ids_hash,normalize_image_id,write_json

FAMILIES=["additive","class_interactions","class_profile","interactions_profile"]
C_VALUES=[.1,.5,1.,2.]
SHRINKAGE=20.

class EnhancedFeatures(BaseEstimator,TransformerMixin):
    def __init__(self,family="additive",variant="SCG",shrinkage=SHRINKAGE):
        self.family=family; self.variant=variant; self.shrinkage=shrinkage

    def fit(self,X,y):
        if self.family not in FAMILIES: raise ValueError(self.family)
        if self.variant not in ["SC","SCG"]: raise ValueError(self.variant)
        self.numeric_=SCORES+(GEOMETRY if self.variant=="SCG" else [])
        self.scaler_=StandardScaler().fit(X[self.numeric_])
        self.encoder_=OneHotEncoder(handle_unknown="ignore",sparse_output=False).fit(X[["pred_label"]])
        self.categories_=self.encoder_.categories_[0].tolist()
        self.use_interactions_=self.family in ["class_interactions","interactions_profile"]
        self.use_profile_=self.variant=="SCG" and self.family in ["class_profile","interactions_profile"]
        self.feature_names_=list(self.numeric_)+[f"class={c}" for c in self.categories_]
        if self.use_interactions_:
            self.feature_names_ += [f"class={c}*{f}" for c in self.categories_ for f in self.numeric_]
        if self.use_profile_:
            tp=X.loc[np.asarray(y,dtype=int)==1]
            g=tp[GEOMETRY].to_numpy(float)
            if not len(g): raise ValueError("A known-TP profile requires fit positives")
            self.global_profile_mean_=g.mean(axis=0)
            self.global_profile_second_=(g*g).mean(axis=0)
            self.global_profile_scale_=np.sqrt(np.maximum(self.global_profile_second_-self.global_profile_mean_**2,0))
            self.profile_floor_=np.maximum(.1*self.global_profile_scale_,1e-6)
            self.profiles_={}
            for label in self.categories_:
                arr=tp.loc[tp.pred_label.eq(label),GEOMETRY].to_numpy(float); n=len(arr)
                mean=(arr.sum(axis=0)+self.shrinkage*self.global_profile_mean_)/(n+self.shrinkage)
                second=((arr*arr).sum(axis=0)+self.shrinkage*self.global_profile_second_)/(n+self.shrinkage)
                scale=np.maximum(np.sqrt(np.maximum(second-mean**2,0)),self.profile_floor_)
                self.profiles_[label]=dict(n_known_tp=n,mean=mean,scale=scale)
            raw_profile=self._profile_values(X)
            self.profile_scaler_=StandardScaler().fit(raw_profile)
            self.feature_names_ += [f"profile_abs_z_{f}" for f in GEOMETRY]+["profile_diagonal_distance"]
        return self

    def _profile_values(self,X):
        g=X[GEOMETRY].to_numpy(float)
        mean=np.tile(self.global_profile_mean_,(len(X),1))
        scale=np.tile(np.maximum(self.global_profile_scale_,self.profile_floor_),(len(X),1))
        labels=X.pred_label.to_numpy()
        for label,profile in self.profiles_.items():
            mask=labels==label; mean[mask]=profile["mean"]; scale[mask]=profile["scale"]
        z=(g-mean)/scale
        return np.column_stack([np.abs(z),np.sqrt((z*z).sum(axis=1))])

    def transform(self,X):
        numeric=self.scaler_.transform(X[self.numeric_]); onehot=self.encoder_.transform(X[["pred_label"]])
        arrays=[numeric,onehot]
        if self.use_interactions_: arrays.append((onehot[:,:,None]*numeric[:,None,:]).reshape(len(X),-1))
        if self.use_profile_: arrays.append(self.profile_scaler_.transform(self._profile_values(X)))
        return np.column_stack(arrays)

    def get_feature_names_out(self,input_features=None): return np.asarray(self.feature_names_,dtype=object)

def fit_enhanced(candidates,fit_ids,gt,family="additive",variant="SCG",C=.5,supervision="full",random_state=12):
    fit_ids=set(map(normalize_image_id,fit_ids)); subset_gt=gt.loc[gt.image_id.isin(fit_ids)]
    if supervision=="known_only" and bool((~subset_gt.gt_is_known).any()):
        raise ValueError("Known-only fitting requires the sanitized known-only GT table")
    train=label_candidates(candidates,subset_gt,fit_ids); y=train.risk_label_tp.to_numpy(int)
    weights=np.ones(len(train))
    if supervision=="full":
        weights[train.risk_error_type.eq("unknown_false_accept").to_numpy()]=2.
        weights[train.risk_error_type.eq("background_false_accept").to_numpy()]=1.25
    elif supervision!="known_only": raise ValueError(supervision)
    model=Pipeline([("features",EnhancedFeatures(family,variant)),("clf",LogisticRegression(C=C,class_weight="balanced",solver="liblinear",max_iter=1000,random_state=random_state))])
    model.fit(train,y,clf__sample_weight=weights)
    features=model.named_steps["features"]
    model.revision_config_=dict(LOCKED_RECIPE,C=C,family=family,variant=variant,supervision=supervision,
        random_state=random_state,fit_ids_sha256=ids_hash(fit_ids),n_fit_images=len(fit_ids),n_fit_candidates=len(train),
        profile_shrinkage=SHRINKAGE,profile_min_scale_fraction=.1,profile_fit_source="known TP fit detections only",
        feature_names=features.feature_names_,feature_dimension=len(features.feature_names_),
        class_conditioned_geometry=bool(features.use_interactions_ and variant=="SCG"),class_score_interactions=features.use_interactions_,
        profile_geometry=features.use_profile_,taxonomy="revised",n_fit_known_tp=int(y.sum()))
    model.revision_training_labels_=train[["image_id","det_id","risk_label_tp","risk_error_type"]].copy()
    return model

def save_enhanced(model,path):
    from pathlib import Path
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); joblib.dump(model,path)
    features=model.named_steps["features"]; clf=model.named_steps["clf"]
    details=dict(model.revision_config_,coef=clf.coef_.tolist(),intercept=clf.intercept_.tolist(),
         scaler_mean=features.scaler_.mean_.tolist(),scaler_scale=features.scaler_.scale_.tolist(),categories=features.categories_)
    if features.use_profile_:
        details["profiles"]={c:dict(n_known_tp=p["n_known_tp"],mean=p["mean"].tolist(),scale=p["scale"].tolist()) for c,p in features.profiles_.items()}
        details["global_profile_mean"]=features.global_profile_mean_.tolist(); details["global_profile_scale"]=features.global_profile_scale_.tolist()
        details["profile_scaler_mean"]=features.profile_scaler_.mean_.tolist(); details["profile_scaler_scale"]=features.profile_scaler_.scale_.tolist()
    write_json(path.with_suffix(".json"),details)
    model.revision_training_labels_.to_csv(path.with_suffix(".training_labels.csv"),index=False)
