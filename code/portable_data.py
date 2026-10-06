"""Package-relative datasets; no detector images are needed for cache reproduction."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent

def load_dataset(name,unknown_access=True):
    import revision_core as core
    configs=json.loads((ROOT/"config/datasets.json").read_text())
    if name not in configs:raise ValueError(f"Unknown dataset {name}; choose {list(configs)}")
    cfg=configs[name];folder=ROOT/cfg["data_folder"]
    read=lambda f:pd.read_csv(folder/f,encoding="utf-8-sig",float_precision="round_trip")
    candidates=read("candidates.csv");gt=read("ground_truth.csv" if unknown_access else "known_ground_truth.csv");split=read("scene_split.csv")
    known=cfg["known_classes"]
    assert len(split)==split.image_id.nunique() and not set(split.loc[split.split.eq("calibration"),"image_id"]) & set(split.loc[split.split.eq("test"),"image_id"])
    assert set(candidates.image_id)<=set(split.image_id) and set(gt.image_id)<=set(split.image_id)
    assert gt.gt_is_known.dtype==np.dtype(bool)
    if not unknown_access:
        assert gt.gt_is_known.all()
        candidates["risk_label_tp"]=core.label_candidates(candidates,gt,split.image_id.tolist(),known).risk_label_tp.to_numpy()
    audit=dict(dataset=name,unknown_access=unknown_access,cache="data/"+name+"/candidates.csv",cache_sha256=core.sha256_file(folder/"candidates.csv"),
        split_sha256=core.sha256_file(folder/"scene_split.csv"),n_candidates=len(candidates),n_images=len(split),n_empty_images=len(set(split.image_id)-set(candidates.image_id)))
    return core.Dataset(name,candidates,gt,split,known,audit)
