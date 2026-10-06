# GORC reproduction guide

This archive reproduces the frozen-detector output-side experiments from public-dataset candidate caches. It contains eight separate streams: the three original detector/dataset streams and five newly inferred COCO permitted-vocabulary streams. Original COCO YOLO and the new original20 control remain separate because their floating-point candidate representations and AP differ slightly.

## Run from the repository root

Use Python 3.11 and install requirements in a fresh environment. The recorded numerical dependency versions match the experiment environment. Numba accelerates the matcher; an identical Python implementation is used when Numba is absent.

```text
python -m pip install -r requirements.txt
python code/portable_cli.py smoke --dataset coco_yolo --out run_outputs/smoke
python code/portable_cli.py replay --dataset coco_yolo --out run_outputs/fixed_replay
python code/portable_cli.py refit --dataset coco_yolo --out run_outputs/refit
python code/portable_cli.py summarize --dataset coco_yolo --out run_outputs/summary
python code/benchmark_cli.py --dataset coco_yolo --out run_outputs/benchmark
```

`replay` loads fitted models and fixed calibration-selected policies, reconstructs score families, evaluates complete calibration/test populations and reconciles every selected row with the archived source result. `refit` rebuilds raw-order TP/rest calibration labels, fits the fixed class_interactions/C=0.1 recipe, rebuilds priors and the full score/threshold family, selects only on calibration, and evaluates the held-out test set. `smoke` exercises fitting/selection/evaluation on small explicit image populations and includes empty images when available. New outputs go only into `run_outputs` or the specified output directory.

Dataset names: `coco_yolo`, `coco_gdino`, `lvis_yolo`, `original20`, `voc20`, `random20_seed611`, `random20_seed612`, `random20_seed613`.

## Inputs, models and evaluation

`data/<stream>` includes minimal candidate boxes/class/raw scores and the ten stored numerical reliability terms, complete known/unknown GT, full class map, and every image in the exact split with image dimensions. The candidate, GT and split row orders are preserved. Numerical CSVs use 17 significant digits and round-trip parsing; export checks proved bitwise numeric preservation. There are no cached risk labels, overlap labels, images, or detector checkpoint weights in the candidate inputs.

`artifacts/<stream>` contains fitted models, readable coefficient metadata, calibration-selected score policies, frozen selected scores and expected metrics. `code/revision_core.py` is the complete matcher/metric/policy core with package-relative loading; `enhanced_rankers.py` is required for fitted interaction-model loading. `compiled_scorer.py` implements the equivalent class-by-numeric coefficient fold. `online_runtime.py` preserves the pure helper functions used in the recorded final timing implementation, and `benchmark_cli.py` reruns the complete raw-output procedure with 10 warmups/100 repetitions. New timings depend on the user's hardware and software environment. The trained primary model has 230 weights plus one intercept; compilation does not train or select anything.

COCO image identifiers retain the historical `lvis_` normalization prefix in the cached evaluation protocol. The numeric suffix is the original COCO image ID; this prefix does not change the dataset or merge COCO with LVIS.

The reliability matcher uses predicted-class one-to-one known TP matching, the union of all overlapping annotated unknown objects for UFA, and GT-independent annotation support for detection-level BG. AP is the project's complete-candidate, class-wise 101-point IoU0.50:0.95 evaluator, before final acceptance filtering. It is distinct from official COCOeval. The original complete `data/instances_val2017.json` is provided for the separate official COCO audit, including crowd/area/category/image metadata.

`config/final_method_lock.json` and `source_data/calibration_family_C_selection` record the fixed family/C decision made without using the enhanced held-out test results. `source_data/final_results` contains experiment tables, policy grids and checks. Historical experiment integrity hashes refer to the original pre-anonymization files; `MANIFEST.json` is the authoritative hash list for this archive.

## Images and detector checkpoints

Images and original detector weights are obtained independently and are not needed for cache replay/refitting. Official sources are [COCO](https://cocodataset.org/#download), [LVIS](https://www.lvisdataset.org/), [YOLO-World](https://github.com/AILab-CVC/YOLO-World), [Ultralytics YOLO-World documentation](https://docs.ultralytics.com/models/yolo-world/), and [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO). The YOLO checkpoint used by the recorded experiments is `yolov8l-worldv2.pt`; detector, prompt, image-size, confidence, NMS, maxDet and text-encoder settings are recorded in `config/detector_vocabulary_lock.json` and the source-data run configurations.

Optional local paths can be configured for downstream detector regeneration:

```text
python code/portable_cli.py configure --dataset original20 --images /path/to/val2017 --checkpoint /path/to/yolov8l-worldv2.pt --out run_outputs/local_detector_paths
```

This configuration command validates the paths and stores them in new local outputs. It does not imply that detector inference is required for cache reproduction. The cache workflow is the independently tested path in this archive; the checkpoint and image files remain subject to their original sources and licenses.

## Verification

`verification/clean_smoke` and `verification/clean_primary_replay` were generated in an isolated temporary checkout outside the original project. A Python audit hook blocked reads beneath the original source root. The replay reconciles all primary selected rows and scores, and the smoke test exercises calibration fitting/selection/evaluation. Package anonymity checks scan Python, Markdown, JSON, CSV and TXT content for private absolute paths and the local account name. All actual archived files are hashed in `MANIFEST.json`.

## Additional review analyses

The baseline bridge (reply Appendix A2), RF versus per-class thresholds (Table S10), LVIS annotation support (Table S21), and nested AP comparison (Table S15) can be recomputed without detector inference:

```bash
python code/reassessment_cli.py --out run_outputs/reassessment
```

The archive includes the submitted score-only fitting labels and reference row, plus the LVIS v1 bounding-box annotations on the 4809 shared COCO val2017 images. The LVIS subset contains 50,672 boxes; segmentation masks are omitted. LVIS annotations are attributed to Gupta, Dollar and Girshick (CVPR 2019) and retain their CC BY 4.0 terms. Images and detector weights remain with their original providers. Cross-annotation support is not semantic reannotation, and the supplied original UFA/BG definitions are retained.

## Author inspection of background detections

Table S21(b) describes one author's inspection of 150 randomly sampled RF background detections from 125 primary COCO test images. Anonymized ratings and sample identifiers are in `source_data/manual_review_20261006`. Counts and percentages can be checked with `python code/manual_review_summary.py`. The ratings describe visible content; they do not relabel the original COCO evaluation or estimate semantic error differences between policies. Personal identifiers and original images are excluded.


## Crowd-box and recorded-name diagnostic

Run `python code/round5_crowd_review.py --out run_outputs/crowd_review` from the extracted archive root. This calculation uses cached detections, COCO crowd bounding boxes and the anonymized author ratings. It requires no detector inference. Archived outputs are in `source_data/round5_20261006`.

For each BG detection, the diagnostic takes the largest ratio of intersection area to detection area over all COCO `iscrowd=1` boxes in that image, regardless of category. A ratio of at least 0.5 indicates that the detection lies mainly within a crowd box. This uses bounding boxes, not segmentation masks, and does not replace the paper's matching or official COCO crowd rules. The original UFA and BG FP metrics are unchanged.

The vocabulary_relation field was recorded during rating and is not used in the reported counts; class counts are based on object_name. Names are matched literally after case, underscore and whitespace normalization. The 103 matching names and nine nonmatching names among 112 recognizable cases describe recorded names. The latter include generic signs and body parts and are not nine verified distinct unknown objects. The original anonymized ratings are preserved without relabelling.
