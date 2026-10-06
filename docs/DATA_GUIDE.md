# Data and evaluation guide

## Inputs and models

`config/datasets.json` maps eight stream names to data/model folders, known classes and AP tolerances. Each COCO stream retains all 5000 images, including empty detector outputs: 1000 calibration and 4000 test images. The small LVIS protocol uses 105 calibration and 195 test images.

`coco_yolo` is the primary cache; `original20` is a separately inferred control using the same vocabulary. Do not substitute the latter when reproducing Table 3. VOC20 and random vocabularies have separately inferred/calibrated outputs. Grounding DINO is calibrated separately; a fitted YOLO-World ranker is not transferred to it.

| File / fields | Meaning |
|---|---|
| `data/<stream>/candidates.csv` | One row per post-suppression candidate in preserved order |
| `det_id`, `image_id` | Stable detection/image identifiers |
| `x1`, `y1`, `x2`, `y2` | Box corners in original-image pixels |
| `pred_label`, `raw_score` | Permitted-class prediction and unmodified detector confidence |
| Remaining numeric candidate fields | Confidence and normalized geometry terms in the order defined by the core |
| `ground_truth.csv` | Known/unknown boxes with `gt_id`, `gt_label`, coordinates and Boolean `gt_is_known` |
| `known_ground_truth.csv` | Known objects only, for the lower-annotation comparison |
| `scene_split.csv` | Every image's split and dimensions, including empty-output images |
| `class_map.csv` | Category and known/unknown role mapping |

The historical `lvis_` prefix on COCO image IDs is only a normalization convention; its numeric suffix is the COCO image ID. It does not change the dataset. CSV values retain numerical round-trip precision. Candidate order must stay aligned with the archived score arrays.

`artifacts/<stream>/models/*.joblib` contains fitted scikit-learn pipelines; adjacent JSON files expose configurations and coefficients. `selected_policies.csv`, `class_prior.json`, `frozen_selected_scores.npz` and `main_metrics.csv` bind the policies to reference results. Joblib is Python serialization; use files from the verified repository revision.

## Counting rules

- Known TP: score-ordered, predicted-class, one-to-one matching at IoU 0.50.
- UFA: every annotated unknown object overlapped by at least one accepted non-TP detection, counted once per object.
- BG FP: accepted non-TP detections below the matching IoU against every known and annotated unknown box. Already matched known objects still provide support.
- URR: `1 - UFA / number_of_annotated_unknown_objects`.
- B: `3 / (1/P + 1/R + 1/URR)`, with the manuscript's zero conventions. BG errors enter through precision.
- Relative coverage: accepted count divided by that of the calibration-selected Raw policy on the same split.
- Full-list AP: project class-wise 101-recall-point AP over IoU 0.50–0.95 on the entire candidate list before acceptance. Official COCOeval and accepted-set AP are reported separately.

The project evaluator retains COCO crowd bounding boxes as ordinary boxes. This differs from official crowd/ignore handling. `data/instances_val2017.json` preserves the original metadata for the official evaluation diagnostic.

## Author inspection and crowd boxes

`source_data/manual_review_20261006` contains author ratings for 150 random RF background detections from 125 images. The counts are 112 recognizable objects, 29 parts/ambiguous regions, 3 background/texture and 6 unable to judge. Original ratings have not been relabelled.

`vocabulary_relation` is not used for reported class counts; these use `object_name`. Of 112 recognizable cases, 103 names match the 20 known classes, including 81 people. Nine names do not match; they include generic signs and body parts and are not nine confirmed distinct unknown objects.

`source_data/round5_20261006` gives the crowd diagnostic. For detection d, take the maximum `area(d intersect c) / area(d)` across all `iscrowd=1` boxes c of any category in the image. At least 0.5 means mainly inside a crowd box. This uses boxes, not masks, and does not change UFA/BG metrics.

| Policy | BG detections | Mainly inside crowd box | Fraction |
|---|---:|---:|---:|
| Raw | 2875 | 1295 | 45.0% |
| SC | 2833 | 1307 | 46.1% |
| SCG | 2848 | 1324 | 46.5% |
| RF | 2828 | 1318 | 46.6% |

The RF random sample contains 71 such boxes. Similar fractions do not establish that annotation effects leave every policy comparison unchanged.

## Precision and provenance

`source_data/manuscript_tables` and `source_data/supplementary_tables` preserve display strings extracted from DOCX, with captions and source hashes in their manifests. Empty display cells remain empty. Use `artifacts` and `source_data/final_results` for unrounded computation.

Historical hashes inside experiment records refer to the pre-anonymization workflow. `MANIFEST.json` binds current repository files. `docs/REPRODUCIBILITY_ARCHIVE_MANIFEST.json` preserves the source archive manifest and does not cover newly added public documentation/assets.
