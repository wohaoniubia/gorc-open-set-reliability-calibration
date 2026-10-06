# Third-party sources and reuse terms

This update preserves the previous repository's licensing status for project-authored code and materials. No new project-wide reuse license is assigned. Third-party annotations, images, models and software retain their original terms.

## COCO

COCO 2017 validation annotations originate from the COCO Consortium. Included forms are the original `data/instances_val2017.json` and derived box/class/split CSV files.

The annotations are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated in the [COCO terms](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm). COCO does not own the photographs' copyrights. Source-image terms remain applicable; the full photograph dataset is not redistributed here. The manuscript's example figures retain provenance in `source_data/figure_data` and the COCO metadata.

Reference: T.-Y. Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014. [DOI](https://doi.org/10.1007/978-3-319-10602-1_48). [Downloads](https://cocodataset.org/#download).

## LVIS

LVIS v1 annotations originate from the LVIS dataset project and retain CC BY 4.0 terms. The cross-check subset `data/lvis_coco_val_bbox_annotations.json` retains bounding boxes on 4809 COCO validation images; segmentation masks are omitted. Small-protocol box/class CSV files and exact splits are also included.

Reference: A. Gupta, P. Dollar and R. Girshick, *LVIS: A Dataset for Large Vocabulary Instance Segmentation*, CVPR 2019. [Paper](https://arxiv.org/abs/1908.03195). [Website](https://www.lvisdataset.org/). [Dataset team's license explanation](https://groups.google.com/g/lvis-dataset/c/_bEl0DL8N4g).

## Detectors and software

Pretrained detector checkpoints are obtained separately. The repository supplies derived candidate outputs and fitted GORC acceptance models, not detector weights.

- [YOLO-World](https://github.com/AILab-CVC/YOLO-World) / [Ultralytics YOLO-World](https://docs.ultralytics.com/models/yolo-world/): recorded checkpoint `yolov8l-worldv2.pt`.
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO): recorded model `IDEA-Research/grounding-dino-tiny`.
- NumPy, pandas, SciPy, scikit-learn, joblib, threadpoolctl, psutil, Numba, llvmlite and pycocotools are installed separately; their distributions supply their own licenses.

Settings and prompts are in `config` and Online Resource 1. Extracted figures are identified in `figures/manifest.json`. Existing Word crops do not change scientific values or source-image rights. Asset extraction additionally requires Pillow; it is not needed for cached metric reproduction.
