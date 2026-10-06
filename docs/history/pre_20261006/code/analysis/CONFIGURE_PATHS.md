# Configure Local Resources

The analysis scripts expect local access to datasets, detector checkpoints, and, where available, detector-output caches. These resources are not included in this repository.

Before rerunning scripts, configure the following resources in the relevant script constants or through a local wrapper:

- `PROJECT_ROOT`: local clone of this repository.
- `COCO_ROOT`: COCO 2017 directory with annotations and `val2017` images.
- `LVIS_ROOT`: LVIS annotation and image resources used for LVIS-Clear-Mini-300.
- `DETECTOR_CACHE_ROOT`: local scored-candidate cache directory, if cached detector outputs are available.
- `MODEL_CHECKPOINT_ROOT`: local YOLO-World or Grounding DINO checkpoint directory.

Paths should be set locally and kept outside version control.
