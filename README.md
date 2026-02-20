# Car Counter / Speed Estimator (car_counter)

Small demo that detects and tracks vehicles, estimates speed, and counts crossings.

## Files
- `speed_estimator.py` — main script (uses Ultralytics YOLO for detection + tracking).
- `mask.png` — ROI mask used by the script (referenced by `MASK_PATH`).
- `yolov8n.pt` — model file (large; recommended to keep out of repo).

## Requirements
- Python 3.8+ (tested on macOS)
- Recommended packages: `ultralytics`, `opencv-python`, `cvzone`, `numpy`, `torch`

Quick install (use a virtualenv):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ultralytics opencv-python cvzone numpy torch
```

## Run
1. Ensure `MASK_PATH` in `speed_estimator.py` points to the mask (default: `mask.png`).
2. Adjust `VIDEO_PATH` (default `demo/input_short.mp4`) if needed.
3. Calibrate `pixels_per_meter` in `speed_estimator.py` for accurate speed estimates.

```bash
python3 speed_estimator.py
```

Press `q` in the display window to quit.

