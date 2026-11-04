"""Paths and constants shared by every notebook."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"
CONFUSION = RESULTS / "confusion_matrices"
LOGS = RESULTS / "logs"

for _d in (DATA_PROCESSED, MODELS_DIR, FIGURES, METRICS, CONFUSION, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

# Common signal representation for both domains
FS = 64                 # Hz, Daphnet's native rate; SisFall is downsampled to it
WIN_SEC = 2.0
WIN = int(FS * WIN_SEC)        # 128 samples
# All windows are first cut on a fine 0.25 s grid. Evaluation sets keep only the
# 1 s grid (50% overlap). Training additionally keeps the fine-grid fall windows,
# i.e. each impact is seen at four temporal offsets: a shift augmentation that
# raises the number of positives without touching the test protocol.
FINE_STRIDE = int(FS * 0.25)   # 16 samples
STRIDE = int(FS * 1.0)         # 64 samples
STRIDE_SEC = STRIDE / FS
CHANNELS = ["a_V", "a_AP", "a_ML", "|a|"]

# SisFall raw-count conversion (ADXL345, +/-16 g, 13 bit)
SISFALL_FS = 200
ADXL345_G_PER_COUNT = (2 * 16) / (2 ** 13)

# Fall labelling around the impact peak
FALL_HALF_EVENT = 1.0     # s, event interval = peak +/- 1 s
FALL_MIN_OVERLAP = 1.5    # s of overlap with the event interval to call a window positive

SEEDS = [0, 1, 2]
ARCHS = ["CNN-1D", "LSTM", "GRU", "Transformer"]

# Subject-level split for SisFall. SE06 is the only elderly subject who performed falls,
# so it has to stay in training; otherwise the model never sees an elderly fall.
SISFALL_SPLIT = {
    "train": [f"SA{i:02d}" for i in range(1, 16)] + [f"SE{i:02d}" for i in range(1, 11)],
    "val":   [f"SA{i:02d}" for i in range(16, 19)] + ["SE11", "SE12"],
    "test":  [f"SA{i:02d}" for i in range(19, 24)] + ["SE13", "SE14", "SE15"],
}

PALETTE = {
    "CNN-1D": "#4C72B0",
    "LSTM": "#DD8452",
    "GRU": "#55A868",
    "Transformer": "#C44E52",
}
