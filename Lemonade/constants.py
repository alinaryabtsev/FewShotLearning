SAVE_NAME = "support_analysis_10"
SUPPORT_FRAC = 0.1
K_SHOTS = 10
MIN_LESION_AREA = 30  # usually 30
LOGGER_NAME = "liver_prediction_support_analysis_10.log"
PATCH_SIZE = (128, 128)
np.random.seed(42)
NUM_OF_SAMPLED_PATCHES = 450  # maximum number of support patches that can be inserted into the GPU 4090 memory
NUM_OF_FP_PATCHES = 0
HAS_FP_PATCHES = bool(NUM_OF_FP_PATCHES)

# support patches prepeocessing and selection methods:
CLUSTERING = "clustering"
GAUSSIAN_KERNEL = "gaussian kernel"
NO_FILTER = "no filter"


import torch
import numpy as np
from typing import TypeVar
T = TypeVar('T', torch.Tensor, np.ndarray)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOFT_PREDICTION = "Soft Prediction"
HARD_PREDICTION = "Prediction"

VARIABILITY_THRESHOLD = 3

RANDOM_SEED = 42
RICHARDS_ADVICE = "Richard says: 'Radiologists are the best. And I am the best radiologist'"