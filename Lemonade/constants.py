SAVE_NAME = "support_analysis_10"
SAVE_SOFT_PREDICTION = True
SAVE_SOFT_PREDICTION_NAME = "soft_prediction_" + SAVE_NAME
SUPPORT_FRAC = 0.1
K_SHOTS = 10
MIN_LESION_AREA = 30  # usually 30
MIN_CONNECTED_COMPONENT = 30  # usually 10
LOGGER_NAME = "liver_prediction_support_analysis_10.log"
PATCH_SIZE = (128, 128)
NUM_OF_SAMPLED_PATCHES = 450  # maximum number of support patches that can be inserted into the GPU 4090 memory
NUM_OF_FP_PATCHES = 0
HAS_FP_PATCHES = bool(NUM_OF_FP_PATCHES)

# support patches prepeocessing and selection methods:
CLUSTERING = "clustering"
GAUSSIAN_KERNEL = "gaussian kernel"
NO_FILTER = "no filter"

# dataset paths and clip values (dataset utils)
LUNG_LESIONS_DATASET = "/cs/casmip/alina.ryabtsev/FewShotLearning/datasets/lungs_lesions"
CLIP_VALUES_LUNGS = (-1000, 150)
LIVER_LESIONS_DATASET = "/cs/casmip/alina.ryabtsev/FewShotLearning/datasets/liver_lesions"
CLIP_VALUES_LIVER = (-150, 150)

SCANS_FORMAT = "_scan.nii.gz"
SEGMENTATIONS_FORMAT = "_seg.nii.gz"
RESIZE_RESOLUTION = (128, 128)

import torch
import numpy as np
BINARY_FILLER_MATRIX = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).reshape([3, 3, 1])  # matrix for semantic operations
np.random.seed(42)
from typing import TypeVar
T = TypeVar('T', torch.Tensor, np.ndarray)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOFT_PREDICTION = "Soft Prediction"
HARD_PREDICTION = "Prediction"

VARIABILITY_THRESHOLD = 8

RANDOM_SEED = 42
RICHARDS_ADVICE = "Richard says: 'Radiologists are the best. And I am the best radiologist'"