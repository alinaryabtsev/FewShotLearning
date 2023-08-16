import pathlib
import subprocess
from dataclasses import dataclass
from typing import Literal, Tuple
from glob import glob
import os
import numpy as np
import nibabel as nib
import PIL
import torch
from torch.utils.data import Dataset


LIVER_LESIONS_DATASET = "/cs/casmip/alina.ryabtsev/FewShotLearning/datasets/liver_lesions"
SCANS_FORMAT = "_scan.nii.gz"
SEGMENTATIONS_FORMAT = "_seg.nii.gz"


def preprocess_scan_and_segmentation(scan, segmentation):
    """
    Given a scan and a segmentation, returns an array with tuples of scan and segmentation slices
    :param scan: scan with liver
    :param segmentation: liver tumors segmentation
    """
    # consider to resize to (256, 256)
    # normalize scan to [-150, 150]
    # copy final segmentation / scan slice?
    return []


def load_scans(path=LIVER_LESIONS_DATASET):
    data = []
    scans = glob(os.path.join(path, f"*{SCANS_FORMAT}"))
    segmentations = glob(os.path.join(path, f"*{SEGMENTATIONS_FORMAT}"))
    for scan, seg in zip(scans, segmentations):
        data.append(preprocess_scan_and_segmentation(scan, seg))
    return data


@dataclass
class LiverTumorsDataset(Dataset):
    """Creates a dataset from CASMIP's liver metastasis training set"""
    split: Literal["support", "query"]
    label: int
    support_frac: float = 0.7

    def __post_init__(self):
        # arrange data: self.data = [(img1, seg1), (img2, seg2) ...]
        T = torch.from_numpy
        self._data = [(T(x)[None], T(y)) for x, y in load_scans(LIVER_LESIONS_DATASET)]
        self._idxs = self._split_indexes()

    def _split_indexes(self):
        rng = np.random.default_rng(42)
        N = len(self._data)
        p = rng.permutation(N)
        i = int(np.floor(self.support_frac * N))
        return {"support": p[:i], "query": p[i:]}[self.split]

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, idx):
        img, seg = self._data[self._idxs[idx]]
        return img, seg
