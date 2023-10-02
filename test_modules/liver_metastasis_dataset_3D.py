from dataclasses import dataclass
from typing import Literal
from glob import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from dataset_utils import load_scans, SCANS_FORMAT, LIVER_LESIONS_DATASET


@dataclass
class LiverTumorsDataset3D(Dataset):
    """Creates a dataset from CASMIP's liver metastasis training set"""
    split: Literal["support", "query"]
    label: int
    dataset_path: str = LIVER_LESIONS_DATASET
    support_frac: float = 0.7
    resize_scan: bool = True

    def __post_init__(self):
        # arrange data: self.data = [(img1, seg1), (img2, seg2) ...]
        # get number of items in folder:
        N = len(glob(os.path.join(self.dataset_path, f"*{SCANS_FORMAT}")))  # directory contains both scans and segmentations.
        self.split_i = int(np.floor(self.support_frac * N))

        T = torch.from_numpy
        scans_segs, scans_segs_names = load_scans(self.split, self.split_i, N, self.dataset_path, self.resize_scan)
        self._data = [(T(x), T(y)) for x, y in scans_segs]
        self._data_files = scans_segs_names
        self._idxs = range(len(scans_segs))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        img, seg = self._data[self._idxs[idx]]
        filenames = self._data_files[self._idxs[idx]]
        return img, seg, filenames
