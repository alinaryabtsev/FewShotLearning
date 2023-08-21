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
from skimage.transform import resize
from tqdm import tqdm


LIVER_LESIONS_DATASET = "/cs/casmip/alina.ryabtsev/FewShotLearning/datasets/liver_lesions"
SCANS_FORMAT = "_scan.nii.gz"
SEGMENTATIONS_FORMAT = "_seg.nii.gz"


def preprocess_scan_and_segmentation(scan, segmentation):
    """
    Given a scan and a segmentation, returns an array with tuples of scan and segmentation slices
    :param scan: scan with liver
    :param segmentation: liver tumors segmentation
    :return: array (iterable) of tuples of scan and segmentation slices
    """
    scan_data = nib.load(scan).get_fdata()
    seg_data = nib.load(segmentation).get_fdata()
    scan_data = np.clip(scan_data, -150, 150)
    scan_data = (scan_data - scan_data.min()) / (scan_data.max() - scan_data.min())
    scan_data = resize(scan_data, (256, 256, scan_data.shape[2]))
    seg_data = resize(seg_data, (256, 256, seg_data.shape[2]))
    seg_data = (seg_data >= 1).astype(seg_data.dtype)
    scan_slices_ = np.split(scan_data, scan_data.shape[2], axis=2)
    scan_slices = [np.rot90(s).copy() for s in scan_slices_]
    seg_slices_ = np.split(seg_data, seg_data.shape[2], axis=2)
    seg_slices = [np.rot90(s).copy() for s in seg_slices_]
    return zip(scan_slices, seg_slices)  # [(slice1, seg_slice1), (slice2, seg_slice2), ...]


def load_scans(path=LIVER_LESIONS_DATASET):
    data = []
    scans = sorted(glob(os.path.join(path, f"*{SCANS_FORMAT}")))
    segmentations = sorted(glob(os.path.join(path, f"*{SEGMENTATIONS_FORMAT}")))
    iter_data = zip(scans, segmentations)
    total = len(scans)
    with tqdm(total=total) as pbar:
        for scan, seg in iter_data:
            data.extend(preprocess_scan_and_segmentation(scan, seg))
            pbar.update(1)
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
        scans_segs = load_scans(LIVER_LESIONS_DATASET)
        self._data = [(T(x), T(y)) for x, y in scans_segs]
        self._idxs = self._split_indexes()

    def _split_indexes(self):
        rng = np.random.default_rng(42)
        N = len(self._data)
        # p = rng.permutation(N)
        p = range(N)  # TODO: remove this
        i = int(np.floor(self.support_frac * N))
        return {"support": p[:i], "query": p[i:]}[self.split]

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, idx):
        img, seg = self._data[self._idxs[idx]]
        return img, seg
