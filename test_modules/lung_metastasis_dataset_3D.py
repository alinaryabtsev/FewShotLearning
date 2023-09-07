import pathlib
import subprocess
from dataclasses import dataclass
from typing import Literal, Tuple
from glob import glob
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from skimage.transform import resize
from tqdm import tqdm
import einops as E

LUNG_LESIONS_DATASET = "/cs/casmip/alina.ryabtsev/FewShotLearning/datasets/lung_lesions"
SCANS_FORMAT = "_scan.nii.gz"
SEGMENTATIONS_FORMAT = "_seg.nii.gz"
RESIZE_RESOLUTION = (128, 128)
CLIP_VALUES_LIVER = (-150, 150)

def preprocess_scan_and_segmentation(scan, segmentation, split):
    """
    Given a scan and a segmentation, returns an array with tuples of scan and segmentation slices
    :param scan: scan with liver
    :param segmentation: liver tumors segmentation
    :return: array (iterable) of tuples of scan and segmentation slices
    """
    # scan_data = nib.load(scan).get_fdata().astype(np.float32)
    # seg_data = nib.load(segmentation).get_fdata().astype(np.float32)
    # scan_data = np.clip(scan_data, *CLIP_VALUES_LIVER)
    # scan_data = (scan_data - scan_data.min()) / (scan_data.max() - scan_data.min())
    # scan_data = resize(scan_data, (*RESIZE_RESOLUTION, scan_data.shape[2]))
    # seg_data = resize(seg_data, (*RESIZE_RESOLUTION, seg_data.shape[2]), order=0, preserve_range=True, anti_aliasing=False)
    # seg_data = (seg_data >= 1).astype(seg_data.dtype)
    # scan_data = E.rearrange(scan_data, "H W D ->  D 1 H W")
    # seg_data = E.rearrange(seg_data, "H W D ->  D 1 H W")
    # if split == "support":
    #     seg_slices_ = np.split(seg_data, seg_data.shape[0], axis=0)
    #     scan_slices_ = np.split(scan_data, scan_data.shape[0], axis=0)
    #     seg_slices_idx = [i for i, s in enumerate(seg_slices_) if np.sum(s) > 0]
    #     seg_data = np.squeeze(np.array([s for i, s in enumerate(seg_slices_) if i in seg_slices_idx]))
    #     # seg_data = np.array([s for i, s in enumerate(seg_slices_) if i in seg_slices_idx])
    #     scan_data = np.squeeze(np.array([s for i, s in enumerate(scan_slices_) if i in seg_slices_idx]))
    #     # scan_data = np.array([s for i, s in enumerate(scan_slices_) if i in seg_slices_idx])
    #     if len(seg_data.shape) == 3:
    #         scan_data = E.rearrange(scan_data, "D H W ->  D 1 H W")
    #         seg_data = E.rearrange(seg_data, "D H W ->  D 1 H W")
    #     else:  # if there is only one slice
    #         scan_data = E.rearrange(scan_data, "H W ->  1 1 H W")
    #         seg_data = E.rearrange(seg_data, "H W ->  1 1 H W")
    # return [scan_data.copy(), seg_data.copy()]


def load_scans(split, split_i, N, path=LUNG_LESIONS_DATASET):
    rng = np.random.default_rng(42)
    p = rng.permutation(N)
    data = []
    scans = sorted(glob(os.path.join(path, f"*{SCANS_FORMAT}")))
    scans = [scans[i] for i in p]
    segmentations = sorted(glob(os.path.join(path, f"*{SEGMENTATIONS_FORMAT}")))
    segmentations = [segmentations[i] for i in p]

    if split == "support":
        iter_data = lambda: zip(scans[:split_i], segmentations[:split_i])
        total = split_i
    else:
        iter_data = lambda: zip(scans[split_i:], segmentations[split_i:])
        total = len(scans) - split_i

    with tqdm(total=total) as pbar:
        for scan, seg in iter_data():
            data.append(preprocess_scan_and_segmentation(scan, seg, split))
            pbar.update(1)
    return data, list(iter_data())


@dataclass
class LungTumorsDataset3D(Dataset):
    """Creates a dataset from CASMIP's liver metastasis training set"""
    split: Literal["support", "query"]
    label: int
    support_frac: float = 0.7

    def __post_init__(self):
        # arrange data: self.data = [(img1, seg1), (img2, seg2) ...]
        # get number of items in folder:
        N = len(glob(os.path.join(LUNG_LESIONS_DATASET, f"*{SCANS_FORMAT}")))  # directory contains both scans and segmentations.
        self.split_i = int(np.floor(self.support_frac * N))

        T = torch.from_numpy
        scans_segs, scans_segs_names = load_scans(self.split, self.split_i, N, LUNG_LESIONS_DATASET)
        self._data = [(T(x), T(y)) for x, y in scans_segs]
        self._data_files = scans_segs_names
        self._idxs = range(len(scans_segs))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        img, seg = self._data[self._idxs[idx]]
        filenames = self._data_files[self._idxs[idx]]
        return img, seg, filenames
