from glob import glob
import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
from tqdm import tqdm
import einops as E
# import the constants from the constants.py file that in parent directory
import sys
sys.path.append("..")
import constants
# from .. import constants


def preprocess_scan_and_segmentation(scan, segmentation, split, scan_resize, clip_values, half_precision=False):
    """
    Given a scan and a segmentation, returns an array with tuples of scan and segmentation slices
    :param scan: scan with liver
    :param segmentation: liver tumors segmentation
    :return: array (iterable) of tuples of scan and segmentation slices
    """
    scan_data = nib.load(scan).get_fdata().astype(np.float32)
    seg_data = nib.load(segmentation).get_fdata().astype(np.float32)
    scan_data = np.clip(scan_data, *clip_values)
    scan_data = (scan_data - scan_data.min()) / (scan_data.max() - scan_data.min())
    if scan_resize:
        scan_data = resize(scan_data, (*constants.RESIZE_RESOLUTION, scan_data.shape[2]))
        seg_data = resize(seg_data, (*constants.RESIZE_RESOLUTION, seg_data.shape[2]), order=0, preserve_range=True,
                          anti_aliasing=False)
    seg_data = (seg_data >= 1).astype(seg_data.dtype)
    scan_data = E.rearrange(scan_data, "H W D ->  D 1 H W")
    seg_data = E.rearrange(seg_data, "H W D ->  D 1 H W")
    if split == "support":
        seg_slices_ = np.split(seg_data, seg_data.shape[0], axis=0)
        scan_slices_ = np.split(scan_data, scan_data.shape[0], axis=0)
        seg_slices_idx = [i for i, s in enumerate(seg_slices_) if np.sum(s) > 0]
        seg_data = np.squeeze(np.array(seg_slices_)[seg_slices_idx])
        scan_data = np.squeeze(np.array(scan_slices_)[seg_slices_idx])
        if len(seg_data.shape) == 3:
            scan_data = E.rearrange(scan_data, "D H W ->  D 1 H W")
            seg_data = E.rearrange(seg_data, "D H W ->  D 1 H W")
        else:  # if there is only one slice
            scan_data = E.rearrange(scan_data, "H W ->  1 1 H W")
            seg_data = E.rearrange(seg_data, "H W ->  1 1 H W")
    if half_precision:
        scan_data = scan_data.astype(np.float16)
        seg_data = seg_data.astype(np.int16)
    return [scan_data.copy(), seg_data.copy()]


def load_scans(split, split_i, N, path, resize_scan=True, half_precision=False):
    rng = np.random.default_rng(constants.RANDOM_SEED)
    p = rng.permutation(N)
    data = []
    scans_files = sorted(glob(os.path.join(path, f"*{constants.SCANS_FORMAT}")))
    scans_per = [scans_files[i] for i in p]
    # do not allow to have two scans or more of the same patient in the support set
    scans = []
    while len(scans) <= split_i:
        cur_scan = scans_per.pop(0)
        if os.path.basename(cur_scan)[:-22] not in [os.path.basename(s)[:-22] for s in scans]:
            scans.append(cur_scan)
        else:
            scans_per.append(cur_scan)
    scans.extend(scans_per)
    segmentations = [s.replace(constants.SCANS_FORMAT, constants.SEGMENTATIONS_FORMAT) for s in scans]

    if split == "support":
        iter_data = lambda: zip(scans[:split_i], segmentations[:split_i])
        total = split_i
    elif split == "query-support":
        iter_data = lambda: zip(scans[:split_i], segmentations[:split_i])
        total = split_i
    else:
        iter_data = lambda: zip(scans[split_i:], segmentations[split_i:])
        total = len(scans) - split_i

    if path == constants.LIVER_LESIONS_DATASET:
        clip_values = constants.CLIP_VALUES_LIVER
    elif path == constants.LUNG_LESIONS_DATASET:
        clip_values = constants.CLIP_VALUES_LUNGS
    else:
        raise ValueError("dataset path not specified")

    with tqdm(total=total) as pbar:
        for scan, seg in iter_data():
            data.append(preprocess_scan_and_segmentation(scan, seg, split, resize_scan, clip_values, half_precision))
            pbar.update(1)
    return data, list(iter_data())
