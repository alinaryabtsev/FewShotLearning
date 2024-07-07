from glob import glob
import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
from tqdm import tqdm
import einops as E
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

SCANS_FORMAT = "_scan.nii.gz"
SEGMENTATIONS_FORMAT = "_seg.nii.gz"
RESIZE_RESOLUTION = (128, 128)

LUNG_LESIONS_DATASET = "/cs/casmip/alina.ryabtsev/FewShotLearning/datasets/lungs_lesions"
CLIP_VALUES_LUNGS = (-1000, 150)
LIVER_LESIONS_DATASET = "/cs/casmip/alina.ryabtsev/FewShotLearning/datasets/liver_lesions"
CLIP_VALUES_LIVER = (-150, 150)


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
        scan_data = resize(scan_data, (*RESIZE_RESOLUTION, scan_data.shape[2]))
        seg_data = resize(seg_data, (*RESIZE_RESOLUTION, seg_data.shape[2]), order=0, preserve_range=True, anti_aliasing=False)
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

def get_scan_fingerprint(scan, segmentation):
    """
    analyze the figerprint of a single scan
    :param scan: scan with liver
    :param segmentation: liver tumors segmentation
    :return: a DAST containing properties of the scan, inspired by nnUnet dataset fingerprint
    """
    nifti_loaded = nib.load(scan)
    # scan_data = nifti_loaded.get_fdata().astype(np.float32)
    # seg_data = nib.load(segmentation).get_fdata().astype(np.float32)

    affine = nifti_loaded.affine
    voxel_sizes = nib.affines.voxel_sizes(affine)

    scan_prop = {
        "z_spacing": voxel_sizes[2],
        "xy_spacing": voxel_sizes[0:2]
    }

    return scan_prop







def load_scans(split, split_i, N, path, resize_scan=True, half_precision=False):
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
    elif split == "query-support":
        iter_data = lambda: zip(scans[:split_i], segmentations[:split_i])
        total = split_i
    else:
        iter_data = lambda: zip(scans[split_i:], segmentations[split_i:])
        total = len(scans) - split_i

    if path == LIVER_LESIONS_DATASET:
        clip_values = CLIP_VALUES_LIVER
    elif path == LUNG_LESIONS_DATASET:
        clip_values = CLIP_VALUES_LUNGS
    else:
        raise ValueError("dataset path not specified")

    with tqdm(total=total) as pbar:
        props = []
        for scan, seg in iter_data():
            prop  =get_scan_fingerprint(scan,seg)
            props.append(prop)

            data.append(preprocess_scan_and_segmentation(scan, seg, split, resize_scan, clip_values, half_precision))
            pbar.update(1)

    with tqdm(total=total) as pbar:
        props = []
        for scan, seg in iter_data():
            prop = get_scan_fingerprint(scan,seg)
            props.append(prop)

            data.append(preprocess_scan_and_segmentation(scan, seg, split, resize_scan, clip_values, half_precision))
            pbar.update(1)

    x_median = np.percentile([prop["xy_spacing"][0] for prop in props], 50)
    y_median = np.percentile([prop["xy_spacing"][1] for prop in props], 50)
    z_median = np.percentile([prop["z_spacing"] for prop in props], 50)

    # x_percentile = np.percentile([prop["xy_spacing"][0] for prop in props], 10)
    # y_percentile = np.percentile([prop["xy_spacing"][1] for prop in props], 10)
    z_percentile = np.percentile([prop["z_spacing"] for prop in props], 10)

    data_list = list(iter_data())

    for ii, prop in enumerate(props):
        min_spacing = min(prop["z_spacing"], prop["xy_spacing"][0], prop["xy_spacing"][1])
        max_spacing = max(prop["z_spacing"], prop["xy_spacing"][0], prop["xy_spacing"][1])

        # props[ii]["is_anisotropic"] = (max_spacing/min_spacing) > 3
        # props[ii]["z_target_3d"] = z_percentile if props[ii]["is_anisotropic"] else z_median

        is_anisotropic_med = (max(x_median,y_median,z_median) / min(x_median,y_median,z_median)) > 3
        z_target = z_percentile if is_anisotropic_med else z_median

        props[ii]["xy_target"] = (x_median,y_median)
        props[ii]["z_target_3d"] = z_target

        scan = data_list[ii][0]
        segmentation = data_list[ii][1]
        scan_data = nib.load(scan).get_fdata().astype(np.float32)
        seg_data = nib.load(segmentation).get_fdata().astype(np.float32)

        max_mm = ((scan_data.shape[0]) * prop["xy_spacing"][0], (scan_data.shape[1]) * prop["xy_spacing"][1])

        x = np.arange(0, max_mm[0], props[ii]["xy_spacing"][0])
        xi = np.arange(0, max_mm[0], props[ii]["xy_target"][0])

        y = np.arange(0, max_mm[1], props[ii]["xy_spacing"][1])
        yi = np.arange(0, max_mm[1], props[ii]["xy_target"][1])

        for jj in range(0,scan_data.shape[2]):
            slc = scan_data[:, :, jj]
            z = np.squeeze(slc)
            f = interpolate.interp2d(x, y, z, kind='cubic')
            zi = f(xi, yi)

            # plt.imshow(zi)
            # plt.show()

            slc = seg_data[:, :, jj]
            z = np.squeeze(slc)
            f = interpolate.interp2d(x, y, z, kind='linear')
            zi = f(xi, yi) > 0.5

            # if np.sum(zi) >0:
                # plt.imshow(zi, cmap = 'gray')
                # plt.show()

        # need to do linear interp on segmentation

    df = pd.Series([prop["z_spacing"] for prop in props], name="z_spacing")
    print(df.describe())
    df.plot.hist(bins=50, xlim=(0, 5))

    df = pd.Series([prop["xy_spacing"][0] for prop in props], name="x_spacing")
    print(df.describe())
    print(np.percentile([prop["xy_spacing"][0] for prop in props], 50))
    print(np.percentile([prop["xy_spacing"][0] for prop in props], 10))
    df.plot.hist(bins=50, xlim=(0, 5))

    df = pd.Series([prop["xy_spacing"][1] for prop in props], name="y_spacing")
    print(df.describe())
    df.plot.hist(bins=50, xlim=(0, 5))
    plt.show()

    return data, data_list

def main():
    path = LIVER_LESIONS_DATASET

    split = "query"
    n = len(glob(os.path.join(path, f"*{SCANS_FORMAT}")))  # directory contains both scans and segmentations.

    query_frac = 0.1
    split_i = int(np.floor((1-query_frac) * n))

    load_scans(split, split_i, n, path)

if __name__ == '__main__':
    main()
