import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
sys.path.append("/cs/casmip/alina.ryabtsev/FewShotLearning/")
sys.path.append('UniverSeg')
from UniverSeg.universeg import universeg
import itertools
from liver_metastasis_dataset_3D import LiverTumorsDataset3D, LIVER_LESIONS_DATASET
from monai.inferers import SliceInferer, PatchInferer, SlidingWindowSplitter, Inferer
import einops as E
from typing import TypeVar
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from skimage.transform import resize
from skimage import measure
from skimage.util import view_as_windows
from glob import glob
import nibabel as nib
import logging
sys.path.append("/cs/casmip/alina.ryabtsev/Tools")
from scipy.stats import gaussian_kde
from tqdm.contrib.logging import logging_redirect_tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATCH_SIZE=(128, 128)
T = TypeVar('T', torch.Tensor, np.ndarray)
np.random.seed(42)

def get_positive_patches_idx(masks: T) -> np.ndarray:
    # Label connected components in the mask
    positive_masks = []
    for i, mask in enumerate(masks):
        mask = mask.reshape((1, 128, 128))
        if torch.is_tensor(mask):
            mask = mask.cpu().detach().numpy()
        labels = measure.label(mask, background=0)
        # Loop through each labeled region
        if np.any(labels):
            for region in measure.regionprops(labels):
                # Check if the bounding box is entirely within the mask boundaries
                if all(0 < point < PATCH_SIZE[0] for point in (region.bbox[1:3] + region.bbox[4:])):
                    # check if region is bigger than some minimal area threshold
                    if region.area > 30:
                        positive_masks.append(i)
    return np.unique(positive_masks).astype(int)

def get_support_patches(support_images: torch.Tensor, support_labels: torch.Tensor,
                        patch_size: tuple[int, int] = (64, 64)) -> tuple[torch.Tensor, torch.Tensor]:
    splitter = SlidingWindowSplitter(patch_size=patch_size, overlap=0.5, device=device)
    # eliminate all negative patches
    images_patches = torch.concat([t[0] for t in splitter(support_images[0])]).to(device)
    labels_patches = torch.concat([t[0] for t in splitter(support_labels[0])]).to(device)
    positive_patches_idx = get_positive_patches_idx(labels_patches)
    images_patches = images_patches[positive_patches_idx].to(device)
    labels_patches = labels_patches[positive_patches_idx].to(device)
    return images_patches, labels_patches

@torch.no_grad()
def inference_with_slice_inferer(inferer: Inferer, model_inferer: Inferer, model: nn.Module, image: torch.Tensor,
                                 label: torch.Tensor, support_images: torch.Tensor,
                                 support_labels: torch.Tensor) -> dict:
    image, label = image.to(device), label.to(device)
    # inference with Monai's slice inference
    logits = inferer(image, model_inferer, model, support_images[None], support_labels[None])
    logits = torch.squeeze(logits)
    soft_pred = torch.sigmoid(logits)
    hard_pred = soft_pred.round().clip(0, 1)

    # return a dictionary of all relevant variables
    return {'Soft Prediction': soft_pred,
            'Prediction': hard_pred}


@torch.no_grad()
def model_patches_inferer(image: torch.Tensor, model: nn.Module, support_images_patches: torch.Tensor,
                          support_labels_patches) -> Inferer:
    patch_size = PATCH_SIZE
    splitter = SlidingWindowSplitter(patch_size=patch_size, overlap=0.5, device=device)
    patch_inferer = PatchInferer(splitter, batch_size=1, device=device)
    return patch_inferer(
        image.to(device), model, support_images_patches, support_labels_patches
    )


def preprocess_prediction(pred: torch.Tensor, seg_name: str, save: bool, save_name: str = "pred",
                          resize_scan: bool = True) -> np.ndarray:
    """
    This function gets a 3D scan and preprocess it to the nifti shape (W, H, D) and resizes it to 512*512*D
    :param pred: prediction of the model as a tensor
    :param seg_name: the filename of the corresponding segmentation file
    :param save: a boolean whether to save to not
    :param save_name: if saving the prediction, what is the filename of the prediction.
    :param resize_scan: a boolean whether to resize or not
    """
    pred = E.rearrange(pred, "D W H -> W H D")
    pred = pred.cpu().numpy()
    # pred = pred.astype(np.float64)
    pred = pred.astype(np.float32)
    if resize_scan:
        pred = resize(pred, (512, 512, pred.shape[2]), anti_aliasing=True)
    if save:
        pred_filename = seg_name.replace("seg", save_name)
        affine_nifti = nib.load(seg_name).affine
        nifti_to_save = nib.Nifti1Image(pred, affine_nifti)
        # nifti_to_save.header.set_data_dtype(np.unit16)
        nib.save(nifti_to_save, pred_filename)
    return pred

def main():
    logging.basicConfig(filename='liver_prediction_support_20.log', encoding='utf-8', level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')
    logging.info("Started running liver prediction")
    model = universeg(pretrained=True)
    _ = model.to(device)

    d_support = LiverTumorsDataset3D(split="support", support_frac=0.2, label=1, resize_scan=False)
    support_images, support_labels, support_filenames = zip(*itertools.islice(d_support, 20))
    support_images = torch.cat(support_images, dim=0).to(device)
    support_labels = torch.cat(support_labels, dim=0).to(device)
    support_images_patches, support_labels_patches = get_support_patches(support_images[None], support_labels[None],
                                                                         patch_size=PATCH_SIZE)
    leasions_areas = []
    lesios_indices = []
    for i, seg_patch in enumerate(support_labels_patches):
        mask = seg_patch.cpu().detach().numpy()
        labels = measure.label(mask, background=0)
        for region in measure.regionprops(labels):
            if all(0 < point < PATCH_SIZE[0] for point in (region.bbox[1:3] + region.bbox[4:])) and region.area > 30:
                leasions_areas.append(region.area)
                lesios_indices.append(i)

    # Assuming your data is stored in a NumPy array called 'data_samples'
    data_samples = np.array(leasions_areas)
    # Create a kernel density estimate (KDE) from the data
    kde = gaussian_kde(data_samples)
    # Generate points along the x-axis for the PDF plot
    x_vals = np.linspace(min(data_samples), max(data_samples), 1000)
    # Evaluate the PDF at the specified points
    pdf_values = kde(x_vals)
    indices = np.array([], dtype=int)
    num_samples = 150
    while len(indices) < 150 and num_samples < len(data_samples):
        drawn_samples = kde.resample(num_samples)
        drawn_samples = np.squeeze(drawn_samples.reshape(-1, 1))
        indices_from_samples = np.abs(data_samples[:, None] - drawn_samples).argmin(axis=0)
        indices_from_samples = np.array(lesios_indices)[np.unique(indices_from_samples).astype(int)]
        indices = np.concatenate((indices, indices_from_samples))
        num_samples += 10

    d_query = LiverTumorsDataset3D(split="query", support_frac=0.2, label=1, resize_scan=False)
    slice_inferer = SliceInferer(spatial_dim=0, roi_size=(512, 512), sw_batch_size=1, progress=True, device=device)
    total = len(d_query)
    selected_support_images_patches = torch.index_select(support_images_patches, 0, torch.tensor(indices).to(device))
    selected_support_labels_patches = torch.index_select(support_labels_patches, 0, torch.tensor(indices).to(device))
    with tqdm(total=total) as pbar:
        with logging_redirect_tqdm():
            for i, pack in enumerate(d_query):
                image, label, filename = pack
                image, label = image.to(device), label.to(device)
                print(f"Inferring on {filename[0]}")
                # inference with Monai's slice inference
                image = torch.unsqueeze(image, dim=1).to(device)
                res = inference_with_slice_inferer(slice_inferer, model_patches_inferer, model, image, label,
                                                   selected_support_images_patches, selected_support_labels_patches)
                scan_name, seg_name = filename
                hard_pred = res["Prediction"]
                assert not torch.any(torch.isnan(hard_pred)), f"Prediction contains NaNs: {scan_name}"
                logging.info(f"finished inference of scan {i + 1}: {scan_name}")
                preprocess_prediction(hard_pred, seg_name, True, save_name=f"support_analysis", resize_scan=False)
                logging.info(f"finished postprocessing of prediction {i + 1}: {scan_name}")
                torch.cuda.empty_cache()
                pbar.update(1)


if __name__ == '__main__':
    main()