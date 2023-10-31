import sys
sys.path.append("/cs/casmip/alina.ryabtsev/FewShotLearning/")
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from UniverSeg.universeg import universeg
import itertools
from liver_metastasis_dataset import LiverTumorsDataset
from liver_metastasis_dataset_3D import LiverTumorsDataset3D, LIVER_LESIONS_DATASET
from lung_metastasis_dataset_3D import LungTumorsDataset3D, LUNG_LESIONS_DATASET
from monai.inferers import SliceInferer, PatchInferer, SlidingWindowSplitter
import math
import matplotlib.pyplot as plt
import einops as E
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from skimage.transform import resize
import pandas as pd
import nibabel as nib
from glob import glob
import os
sys.path.append("/cs/casmip/alina.ryabtsev/Tools")
from CalculateMeasures import calculate_measures_dataframe, calculate_measures
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
import gc

sys.path.append('UniverSeg')

# Dice metric for measuring volume agreement
def dice_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_pred = y_pred.long()
    y_true = y_true.long()
    intersection = torch.logical_and(y_true, y_pred)
    if y_pred.sum() + y_true.sum() == 0:
        return 1
    return float(format(2. * intersection.sum() / (y_true.sum() + y_pred.sum()), '.3f'))


def load_data():
    d_support = LiverTumorsDataset3D(split="support", support_frac=0.1, label=1, resize_scan=False)
    d_query = LiverTumorsDataset3D(split="query", support_frac=0.1, label=1, resize_scan=False)
    return d_support, d_query


def get_support_images(d_support, K=5):
    K = 5  # number of shots in support
    support_images, support_labels, support_filenames = zip(*itertools.islice(d_support, K))
    support_images = torch.cat(support_images, dim=0).to(device)
    support_labels = torch.cat(support_labels, dim=0).to(device)
    return support_images, support_labels, support_filenames


@torch.no_grad()
def inference_with_slice_inferer(inferer, model_inferer, model, image, label, support_images, support_labels):
    image, label = image.to(device), label.to(device)

    # inference with Monai's slice inference
    logits = inferer(image, model_inferer, model, support_images[None], support_labels[None])
    logits = torch.squeeze(logits)
    soft_pred = torch.sigmoid(logits)
    hard_pred = soft_pred.round().clip(0,1)

    # return a dictionary of all relevant variables
    return {'Soft Prediction': soft_pred,
            'Prediction': hard_pred}


@torch.no_grad()
def model_patches_inferer(image, model, support_images, support_labels):
    patch_size = (64, 64)
    splitter = SlidingWindowSplitter(patch_size=patch_size, overlap=0.5, device=device)
    patch_inferer = PatchInferer(splitter, batch_size=1, device=device)

    # eliminate all negative patches
    support_images_patches = torch.concat([t[0] for t in splitter(support_images[0])]).to(device)
    support_labels_patches = torch.concat([t[0] for t in splitter(support_labels[0])]).to(device)
    positive_patches_idx = [i for i, l in enumerate(support_labels_patches) if l.sum() > 0]
    support_labels_patches = support_labels_patches[positive_patches_idx].to(device)
    support_images_patches = support_images_patches[positive_patches_idx].to(device)
    support_images_patches = E.rearrange(support_images_patches, "P 1 H W -> 1 P 1 H W").to(device)
    support_labels_patches = E.rearrange(support_labels_patches, "P 1 H W -> 1 P 1 H W").to(device)

    return patch_inferer(
        image.to(device), model, support_images_patches, support_labels_patches
    )


def preprocess_prediction(pred, seg_name, save, save_name="pred", resize_scan=True):
    """
    This function gets a 3D scan and preprocess it to the nifti shape (W, H, D) and resizes it to 512*512*D
    :param pred: prediction of the model as a tensor
    :param seg_name: the filename of the corresponding segmentation file
    :param save: a boolean whether to save to not
    :param save_name: if saving the prediction, what is the filename of the prediction.
    :param resize: a boolean whether to resize or not
    """
    pred = E.rearrange(pred, "D W H -> W H D")
    pred = pred.cpu().numpy()
    if resize_scan:
        pred = resize(pred, (512, 512, pred.shape[2]), anti_aliasing=True)
    if save:
        pred_filename = seg_name.replace("seg", save_name)
        affine_nifti = nib.load(seg_name).affine
        nib.save(nib.Nifti1Image(pred, affine_nifti), pred_filename)
    return pred


def main():
    logging.basicConfig(filename='liver_prediction.log', encoding='utf-8', level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')
    logging.info("Started running liver prediction")
    model = universeg(pretrained=True)
    _ = model.to(device)
    print(f"Running on device: {device}")
    d_support, d_query = load_data()
    support_images, support_labels, support_filenames = get_support_images(d_support, K=5)
    slice_inferer = SliceInferer(spatial_dim=0, roi_size=(512, 512), sw_batch_size=1, progress=True, device=device)
    total = len(d_query)
    with tqdm(total=total) as pbar:
        with logging_redirect_tqdm():
            for i, pack in enumerate(d_query):
                image, label, filename = pack
                image, label = image.to(device), label.to(device)
                # inference with Monai's slice inference
                image = torch.unsqueeze(image, dim=1).to(device)
                res = inference_with_slice_inferer(slice_inferer, model_patches_inferer, model, image, label, support_images,
                                                   support_labels)
                scan_name, seg_name = filename
                hard_pred = res["Prediction"]
                logging.info(f"finished inference of scan {i+1}: {scan_name}")
                preprocess_prediction(hard_pred, seg_name, True, save_name="patches", resize_scan=False)
                logging.info(f"finished postprocessing of prediction {i+1}: {scan_name}")
                torch.cuda.empty_cache()
                gc.collect()
                pbar.update(1)


if __name__ == '__main__':
    main()