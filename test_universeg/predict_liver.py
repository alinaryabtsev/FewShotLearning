import sys
from typing import List

sys.path.append("/cs/casmip/alina.ryabtsev/FewShotLearning/")
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from UniverSeg.universeg import universeg
import itertools
from liver_metastasis_dataset_3D import LiverTumorsDataset3D, LIVER_LESIONS_DATASET
from monai.inferers import SliceInferer, PatchInferer, SlidingWindowSplitter, Inferer
import einops as E
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
import nibabel as nib
from skimage import measure
sys.path.append("/cs/casmip/alina.ryabtsev/Tools")
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
import gc

sys.path.append('UniverSeg')

# PATCH_SIZE = (64, 64)  # old patch size
PATCH_SIZE = (128, 128)
K_SHOTS = 6
HALF_PRECISION = False


# Dice metric for measuring volume agreement
def dice_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_pred = y_pred.long()
    y_true = y_true.long()
    intersection = torch.logical_and(y_true, y_pred)
    if y_pred.sum() + y_true.sum() == 0:
        return 1
    return float(format(2. * intersection.sum() / (y_true.sum() + y_pred.sum()), '.3f'))


def load_data(half_precision: bool = False) -> tuple[LiverTumorsDataset3D, LiverTumorsDataset3D]:
    """
    This function loads the data from the dataset
    :return: support and query sets
    """
    d_support = LiverTumorsDataset3D(split="support", support_frac=0.1, label=1, resize_scan=False,
                                     half_precision=half_precision)
    d_query = LiverTumorsDataset3D(split="query", support_frac=0.1, label=1, resize_scan=False,
                                   half_precision=half_precision)
    return d_support, d_query


def get_support_images(d_support: LiverTumorsDataset3D, K: int = 5) -> tuple[
    torch.Tensor, torch.Tensor, tuple[str, str]]:
    """
    This function gets the support images and labels from the support set
    :param d_support: the support set
    :param K: the number of support images to take (number of shots)
    :return: support images and labels and filenames
    """
    support_images, support_labels, support_filenames = zip(*itertools.islice(d_support, K))
    support_images = torch.cat(support_images, dim=0).to(device)
    support_labels = torch.cat(support_labels, dim=0).to(device)
    return support_images, support_labels, support_filenames


def get_positive_patches_idx(masks: torch.Tensor) -> List[int]:
    # Label connected components in the mask
    positive_masks = []
    for i, mask in enumerate(masks):
        mask = mask.cpu().numpy()
        labels = measure.label(mask, background=0)

        # # Create an empty mask to store the connected components not on the boundaries
        # result_mask = np.zeros_like(mask)

        # Loop through each labeled region
        for region in measure.regionprops(labels):
            # Check if the bounding box is entirely within the mask boundaries
            if all(point > 0 and point < size for point, size in zip(region.bbox[1:3] + region.bbox[4:], PATCH_SIZE*2)):
                # Draw the region on the result mask
                # result_mask[labels == region.label] = 1
                if region.area > 30:
                    positive_masks.append(i)

        # if result_mask.sum() != 0:
        #     positive_masks.append(i)

    return np.unique(positive_masks)


def get_support_patches(support_images: torch.Tensor, support_labels: torch.Tensor,
                        patch_size: tuple[int, int] = (64, 64)) -> tuple[torch.Tensor, torch.Tensor]:
    splitter = SlidingWindowSplitter(patch_size=patch_size, overlap=0.5, device=device)
    # eliminate all negative patches
    images_patches = torch.concat([t[0] for t in splitter(support_images[0])]).to(device)
    labels_patches = torch.concat([t[0] for t in splitter(support_labels[0])]).to(device)
    # positive_patches_idx = [i for i, l in enumerate(labels_patches) if l.sum() > 50]
    positive_patches_idx = get_positive_patches_idx(labels_patches)
    labels_patches = labels_patches[positive_patches_idx].to(device)
    images_patches = images_patches[positive_patches_idx].to(device)

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
    logging.basicConfig(filename='liver_prediction.log', encoding='utf-8', level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')
    logging.info("Started running liver prediction")
    model = universeg(pretrained=True)
    _ = model.to(device)
    print(f"Running on device: {device}")
    d_support, d_query = load_data(HALF_PRECISION)
    if HALF_PRECISION:
        model = model.half()
        with torch.cuda.amp.autocast():  # 16-bit precision
            print("Running with autocast mode (16-bit precision)")
            _extracted_from_main_10(model, d_support, d_query)
    else:
        _extracted_from_main_10(model, d_support, d_query)


def _extracted_from_main_10(model: nn.Module, d_support: LiverTumorsDataset3D, d_query: LiverTumorsDataset3D):
    """
    Helper function for main
    :param model: universeg loaded model running on cuda and half precision
    """
    support_images, support_labels, support_filenames = get_support_images(d_support, K=K_SHOTS)
    support_images_patches, support_labels_patches = get_support_patches(support_images[None], support_labels[None],
                                                                         patch_size=PATCH_SIZE)
    slice_inferer = SliceInferer(spatial_dim=0, roi_size=(512, 512), sw_batch_size=1, progress=True, device=device)
    total = len(d_query)
    with tqdm(total=total) as pbar:
        with logging_redirect_tqdm():
            for i, pack in enumerate(d_query):
                image, label, filename = pack
                image, label = image.to(device), label.to(device)
                # inference with Monai's slice inference
                image = torch.unsqueeze(image, dim=1).to(device)
                res = inference_with_slice_inferer(slice_inferer, model_patches_inferer, model, image, label,
                                                   support_images_patches, support_labels_patches)
                scan_name, seg_name = filename
                hard_pred = res["Prediction"]
                assert not torch.any(torch.isnan(hard_pred)), f"Prediction contains NaNs: {scan_name}"
                logging.info(f"finished inference of scan {i + 1}: {scan_name}")
                preprocess_prediction(hard_pred, seg_name, True, save_name=f"patches_k{K_SHOTS}", resize_scan=False)
                logging.info(f"finished postprocessing of prediction {i + 1}: {scan_name}")
                torch.cuda.empty_cache()
                gc.collect()
                pbar.update(1)


if __name__ == '__main__':
    main()
