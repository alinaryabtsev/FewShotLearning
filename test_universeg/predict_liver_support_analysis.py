import os
from sklearn.cluster import KMeans
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
from tqdm import tqdm
from skimage.transform import resize
from skimage import measure
from skimage.util import view_as_windows
from glob import glob
import nibabel as nib
import logging
from sklearn.neighbors import KernelDensity
sys.path.append("/cs/casmip/alina.ryabtsev/Tools")
from scipy.stats import gaussian_kde
from tqdm.contrib.logging import logging_redirect_tqdm

T = TypeVar('T', torch.Tensor, np.ndarray)
PATCH_SIZE = (128, 128)
MIN_LESION_AREA = 30  # usually 30
np.random.seed(42)
SUPPORT_FRAC = 0.1
K_SHOTS = 10
NUM_OF_SAMPLED_PATCHES = 450  # maximum number of support patches that can be inserted into the GPU 4090 memory
NUM_OF_FP_PATCHES = 0
FP_PATCHES = bool(NUM_OF_FP_PATCHES)
SAVE_NAME = "support_analysis_10"
LOGGER_NAME = "liver_prediction_support_analysis_10.log"


def get_FP_patches(pred_filename, seg_filename, roi_filename, patch_size=(128, 128)):
    """
    This function gets a prediction and segmentation filenames and returns the patches that are FP.
    :param pred_filename: the filename of the prediction
    :param seg_filename: the filename of the segmentation
    :param roi_filename: the filename of the ROI
    :param patch_size: the size of the patches
    :return: a list of FP patches
    """
    pred = nib.load(pred_filename).get_fdata()
    seg = nib.load(seg_filename).get_fdata()
    roi = nib.load(roi_filename).get_fdata()
    roi = roi.astype(bool)
    pred = pred.astype(bool)
    seg = seg.astype(bool)
    FP = np.logical_and(pred, np.logical_not(seg))
    FP = np.logical_and(FP, roi)
    FP_patches = view_as_windows(FP, (patch_size[0], patch_size[1], 1), step=(patch_size[0] / 2, patch_size[1] / 2, 1))

    # take positive patches from FP_patches
    FP_patches = np.concatenate(FP_patches, axis=0)
    FP_patches = np.concatenate(FP_patches, axis=0)
    # patches = np.array([x for x in FP_patches if np.any(x)])
    patches = FP_patches[get_positive_patches_idx(FP_patches)]
    patches = E.rearrange(patches, "N H W 1 -> N 1 H W")
    return patches


def get_support_set_FP_patches():
    """
    This function gets the support set and returns the FP patches from the support set0
    :return: support images and labels patches
    """
    support_predictions = glob(os.path.join(LIVER_LESIONS_DATASET, "*support_FP_analysis.nii.gz"))
    support_gt = [p.replace("support_FP_analysis.nii.gz", "seg.nii.gz") for p in support_predictions]
    support_roi = [p.replace("support_FP_analysis.nii.gz", "liver.nii.gz") for p in support_predictions]
    support_images_patches = torch.Tensor([]).to(device)
    support_labels_patches = torch.Tensor([]).to(device)
    for pred, gt, roi in zip(support_predictions, support_gt, support_roi):
        FP_patches = get_FP_patches(pred, gt, roi)
        FP_patches = torch.from_numpy(FP_patches).to(device)
        support_images_patches = torch.cat((support_images_patches, FP_patches), dim=0)
        support_labels_patches = torch.cat((support_labels_patches, torch.zeros_like(FP_patches)), dim=0)
    return support_images_patches, support_labels_patches


def get_lesions_areas(support_labels_patches):
    leasions_areas = []
    lesios_indices = []
    for i, seg_patch in enumerate(support_labels_patches):
        mask = seg_patch.cpu().detach().numpy()
        labels = measure.label(mask, background=0)
        for region in measure.regionprops(labels):
            # if all(0 < point < PATCH_SIZE[0] for point in (region.bbox[1:3] + region.bbox[4:])) and \
            if region.area > MIN_LESION_AREA:
                leasions_areas.append(i)
    return leasions_areas, lesios_indices


def sample_lesions_from_gaussian_distribution(lesions_areas, lesions_indices):
    """
    This function samples patches from the support set based on the lesions areas.
    We assume that the lesions areas are normally distributed.
    :param lesions_areas: a list of the areas of the lesions
    :param lesions_indices: a list of the indices of the lesions
    :return: a list of the indices of the patches that were sampled
    """
    # Assuming your data is stored in a NumPy array called 'data_samples'
    data_samples = np.array(lesions_areas)
    # Create a kernel density estimate (KDE) from the data
    kde = gaussian_kde(data_samples)
    # Generate points along the x-axis for the PDF plot
    x_vals = np.linspace(min(data_samples), max(data_samples), 1000)
    # Evaluate the PDF at the specified points
    pdf_values = kde(x_vals)
    # sample from the estimated density distribution
    np.random.seed(42)
    probabilities = pdf_values / np.sum(pdf_values)
    drawn_indices = np.random.choice(len(x_vals), size=NUM_OF_SAMPLED_PATCHES, replace=False, p=probabilities)
    return np.array(lesions_indices)[drawn_indices]


def sample_lesions_from_exponential_distribution(lesions_areas, lesions_indices):
    """
    This function samples patches from the support set based on the lesions areas.
    We assume that the lesions areas are distributed according to the exponential distribution.
    :param lesions_areas: a list of the areas of the lesions
    :param lesions_indices: a list of the indices of the lesions
    :return: a list of the indices of the patches that were sampled
    """
    data_samples = np.array([lesions_areas]).reshape(-1, 1)  # Your data samples
    # Bandwidth parameter for the kernel density estimate
    bandwidth = 1
    # Create a KernelDensity estimator with the custom kernel
    kde = KernelDensity(kernel='exponential', bandwidth=bandwidth)
    # Fit the estimator to the data
    kde.fit(data_samples)
    # Evaluate the density model at specific points
    x_vals = np.linspace(min(data_samples), max(data_samples), len(data_samples))
    log_dens = kde.score_samples(x_vals)
    pdf_values = np.exp(log_dens)
    # Normalize densities to form a probability distribution
    probabilities = pdf_values / np.sum(pdf_values)
    x_vals = x_vals.reshape(-1, 1)
    # Draw X samples from the estimated density distribution
    np.random.seed(42)
    drawn_indices = np.random.choice(len(x_vals), size=NUM_OF_SAMPLED_PATCHES, replace=False, p=probabilities)
    return np.array(lesions_indices)[drawn_indices]


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
            positive_masks.extend(i for region in measure.regionprops(labels) if region.area > MIN_LESION_AREA)
    return np.unique(positive_masks).astype(int)


def get_support_patches(support_images: torch.Tensor, support_labels: torch.Tensor,
                        patch_size: tuple[int, int] = (64, 64), FP_patches: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    splitter = SlidingWindowSplitter(patch_size=patch_size, overlap=0.5, device=device)
    # eliminate all negative patches
    images_patches = torch.concat([t[0] for t in splitter(support_images[0])]).to(device)
    labels_patches = torch.concat([t[0] for t in splitter(support_labels[0])]).to(device)
    positive_patches_idx = get_positive_patches_idx(labels_patches)
    images_patches = images_patches[positive_patches_idx].to(device)
    labels_patches = labels_patches[positive_patches_idx].to(device)

    # add FP patches to support set:
    if FP_patches:
        support_FP_images, support_FP_labels = get_support_set_FP_patches()
        images_patches = torch.cat((images_patches, support_FP_images[:NUM_OF_FP_PATCHES]))
        labels_patches = torch.cat((labels_patches, support_FP_labels[:NUM_OF_FP_PATCHES]))
    return images_patches, labels_patches


def get_support_patches_by_clustering(support_patches_images: torch.Tensor, support_patches_labels: torch.Tensor,
                                      num_clusters: int, support_size: int = 450) -> tuple[torch.Tensor, torch.Tensor]:
    kmeans_clusters = KMeans(n_clusters=num_clusters, random_state=42)
    support_labels_patches_reshaped = E.rearrange(support_patches_labels, "N 1 W H -> N (H W)")
    cluster_labels = kmeans_clusters.fit_predict(support_labels_patches_reshaped)
    unique, counts = np.unique(cluster_labels, return_counts=True)
    # remove the most frequent cluster from cluster labels
    unique = np.delete(unique, np.argmax(counts))
    counts = np.delete(counts, np.argmax(counts)).astype('float64')
    probabilities = dict(zip(unique, np.true_divide(counts, np.sum(counts))))

    indices = np.where(np.isin(cluster_labels, unique))[0]
    clusters_labels_filtered = cluster_labels[indices]
    indices_probabilities = np.array([probabilities[c] for c in clusters_labels_filtered])
    indices_probabilities = indices_probabilities / np.sum(indices_probabilities)
    randomized_patches_indices = np.random.choice(indices, size=support_size, replace=False, p=indices_probabilities)
    return torch.index_select(support_patches_images, 0, torch.tensor(randomized_patches_indices)), \
           torch.index_select(support_patches_labels, 0, torch.tensor(randomized_patches_indices))


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
        nib.save(nifti_to_save, pred_filename)
    return pred


def main():
    logging.basicConfig(filename=LOGGER_NAME, encoding='utf-8', level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')
    logging.info("Started running liver prediction")
    model = universeg(pretrained=True)
    _ = model.to(device)

    d_support = LiverTumorsDataset3D(split="support", support_frac=SUPPORT_FRAC, label=1, resize_scan=False)
    support_images, support_labels, support_filenames = zip(*itertools.islice(d_support, K_SHOTS))
    support_images = torch.cat(support_images, dim=0).to(device)
    support_labels = torch.cat(support_labels, dim=0).to(device)
    support_images_patches, support_labels_patches = get_support_patches(support_images[None], support_labels[None],
                                                                         patch_size=PATCH_SIZE, FP_patches=FP_PATCHES)

    # lesions_areas, lesions_indices = get_lesions_areas(support_labels_patches)
    # indices = sample_lesions_from_gaussian_distribution(lesions_areas, lesions_indices)
    # selected_support_images_patches = torch.index_select(support_images_patches, 0, torch.tensor(indices).to(device))
    # selected_support_labels_patches = torch.index_select(support_labels_patches, 0, torch.tensor(indices).to(device))
    selected_support_images_patches, selected_support_labels_patches = get_support_patches_by_clustering(support_images_patches.to("cpu"), support_labels_patches.to("cpu"), 20, NUM_OF_SAMPLED_PATCHES)

    slice_inferer = SliceInferer(spatial_dim=0, roi_size=(512, 512), sw_batch_size=1, progress=True, device=device)
    d_query = LiverTumorsDataset3D(split="query", support_frac=0.2, label=1, resize_scan=False)
    total = len(d_query)
    with tqdm(total=total) as pbar:
        with logging_redirect_tqdm():
            for i, pack in enumerate(d_query):
                image, label, filename = pack
                image, label = image.to(device), label.to(device)
                print(f"Inferring on {filename[0]}")
                # inference with Monai's slice inference
                image = torch.unsqueeze(image, dim=1).to(device)
                res = inference_with_slice_inferer(slice_inferer, model_patches_inferer, model, image, label,
                                                   selected_support_images_patches.to(device), selected_support_labels_patches.to(device))
                scan_name, seg_name = filename
                hard_pred = res["Prediction"]
                assert not torch.any(torch.isnan(hard_pred)), f"Prediction contains NaNs: {scan_name}"
                logging.info(f"finished inference of scan {i + 1}: {scan_name}")
                preprocess_prediction(hard_pred, seg_name, True, save_name=SAVE_NAME, resize_scan=False)
                logging.info(f"finished postprocessing of prediction {i + 1}: {scan_name}")
                torch.cuda.empty_cache()
                pbar.update(1)


if __name__ == '__main__':
    main()