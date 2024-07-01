from torch import Tensor
from torch.utils.data import Dataset
import torch
import constants
import itertools
import numpy as np
from skimage import measure
from monai.inferers import SliceInferer, PatchInferer, SlidingWindowSplitter, Inferer
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
import einops as E
from skimage.util import view_as_windows
from glob import glob
import os
import nibabel as nib
from typing import List, Tuplesupport_analysis_100_clusters_k10


class SupportPreprocessor:
    """
    This class implements a preprocessor for drawing patches from the support set scans for using them as the actual
    support set for the UniverSeg model
    """

    def __init__(self, d_support: Dataset, k_shots: int = constants.K_SHOTS, device: torch.device = constants.DEVICE,
                 support_size: int = constants.NUM_OF_SAMPLED_PATCHES):
        """
        Inits the class
        :param d_support: the dataset that contains (some or all) support set scans
        :param k_shots: the number of scans to sample from
        :param device: GPU or CPU
        :param support_size: number of the patches that will be eventually in the support set
        """
        self.d_support = d_support
        self.k_shots = k_shots
        self.support_images, self.support_labels, self.filenames = zip(*itertools.islice(self.d_support, self.k_shots))
        self.support_images = torch.cat(self.support_images, dim=0).unsqueeze(0)
        self.support_labels = torch.cat(self.support_labels, dim=0).unsqueeze(0)
        self.device = device
        self.desired_support_size = support_size

    def _get_support_patches(self, patch_size: tuple[int, int] = constants.PATCH_SIZE,
                             has_FP_patches: bool = constants.HAS_FP_PATCHES) -> tuple[Tensor, Tensor]:
        """
        This function takes the slices of the support scans and extracts positive patches from them
        :param patch_size: the shape of the extracted patch
        :param has_FP_patches: a boolean that conducts wether there are FP patches to extract
        :return: the support images patches and their labels as patches
        """
        splitter = SlidingWindowSplitter(patch_size=patch_size, overlap=0.5)
        # eliminate all negative patches and concatinte the patches
        images_patches = torch.concat([t[0] for t in splitter(self.support_images[0])])
        labels_patches = torch.concat([t[0] for t in splitter(self.support_labels[0])])
        positive_patches_idx = self._get_positive_patches_idx(labels_patches)
        images_patches = images_patches[positive_patches_idx]
        labels_patches = labels_patches[positive_patches_idx]

        # add FP patches to support set (for hard mininig of the FP patches):
        if has_FP_patches:
            support_FP_images, support_FP_labels = self._get_support_set_FP_patches()
            images_patches = torch.cat((images_patches, support_FP_images[:constants.NUM_OF_FP_PATCHES]))
            labels_patches = torch.cat((labels_patches, support_FP_labels[:constants.NUM_OF_FP_PATCHES]))
        return images_patches, labels_patches

    @staticmethod
    def _get_positive_patches_idx(masks: constants.T) -> np.ndarray:
        """
        This function returns the indices of the positive patches
        :param masks: the patched labels
        :return: a set of indices
        """
        # Label connected components in the mask
        positive_masks = []
        for i, mask in enumerate(masks):
            mask = mask.reshape((1, 128, 128))
            if torch.is_tensor(mask):
                mask = mask.cpu().detach().numpy()
            labels = measure.label(mask, background=0)
            # Loop through each labeled region, if it consits some region with a big enough area - add its index
            if np.any(labels):
                positive_masks.extend(
                    i for region in measure.regionprops(labels) if region.area > constants.MIN_LESION_AREA)
        return np.unique(positive_masks).astype(int)

    def _get_support_set_FP_patches(self):
        """
        This function gets the support set and returns the FP patches from the support set
        :return: support images and labels as patches
        """
        support_predictions = glob(os.path.join(constants.LIVER_LESIONS_DATASET, "*support_FP_analysis.nii.gz"))
        support_gt = [p.replace("support_FP_analysis.nii.gz", "seg.nii.gz") for p in support_predictions]
        support_roi = [p.replace("support_FP_analysis.nii.gz", "liver.nii.gz") for p in support_predictions]
        support_images_patches = torch.Tensor([])
        support_labels_patches = torch.Tensor([])
        for pred, gt, roi in zip(support_predictions, support_gt, support_roi):
            FP_patches = self._get_FP_patches(pred, gt, roi)
            FP_patches = torch.from_numpy(FP_patches)
            support_images_patches = torch.cat((support_images_patches, FP_patches), dim=0)
            support_labels_patches = torch.cat((support_labels_patches, torch.zeros_like(FP_patches)), dim=0)
        return support_images_patches, support_labels_patches

    def _get_FP_patches(self, pred_filename, seg_filename, roi_filename, patch_size=(128, 128)):
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
        FP_patches = view_as_windows(FP, (patch_size[0], patch_size[1], 1),
                                     step=(patch_size[0] / 2, patch_size[1] / 2, 1))
        # take positive patches from FP_patches
        FP_patches = np.concatenate(FP_patches, axis=0)
        FP_patches = np.concatenate(FP_patches, axis=0)
        patches = FP_patches[self._get_positive_patches_idx(FP_patches)]
        patches = E.rearrange(patches, "N H W 1 -> N 1 H W")
        return patches

    def _filter_support_patches_by_clustering(self, patches: Tensor, labels: Tensor, num_clusters: int) -> tuple[
        Tensor, Tensor]:
        """
        Filters the support patches by clustering them and getting patches from meaningful clusters
        :param patches: the patches to sample from
        :param labels: the labels associated to the patches
        :param num_clusters: number of clusters to use in K-Means
        :return: the sampled patches and their labels by the clustering sampling
        """
        kmeans_clusters = KMeans(n_clusters=num_clusters, random_state=constants.RANDOM_SEED)
        support_labels_patches_reshaped = E.rearrange(labels, "N 1 W H -> N (H W)")
        cluster_labels = kmeans_clusters.fit_predict(support_labels_patches_reshaped)
        unique, counts = np.unique(cluster_labels, return_counts=True)
        # remove the most frequent cluster from cluster labels (contains non-useful infrormation)
        unique = np.delete(unique, np.argmax(counts))
        counts = np.delete(counts, np.argmax(counts)).astype('float64')
        # create a new distribution of the patches from the clusters that left
        probabilities = dict(zip(unique, np.true_divide(counts, np.sum(counts))))
        indices = np.where(np.isin(cluster_labels, unique))[0]
        clusters_labels_filtered = cluster_labels[indices]
        indices_probabilities = np.array([probabilities[c] for c in clusters_labels_filtered])
        indices_probabilities /= np.sum(indices_probabilities)
        sampled_patches_indices = np.random.choice(indices, size=self.desired_support_size, replace=False,
                                                   p=indices_probabilities)
        return torch.index_select(patches, 0, torch.tensor(sampled_patches_indices)).to(self.device), \
            torch.index_select(labels, 0, torch.tensor(sampled_patches_indices)).to(self.device)

    @staticmethod
    def _get_lesions_areas(support_labels_patches: Tensor) -> Tuple[List, List]:
        """
        This function gets the areas of the lesions from the support labels patches
        :param support_labels_patches: the labels of the support patches
        :return: a list of the areas of the lesions
        """
        lesions_areas = []
        lesions_indices = []
        lesions = []
        for i, seg_patch in enumerate(support_labels_patches):
            mask = seg_patch.cpu().detach().numpy()
            labels = measure.label(mask, background=0)
            # lesions.append(
            #     (i, region.area) for region in measure.regionprops(labels) if region.area > constants.MIN_LESION_AREA)
            for region in measure.regionprops(labels):
                if region.area > constants.MIN_LESION_AREA:
                    lesions.append((region.area, i))
        ## kutiu: sorted the list by area to facilitate future looping
        lesions_sorted = sorted(lesions)
        lesions_areas = [a for a, _ in lesions_sorted]
        lesions_indices = [i for _, i in lesions_sorted]
        return lesions_areas, lesions_indices

    def _sample_lesions_from_gaussian_distribution(self, lesions_areas: List, lesions_indices: List) -> np.ndarray:
        """
        This function samples patches from the support set based on the lesions areas.
        We assume that the lesions areas are normally distributed.
        :param lesions_areas: a list of the areas of the lesions
        :param lesions_indices: a list of the indices of the lesions
        :return: a list of the indices of the patches that were sampled
        """
        data_samples = np.array(lesions_areas)
        # Create a kernel density estimate (KDE) from the data
        kde = gaussian_kde(data_samples)
        # Generate points along the x-axis for the PDF plot
        bin_c, x_vals = np.histogram(data_samples, bins=np.arange(min(data_samples), max(data_samples) + 2))
        nz_idx = np.nonzero(bin_c)[0]  # kutiu: arr of populated bins indices
        nz_bin_c = bin_c[nz_idx]  # kutiu: arr of populated bins indices
        # kutiu: buffers to integrate pdf between for each populated bin
        bin_buffs = (x_vals[nz_idx[1:]] + x_vals[nz_idx[:-1]]).astype(float) / 2
        # kutiu: main loop, iterating over every integer area 0 to max-buffer and integrating pdf
        acc_pdf = np.zeros_like(nz_idx, 'float')
        area_idx = 0
        next_buff = bin_buffs[area_idx]
        for area in range((1 + bin_buffs[-1]).astype(int)):
            if area == next_buff:  # kutiu: current-bin is a buffer, its probability is to be split between populated-bin-neighbours
                acc_pdf[area_idx] += 0.5 * kde(area)[0]
                if area_idx + 1 < len(acc_pdf):
                    acc_pdf[area_idx + 1] -= 0.5 * kde(area)[0]
            if area >= next_buff:  # kutiu: current-bin is beyond buffer, update indicators accordingly
                area_idx += 1
                if area_idx < len(bin_buffs):
                    next_buff = bin_buffs[area_idx]
            acc_pdf[area_idx] += kde(area)[0]

        acc_pdf[-1] = 1 - np.sum(acc_pdf[:-1])  # kutiu: largest area lesion is assumed to complement probability to 1
        nz_bin_p = acc_pdf
        nz_bin_p = nz_bin_p / nz_bin_c  # kutiu: normalize by the number of lesions sharing the bins probability

        xvals_indices = [np.where(x_vals == area)[0][0] for area in
                         data_samples]  # kutiu: assign each integer area (bin) with a corresponding per-lesion-probability
        lesions_p = nz_bin_p[[np.where(nz_idx == xvals_idx)[0][0] for xvals_idx in
                              xvals_indices]]  # kutiu: assign each lesion with its own probability
        np.random.seed(constants.RANDOM_SEED)
        drawn_indices = np.random.choice(len(data_samples), size=constants.NUM_OF_SAMPLED_PATCHES, replace=False,
                                         p=lesions_p)  # kutiu: draw lesions from the sorted list
        return np.array(lesions_indices)[drawn_indices]  # kutiu: respective lesion indices in the original dataset

    def _filter_support_patches_by_gaussian_kernel(self, patches: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """
        This function filters the support patches by sampling them according the gaussian kernel density estimation
        :param patches: the patches to sample from
        :param labels: the labels associated to the patches
        :return: the sampled patches and their labels by the gaussian kernel sampling
        """
        lesions_areas, lesions_indices = self._get_lesions_areas(labels)
        indices = self._sample_lesions_from_gaussian_distribution(lesions_areas, lesions_indices)
        # return (torch.index_select(patches, 0, torch.tensor(indices).to(self.device)),
        #         torch.index_select(labels, 0, torch.tensor(indices).to(self.device)))
        return torch.index_select(patches, 0, torch.tensor(indices)).to(self.device), \
            torch.index_select(labels, 0, torch.tensor(indices)).to(self.device)

    def preprocess_to_patches(self, method: str) -> tuple[Tensor, Tensor]:
        """
        This function preprocesses the support set to patches according to the provided method
        :param method: a method to filter the support set's patches (clustering, gaussian kernel, no filter)
        :return: the support set patches and their labels
        """
        # get all support images patches and their labels as patches
        support_images_patches, support_labels_patches = self._get_support_patches(patch_size=constants.PATCH_SIZE,
                                                                                   has_FP_patches=constants.HAS_FP_PATCHES)
        if self.desired_support_size > len(support_images_patches):
            raise ValueError(f"Desired support size: {self.desired_support_size} is bigger than the number of support "
                             f"patches found in {self.k_shots} scans.")

        if method == constants.GAUSSIAN_KERNEL:
            return self._filter_support_patches_by_gaussian_kernel(support_images_patches, support_labels_patches)
        elif method == constants.CLUSTERING:
            return self._filter_support_patches_by_clustering(support_images_patches, support_labels_patches,
                                                              num_clusters=constants.NUM_OF_CLUSTERS)
        elif method == constants.NO_FILTER:
            return support_images_patches[:self.desired_support_size].to(self.device), \
                support_labels_patches[:self.desired_support_size].to(self.device)
        else:
            raise ValueError(f"Provided method: {method} does not exist")

    def get_support_filenames(self):
        """
        This function returns the filenames of the support set
        :return: the filenames of the support set
        """
        return self.filenames
