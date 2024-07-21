import medpy.metric.binary as medpy_metrics
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage import measure
import numpy as np
from scipy.ndimage import (
    _ni_support,
    binary_erosion,
    distance_transform_edt,
    find_objects,
    generate_binary_structure,
    label,
)
from scipy import ndimage
import constants
from typing import List, Union
import nibabel as nib
from tqdm import tqdm


class DetectionMetrics:
    @staticmethod
    def calculate_TP_score(prediction: np.array, GT_label: np.array):
        """
        This function calculates the True Positive score of the model. The score is calculated as follows:
        TP score = TP / all positives (higher is better) - the ration between the number of true positives and the
        number of all positives
        :param prediction: The tensor that contains prediction of the model
        :param GT_label: The tensor that contains the ground truth label
        :return: The TP score
        """
        try:
            return medpy_metrics.obj_tpr(GT_label, prediction)
        except ZeroDivisionError as ex:
            return 1

    @staticmethod
    def calculate_FN_score(prediction: np.array, GT_label: np.array):
        """
        This function calculates the False Negative score of the model. The score is calculated as follows:
        FN score = FN / all positives (lower is better) - the ration between the number of false negatives and the
        number of all positives
        :param prediction: The tensor that contains prediction of the model
        :param GT_label: The tensor that contains the ground truth label
        :return: The FN score
        """
        return 1 - DetectionMetrics.calculate_TP_score(prediction, GT_label)

    @staticmethod
    def calculate_FP_score(prediction: np.array, GT_label: np.array):
        """
        This function calculates the False Positive score of the model. The score is calculated as follows:
        FP score = FP / all positives (lower is better) - the ration between the number of false positives and the
        number of all positives
        :param prediction: The tensor that contains prediction of the model
        :param GT_label: The tensor that contains the ground truth label
        :return: The FP score
        """
        # return medpy_metrics.obj_fpr(prediction, GT_label)
        try:
            return medpy_metrics.obj_fpr(GT_label, prediction)
        except ZeroDivisionError as ex:
            return 0


class SegmentationMetrics:
    @staticmethod
    def calculate_relative_volume_difference_score(prediction: np.array, GT_label: np.array):
        """
        This function calculates the relative volume difference score of the model. The score is calculated as follows:
        relative volume difference score = abs(GT volume - predicted volume) / GT volume
        :param prediction: The tensor that contains prediction of the model
        :param GT_label: The tensor that contains the ground truth label
        :return: The relative volume difference score
        """
        try:
            return medpy_metrics.ravd(prediction, GT_label)
        except RuntimeError as ex:
            print("DEBUG: Probably the GT label is empty")
            raise ex

    @staticmethod
    def calculate_contour_score(prediction: np.array, GT_label: np.array,
                                variability_threshold: int = constants.VARIABILITY_THRESHOLD):
        """
        This function calculates the contour score of the model. The score is calculated as follows:
        contour score = 1 - (positive contour / GT contour)
        :param variability_threshold: The threshold for the observer variability in terms of the contour (# of pixels)
        :param prediction: The tensor that contains prediction of the model
        :param GT_label: The tensor that contains the ground truth label
        :return: The contour score
        """
        # calculate the border of the prediction and the GT label
        # distances, pred_border, gt_border = SegmentationMetrics.__surface_distances(prediction, GT_label)
        distances = SegmentationMetrics.__obj_surface_distances(prediction, GT_label)
        contour_scores = []
        for dist in distances:
            pred_dist, pred_border, gt_border = dist
            negative_contour = np.count_nonzero(pred_dist[pred_dist > variability_threshold])
            contour_score = 1 - (negative_contour / len(pred_dist))
            contour_scores.append(contour_score)
        return contour_scores

    @staticmethod
    def calculate_contour_score_per_slice(prediction: np.array, GT_label: np.array,
                                          variability_threshold: int = constants.VARIABILITY_THRESHOLD):
        """
        This function calculates the contour score of the model per slice. The score is calculated as follows:
        contour score = 1 - (positive contour / GT contour) where the contour is 2D per slice
        :param prediction: The tensor that contains prediction of the model
        :param GT_label: The tensor that contains the ground truth label
        :param variability_threshold: The threshold for the observer variability in terms of the contour (# of pixels)
        :return: The average contour score per slice
        """
        for i in range(prediction.shape[-1]):
            slice_pred = prediction[..., i]
            slice_GT = GT_label[..., i]
            con = SegmentationMetrics.calculate_contour_score(slice_pred, slice_GT, variability_threshold)
            if len(con) != 0:
                yield np.mean(con)

    @staticmethod
    def __obj_surface_distances(result, reference, voxelspacing=None, connectivity=1):
        """
        The distances between the surface voxel between all corresponding binary
        objects in result and reference. Correspondence is defined as unique and at least one voxel overlap.
        """
        sds = []
        labelmap1, labelmap2, _a, _b, mapping = SegmentationMetrics.__distinct_binary_object_correspondences(
            result, reference, connectivity)
        slicers1 = find_objects(labelmap1)
        slicers2 = find_objects(labelmap2)
        for lid2, lid1 in list(mapping.items()):
            window = SegmentationMetrics.__combine_windows(slicers1[lid1 - 1], slicers2[lid2 - 1])
            object1 = labelmap1[window] == lid1
            object2 = labelmap2[window] == lid2
            sds.append(SegmentationMetrics.__surface_distances(object1, object2, voxelspacing, connectivity))
        return sds

    @staticmethod
    def __combine_windows(w1, w2):
        """
        Joins two windows (defined by tuple of slices) such that their maximum
        combined extend is covered by the new returned window.
        """
        res = [
            slice(min(s1.start, s2.start), max(s1.stop, s2.stop))
            for s1, s2 in zip(w1, w2)
        ]
        return tuple(res)

    @staticmethod
    def __distinct_binary_object_correspondences(reference, result, connectivity=1):
        """
        Determines all distinct (where connectivity is defined by the connectivity parameter
        passed to scipy's `generate_binary_structure`) binary objects in both of the input
        parameters and returns a 1to1 mapping from the labelled objects in reference to the
        corresponding (whereas a one-voxel overlap suffices for correspondence) objects in
        result.

        All stems from the problem, that the relationship is non-surjective many-to-many.

        @return (labelmap1, labelmap2, n_lables1, n_labels2, labelmapping2to1)
        """
        result = np.atleast_1d(result.astype(np.bool))
        reference = np.atleast_1d(reference.astype(np.bool))

        # binary structure
        footprint = generate_binary_structure(result.ndim, connectivity)

        # label distinct binary objects
        labelmap1, n_obj_result = label(result, footprint)
        labelmap2, n_obj_reference = label(reference, footprint)

        # find all overlaps from labelmap2 to labelmap1; collect one-to-one relationships and store all one-two-many for later processing
        slicers = find_objects(labelmap2)  # get windows of labelled objects
        mapping = {}  # mappings from labels in labelmap2 to corresponding object labels in labelmap1
        used_labels = set()  # set to collect all already used labels from labelmap2
        one_to_many = []
        for l1id, slicer in enumerate(slicers):  # iterate over object in labelmap2 and their windows
            l1id += 1  # labelled objects have ids sarting from 1
            bobj = (l1id) == labelmap2[slicer]  # find binary object corresponding to the label1 id in the segmentation
            l2ids = np.unique(labelmap1[slicer][
                                     bobj])  # extract all unique object identifiers at the corresponding positions in the reference (i.e. the mapping)
            l2ids = l2ids[l2ids != 0]
            if len(l2ids) == 1:  # one-to-one mapping: if target label not already used, add to final list of object-to-object mappings and mark target label as used
                l2id = l2ids[0]
                if l2id not in used_labels:
                    mapping[l1id] = l2id
                    used_labels.add(l2id)
            elif len(l2ids) > 1:  # one-to-many mapping: store relationship for later processing
                one_to_many.append((l1id, set(l2ids)))

        # process one-to-many mappings, always choosing the one with the least labelmap2 correspondences first
        while True:
            one_to_many = [(l1id, l2ids - used_labels) for l1id, l2ids in
                           one_to_many]  # remove already used ids from all sets
            one_to_many = [x for x in one_to_many if x[1]]  # remove empty sets
            one_to_many = sorted(one_to_many, key=lambda x: len(x[1]))  # sort by set length
            if len(one_to_many) == 0:
                break
            l2id = one_to_many[0][1].pop()  # select an arbitrary target label id from the shortest set
            mapping[one_to_many[0][0]] = l2id  # add to one-to-one mappings
            used_labels.add(l2id)  # mark target label as used
            one_to_many = one_to_many[1:]  # delete the processed set from all sets

        return labelmap1, labelmap2, n_obj_result, n_obj_reference, mapping

    @staticmethod
    def __surface_distances(result: np.array, reference: np.array, voxelspacing: int = None, connectivity: int = 1):
        """
        The distances between the surface voxel of binary objects in result and their
        nearest partner surface voxel of a binary object in reference.
        :param result: The array that contains prediction of the model
        :param reference: The array that contains the ground truth label
        :param voxelspacing: The voxelspacing of the image
        :param connectivity: The connectivity of the image
        :return: The surface distances and the borders of the result and the reference
        """
        result = np.atleast_1d(result.astype(np.bool_))
        reference = np.atleast_1d(reference.astype(np.bool_))
        if voxelspacing is not None:
            voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
            voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
            if not voxelspacing.flags.contiguous:
                voxelspacing = voxelspacing.copy()

        # binary structure
        footprint = generate_binary_structure(result.ndim, connectivity)

        # test for emptiness
        if np.count_nonzero(result) == 0:
            raise RuntimeError("The first supplied array does not contain any binary object.")
        if np.count_nonzero(reference) == 0:
            raise RuntimeError("The second supplied array does not contain any binary object.")

        # extract only 1-pixel borderline of objects
        result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
        reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

        # Note: scipys distance transform is calculated only inside the borders of the
        #       foreground objects, therefore the input has to be reversed
        dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
        return dt[result_border], result_border, reference_border


def approximate_diameter(tumor_volume):
    """
    approximate the diameter of a tumor from its volume
    :param tumor_volume: the volume of the tumor
    """
    r = ((3 * tumor_volume) / (4 * np.pi)) ** (1 / 3)
    return 2 * r


def get_connected_components(map):
    """
    Remove Small connected components from a binary mask
    :param map: the binary mask
    :return: the binary mask with small connected components removed and the number of connected components
    """
    label_img = measure.label(map)
    cc_num = label_img.max()
    cc_areas = ndimage.sum(map, label_img, range(cc_num + 1))
    area_mask = (cc_areas <= 10)
    label_img[area_mask[label_img]] = 0
    return_value = measure.label(label_img)
    return return_value, return_value.max()


def mask_by_diameter(mask, voxel_volume, diameter):
    """
    classifies predicted shapes into diameters
    :param mask: the binary mask that represents some segmentation
    :param voxel_volume: voxel volume from nifti's header
    :param diameter: diameter to classify (in mm)
    :return: a list of labels, list of masks and their indices according to diameter
    """
    labels, max_label = get_connected_components(mask)
    tumors_indices = []
    tumors_with_diameter_masked = np.zeros(mask.shape)
    for i in range(1, max_label + 1):
        current_tumor = (labels == i)
        num_of_voxels = current_tumor.sum()
        tumor_volume = num_of_voxels * voxel_volume
        approx_diameter = approximate_diameter(tumor_volume)
        if approx_diameter >= diameter:
            tumors_indices.append(i)
            tumors_with_diameter_masked[current_tumor] = 1
    tumors_with_diameter_labeled = measure.label(tumors_with_diameter_masked)
    tumors_with_diameter_labeled = (
        tumors_with_diameter_labeled,
        tumors_with_diameter_labeled.max(),
    )
    return tumors_with_diameter_labeled, tumors_with_diameter_masked, tumors_indices


def postprocess_predictions(predictions: List[np.array], save_postprocessed: bool = False,
                            predictions_affines: List[np.array] = None,
                            predictions_filenames: List[str] = None) -> np.array:
    """
    This function postprocessors the binary predictions of the model. It performs the following steps:
    1. Load the predictions
    2. Postprocess the predictions as following:
        2.1. Remove small connected components
        2.2. Fill holes in the binary mask
        2.3. Remove small connected components again
    3. Save the postprocessed predictions if needed
    4. Return the postprocessor predictions
    :param predictions: list of predictions arrays
    :param save_postprocessed: a boolean whether to save the postprocessed predictions or not
    :param predictions_affines: list of affine matrices of the predictions
    :param predictions_filenames: list of filenames of the predictions
    :return: The postprocessed predictions arrays
    """
    postprocessed_predictions = []
    for i, pred in tqdm(enumerate(predictions), total=len(predictions)):
        pred = binary_fill_holes(pred, constants.BINARY_FILLER_MATRIX.astype(pred.dtype))
        pred = remove_small_objects(pred.astype(np.bool_), constants.MIN_CONNECTED_COMPONENT).astype(pred.dtype)
        postprocessed_predictions.append(pred)
        if save_postprocessed:
            if predictions_affines is None or predictions_filenames is None:
                raise ValueError("Affines and filenames should be provided to save the postprocessed predictions")
            nib.save(nib.Nifti1Image(pred.astype("float64"), predictions_affines[i]),
                     predictions_filenames[i].replace(".nii.gz", "_postprocessed.nii.gz"))
    return postprocessed_predictions


def postprocess_predictions_per_slice(predictions: List[np.array], save_postprocessed: bool = False,
                                      predictions_affines: List[np.array] = None,
                                      predictions_filenames: List[str] = None) -> np.array:
    """
    This function postprocessors the binary predictions of the model, per slice. It performs the following steps:
    1. Load the predictions
    2. Postprocess the predictions as following:
        2.1. Remove small connected components
        2.2. Fill holes in the binary mask
        2.3. Remove small connected components again
    3. Save the postprocessed predictions if needed
    4. Return the postprocessor predictions
    :param predictions: list of predictions arrays
    :param save_postprocessed: a boolean whether to save the postprocessed predictions or not
    :param predictions_affines: list of affine matrices of the predictions
    :param predictions_filenames: list of filenames of the predictions
    :return: The postprocessed predictions arrays
    """
    postprocessed_predictions = []
    for i, pred in tqdm(enumerate(predictions), total=len(predictions)):
        post_pred = []
        for j in range(pred.shape[-1]):
            slice_pred = binary_fill_holes(pred[..., j])
            slice_pred = remove_small_objects(slice_pred).astype(slice_pred.dtype)
            post_pred.append(slice_pred)
        postprocessed_predictions.append(np.array(post_pred).astype(pred.dtype).reshape(pred.shape))
        if save_postprocessed:
            if predictions_affines is None or predictions_filenames is None:
                raise ValueError("Affines and filenames should be provided to save the postprocessed predictions")
            for j, slice_pred in enumerate(post_pred):
                nib.save(nib.Nifti1Image(slice_pred.astype("float64"), predictions_affines[i]),
                         predictions_filenames[i].replace(".nii.gz", f"_postprocessed_{j}.nii.gz"))
    return postprocessed_predictions
