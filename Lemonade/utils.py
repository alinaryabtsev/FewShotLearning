import medpy.metric.binary as medpy_metrics
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
import constants

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
        return medpy_metrics.obj_tpr(GT_label, prediction)

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
        return medpy_metrics.obj_fpr(GT_label, prediction)


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
        countour score = 1 - (positive countour / GT contour)
        :param prediction: The tensor that contains prediction of the model
        :param GT_label: The tensor that contains the ground truth label
        :return: The contour score
        """
        # calculate the border of the prediction and the GT label
        distances, pred_border, gt_border = SegmentationMetrics.__surface_distances(prediction, GT_label)
        GT_contour = np.count_nonzero(gt_border)
        positive_contour = distances[distances > variability_threshold].sum()
        return 1 - (positive_contour / GT_contour)

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

        # extract only 1-pixel border line of objects
        result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
        reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

        # Note: scipys distance transform is calculated only inside the borders of the
        #       foreground objects, therefore the input has to be reversed
        dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
        return dt[result_border], result_border, reference_border
