from typing import Callable
import numpy as np
from utils import DetectionMetrics, SegmentationMetrics
import constants


class VirtualRichard:
    def __init__(self, ranking_function: Callable = None):
        self.ranking_function = ranking_function
        self.advice = constants.RICHARDS_ADVICE

    def rank(self, predictions: np.array):
        if self.ranking_function is None:
            raise ValueError("No ranking function provided")
        return self.ranking_function(predictions)

    @staticmethod
    def evaluate_detection(predictions: np.array, GT_labels: np.array):
        """
        This function evaluates the detection of the model and returns a tuple of scores: [TP score, FP, score and FN
        score]. The scores are calculated as follows:
        TP score = TP / all positives (higher is better) - the ration between the number of true positives and the
        number of all positives
        FP score = FP / FP + TP (lower is better) - the ration between the number of false positives and the sum of the
        number of false positives and the number of true positives
        FN score = FN / all positives (lower is better) - the ration between the number of false negatives and the
        number of all positives
        :param predictions: The tensor that contains predictions of the model
        :param GT_labels: The tensor that contains the ground truth labels
        :return: A tuple of scores: [TP score, FP, score and FN score]
        """
        return DetectionMetrics.calculate_TP_score(predictions, GT_labels), \
            DetectionMetrics.calculate_FP_score(predictions, GT_labels), \
            DetectionMetrics.calculate_FN_score(predictions, GT_labels)

    @staticmethod
    def evaluate_segmentation(predictions: np.array, GT_labels: np.array):
        """
        This function evaluates the segmentation of the model and returns a score. The score is calculated as follows:
        contour score = 1 - (positive contour / GT contour)
        relative volume difference score = abs(GT volume - predicted volume) / GT volume
        :param predictions: The tensor that contains predictions of the model
        :param GT_labels: The tensor that contains the ground truth labels
        :return: A tuple of scores: [contour score, relative volume difference score
        """
        return SegmentationMetrics.calculate_contour_score(predictions, GT_labels), \
            SegmentationMetrics.calculate_relative_volume_difference_score(predictions, GT_labels)

    @staticmethod
    def resegment(predictions: np.array, GT_labels: np.array):
        """
        This function simulates the predictions of the model and returns the re-segmented predictions.
        It does it by just returning the GT labels.
        :param predictions: The tensor that contains predictions of the model
        :param GT_labels: The tensor that contains the ground truth labels
        :return: The re-segmented predictions
        """
        return GT_labels

    def ask_richard(self):
        """
        This function asks Richard for advice.
        :return: advice
        """
        return self.advice

