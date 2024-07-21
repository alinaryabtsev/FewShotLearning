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
        return [DetectionMetrics.calculate_TP_score(p, g) for p, g in zip(predictions, GT_labels)], \
            [DetectionMetrics.calculate_FP_score(p, g) for p, g in zip(predictions, GT_labels)], \
            [DetectionMetrics.calculate_FN_score(p, g) for p, g in zip(predictions, GT_labels)]

    @staticmethod
    def evaluate_segmentation(predictions: np.array, GT_labels: np.array, variability_th: int):
        """
        This function evaluates the segmentation of the model and returns a score. The score is calculated as follows:
        countour score = 1 - (positive countour / GT contour)
        relative volume difference score = abs(GT volume - predicted volume) / GT volume
        :param predictions: The tensor that contains predictions of the model
        :param GT_labels: The tensor that contains the ground truth labels
        :param variability_th: The threshold that determines the variability of the segmentation to the GT
        :return: A tuple of scores: [countour score, relative volume difference score
        """
        return [SegmentationMetrics.calculate_contour_score(p, g, variability_th) for p, g in zip(predictions, GT_labels)], \
            [list(SegmentationMetrics.calculate_contour_score_per_slice(p, g, variability_th)) for p, g in
             zip(predictions, GT_labels)]

    @staticmethod
    def resegment(predictions: np.array, GT_labels: np.array):
        """
        This function simulates the predictions of the model and returns the resegmented predictions.
        It does it by just returning the GT labels.
        :param predictions: The tensor that contains predictions of the model
        :param GT_labels: The tensor that contains the ground truth labels
        :return: The resegmented predictions
        """
        return GT_labels

    def ask_richard(self):
        """
        This function asks Richard for advice.
        :return: advice
        """
        return self.advice
