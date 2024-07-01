from VirtualRichard import VirtualRichard
from SupportPreprocessor import SupportPreprocessor
from UniverSegPedictor import UniverSegPredictor
import constants
from datasets_loaders.liver_metastasis_dataset_3D import LiverTumorsDataset3D
from torch.utils.data import Dataset
import os
import torch
from torch import nn, Tensor
import sys
from Exceptions import LemonadeExceptions
from typing import Tuple, List, Union
from UniverSeg.universeg import universeg

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
sys.path.append("/cs/casmip/alina.ryabtsev/FewShotLearning/")
sys.path.append('UniverSeg')


def obtain_dataset_for_few_shot_learning(dataset_path: str) -> tuple[Tensor, Tensor, LiverTumorsDataset3D]:
    if dataset_path == constants.LIVER_LESIONS_DATASET:
        d_support = LiverTumorsDataset3D(split="support", support_frac=constants.SUPPORT_FRAC, label=1,
                                         resize_scan=False)
        d_query = LiverTumorsDataset3D(split="query", support_frac=constants.SUPPORT_FRAC, label=1, resize_scan=False)
        # extract to function from SupportPreprocessor
        support_preprocessor = SupportPreprocessor(d_support, constants.K_SHOTS, constants.DEVICE,
                                                   constants.NUM_OF_SAMPLED_PATCHES)
        print(*support_preprocessor.get_support_filenames(), sep="\n")
        support_images_patches, support_labels_patches = support_preprocessor.preprocess_to_patches(
            constants.SUPPORT_FILTER)
        return support_images_patches, support_labels_patches, d_query
    elif dataset_path == constants.LUNG_LESIONS_DATASET:
        raise NotImplementedError("lung lesions dataset is not implemented yet")
    else:
        raise LemonadeExceptions.DatasetException


def get_universeg_model() -> nn.Module:
    return universeg(pretrained=True)


def run_few_shot_learning_model(model: nn.Module, support_dataset: Tuple[Tensor, Tensor],
                                query_dataset: Dataset):
    UniverSegPredictor(model, constants.DEVICE, support_dataset[0], support_dataset[1]).predict(query_dataset)


def post_process_predictions(predictions: torch.Tensor) -> torch.Tensor:
    pass


def evaluate_predictions(predictions: torch.Tensor, labels: torch.Tensor) -> Tensor:
    evaluator = VirtualRichard()
    pass


def train_nnUNet(train_set: torch.Tensor, test_set: torch.Tensor) -> nn.Module:
    pass


def predict_with_nnUNet(trained_model: nn.Module, test_set: torch.Tensor) -> torch.Tensor:
    pass


def main():
    # 1. Obtain the initial data for the FSL model. Should be a torch like dataset with seperation of support-query
    # in the format of the FSL model.
    # 2. Run FSL model, obtain the presictions for the query dataset. Perform post-processing also (morphology,
    # removing small enough components).
    # 3. Pass the query predictions to Virtual Richard. Virtual Richard will rank scans and will return a subgroup of
    # fixed annotations. Say - 50 percent of the query set.
    # 4. Train nnUNet with the fixed scans + support scans. Predict on the rest non-annotated scans.
    # 5. Pass the predicted scans to Virtual Richard again. Virtual Richard will reevaluate the predictions and fix a
    # subgroup.
    # 6. Perform the last two steps until fully supervised.
    # Note: Virtual Richard should comapare "times" at each execution, between annotaing from scratch and fixing
    # predicted annotations.
    model = get_universeg_model().to(constants.DEVICE)
    support_images_patches, support_labels_patches, d_query = obtain_dataset_for_few_shot_learning(
        constants.LIVER_LESIONS_DATASET)

    run_few_shot_learning_model(model, (support_images_patches, support_labels_patches), d_query)


if __name__ == '__main__':
    main()
