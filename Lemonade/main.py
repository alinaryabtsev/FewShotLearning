from VirtualRichard import VirtualRichard
from datasets_loaders.liver_metastasis_dataset_3D import LiverTumorsDataset3D
from datasets_loaders.dataset_utils import LIVER_LESIONS_DATASET, LUNG_LESIONS_DATASET
import torch
import sys
import glob
sys.path.append("/cs/casmip/alina.ryabtsev/FewShotLearning/")
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from typing import String
from Exceptions import LemonadeExceptions
sys.path.append('UniverSeg')
from UniverSeg.universeg import universeg
import constants
import itertools


def obtain_dataset_for_few_shot_learning(dataset_path: String) -> torch.Tensor:
    if dataset_path == LIVER_LESIONS_DATASET:
        d_support = LiverTumorsDataset3D(split="support", support_frac=constants.SUPPORT_FRAC, label=1, resize_scan=False)
        d_query = LiverTumorsDataset3D(split="query", support_frac=0.2, label=1, resize_scan=False)
        support_images, support_labels, support_filenames = zip(*itertools.islice(d_support, constants.K_SHOTS))
        support_images = torch.cat(support_images, dim=0).to(device)
        support_labels = torch.cat(support_labels, dim=0).to(device)
    elif dataset_path == LIVER_LESIONS_DATASET:
        pass
    else:
        raise LemonadeExceptions.DatasetException
    return

def get_universeg_model() -> nn.Module:
    model = universeg(pretrained=True)
    return model


def run_few_shot_learning_model(model: nn.Module, support_dataset: torch.Tensor, query_dataset: torch.Tensor) -> torch.Tensor:
    pass


def evaluate_predictions(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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
    model = get_universeg_model().to(device)


if __name__ == '__main__':
    main()
