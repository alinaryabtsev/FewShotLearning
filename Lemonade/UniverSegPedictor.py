from torch import nn
from torch import Tensor
from monai.inferers import Inferer, PatchInferer, SlidingWindowSplitter, SliceInferer
import torch
import constants
import numpy as np
import nibabel as nib
from skimage.transform import resize
import einops as E
from torch.utils.data import Dataset
from tqdm import tqdm


class UniverSegPredictor:
    """
    This class is a wrapper for the FSL model that is used to predict the segmentation of an image.
    It uses slice and patch inferers to predict the segmentation of an image.
    """

    def __init__(self, model: nn.Module, device: str, support_images_patches: Tensor, support_labels_patches: Tensor):
        """
        This function initializes the UniverSegPredictor class.
        :param model: The model to use for prediction, should be a nn.Module receiving 2D images, preferably UniverSeg
        :param device: The device to use for prediction
        :param support_images_patches: The support images patches to use for prediction
        :param support_labels_patches: The support labels patches to use for prediction
        """
        self.model = model
        self.device = device
        self.support_images_patches = support_images_patches.to(device)
        self.support_labels_patches = support_labels_patches.to(device)

    def predict(self, d_query: Dataset):
        """
        This function gets a dataset and predicts the segmentation of the images in the dataset.
        :param d_query: A dataset of images to predict on
        :return: The prediction of the model on the query images
        """
        slice_inferer = SliceInferer(spatial_dim=0, roi_size=(512, 512), sw_batch_size=1, progress=True,
                                     device=self.device)
        total = len(d_query)
        with tqdm(total=total) as pbar:
            for pack in d_query:
                image, label, filename = pack
                scan_name, label_name = filename
                image, label = image.to(self.device), label.to(self.device)
                print(f"Inferring on {filename[0]}")
                # inference with Monai's slice inference
                image = torch.unsqueeze(image, dim=1).to(self.device)
                res = self._inference_with_slice_inferer(slice_inferer, image, label)
                hard_pred = res[constants.HARD_PREDICTION]
                assert not torch.any(torch.isnan(hard_pred)), f"Prediction contains NaNs: {scan_name}"
                self._postprocess_prediction(hard_pred, label_name, True, save_name=constants.SAVE_NAME,
                                             resize_scan=False)
                torch.cuda.empty_cache()
                pbar.update(1)

    @torch.no_grad()
    def _inference_with_slice_inferer(self, inferer: Inferer, image: torch.Tensor, label: torch.Tensor) -> dict:
        """
        This function gets an image and returns the prediction of the model on the image by infering with slices.
        :param inferer: The inferer to use
        :param image: The 3D image to predict on
        :param label: The 3D label of the image
        :return: The prediction of the model on the image
        """
        image, label = image.to(self.device), label.to(self.device)
        # inference with Monai's slice inference
        logits = inferer(image, self._model_patches_inferer)
        logits = torch.squeeze(logits)
        soft_pred = torch.sigmoid(logits)
        hard_pred = soft_pred.round().clip(0, 1)

        # return a dictionary of all relevant variables
        return {constants.SOFT_PREDICTION: soft_pred,
                constants.HARD_PREDICTION: hard_pred}

    @torch.no_grad()
    def _model_patches_inferer(self, image: Tensor) -> Inferer:
        """
        This function gets an image and returns the prediction of the model on the image by infering with patches.
        :param image: The image to predict on
        :return: The prediction of the model on the image
        """
        splitter = SlidingWindowSplitter(patch_size=constants.PATCH_SIZE, overlap=0.5, device=self.device)
        patch_inferer = PatchInferer(splitter, batch_size=1, device=self.device)
        return patch_inferer(
            image.to(self.device),
            self.model,
            self.support_images_patches[None],
            self.support_labels_patches[None]
        )

    @staticmethod
    def _postprocess_prediction(pred: Tensor, seg_name: str, save: bool, save_name: str = "pred",
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
        pred = pred.astype(np.float32)
        if resize_scan:
            pred = resize(pred, (512, 512, pred.shape[2]), anti_aliasing=True)
        if save:
            pred_filename = seg_name.replace("seg", save_name)
            affine_nifti = nib.load(seg_name).affine
            nifti_to_save = nib.Nifti1Image(pred, affine_nifti)
            nib.save(nifti_to_save, pred_filename)
        return pred
