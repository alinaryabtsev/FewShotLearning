from glob import glob
from liver_metastasis_dataset_3D import LiverTumorsDataset3D, LIVER_LESIONS_DATASET
import sys
import os
sys.path.append("/cs/casmip/alina.ryabtsev/Tools")
from CalculateMeasures import calculate_measures_dataframe, calculate_measures


print("started GKF k7 seed 10")

predictions_list = sorted(glob(os.path.join(LIVER_LESIONS_DATASET, "*_support_analysis_k7_s10.nii.gz")))
gt_list = [p.replace("support_analysis_k7_s10.nii.gz", "seg.nii.gz") for p in predictions_list]
roi_list = [p.replace("support_analysis_k7_s10.nii.gz", "liver.nii.gz") for p in predictions_list]
calculate_measures("tumors", predictions_list, gt_list, range(1, 2), excel_path="./support_analysis_K7_seed_10.xlsx", roi_list=roi_list)

print("finished GKF k7 seed 10")
