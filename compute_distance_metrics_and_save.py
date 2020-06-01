import numpy as np
import os
import SimpleITK as sitk
# import nibabel as nib
import pandas as pd
import futils.util as futil
from find_connect_parts import largest_connected_parts



Hausdorff_list = list()
Dice_list = list()
Jaccard_list = list()
Volume_list = list()
mean_surface_dis_list = list()
median_surface_dis_list = list()
std_surface_dis_list = list()
nine5_surface_dis_list = list()


def computeQualityMeasures(lP, lT, spacing=1):
    quality = dict()
    labelPred = sitk.GetImageFromArray(lP, isVector=False)
    labelPred.SetSpacing(spacing)
    labelTrue = sitk.GetImageFromArray(lT, isVector=False)
    labelTrue.SetSpacing(spacing)

    # Hausdorff Distance
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()
    Hausdorff_list.append(quality["Hausdorff"])
    # Dice,Jaccard,Volume Similarity..
    dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    quality["dice"] = dicecomputer.GetDiceCoefficient()
    quality["jaccard"] = dicecomputer.GetJaccardCoefficient()
    quality["volume_similarity"] = dicecomputer.GetVolumeSimilarity()
    Dice_list.append(quality["dice"])
    Jaccard_list.append(quality["jaccard"])
    Volume_list.append(quality["volume_similarity"])

    # Surface distance measures
    label = 1
    ref_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(labelTrue > 0.5, squaredDistance=False, useImageSpacing=True))
    ref_surface = sitk.LabelContour(labelTrue > 0.5)
    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(labelTrue > 0.5)
    num_ref_surface_pixels = int(statistics_image_filter.GetSum())

    seg_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(labelPred > 0.5, squaredDistance=False, useImageSpacing=True))
    seg_surface = sitk.LabelContour(labelPred > 0.5)
    seg2ref_distance_map = ref_distance_map * sitk.Cast(seg_surface, sitk.sitkFloat32)
    ref2seg_distance_map = seg_distance_map * sitk.Cast(ref_surface, sitk.sitkFloat32)

    statistics_image_filter.Execute(labelPred > 0.5)
    num_seg_surface_pixels = int(statistics_image_filter.GetSum())

    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
    seg2ref_distances = seg2ref_distances + list(np.zeros(num_seg_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
    ref2seg_distances = ref2seg_distances + list(np.zeros(num_ref_surface_pixels - len(ref2seg_distances)))

    all_surface_distances = seg2ref_distances + ref2seg_distances
    quality["mean_surface_distance"] = np.mean(all_surface_distances)
    quality["median_surface_distance"] = np.median(all_surface_distances)
    quality["std_surface_distance"] = np.std(all_surface_distances)
    quality["95_surface_distance"] = np.max(all_surface_distances)
    mean_surface_dis_list.append(quality["mean_surface_distance"])
    median_surface_dis_list.append(quality["median_surface_distance"])
    std_surface_dis_list.append(quality["std_surface_distance"])
    nine5_surface_dis_list.append(quality["95_surface_distance"])

    return quality

def one_hot_encode_3D(patch, labels):

    labels = np.array(labels)  # i.e. [0,4,5,6,7,8]
    patches = []
    for i, l in enumerate(labels):
        a = np.where(patch != l, 0, 1)
        patches.append(a)

    patches = np.array(patches)
    patches = np.rollaxis(patches, 0, len(patches.shape))  # from [6, 64, 128, 128] to [64, 128, 128, 6]?

    return np.float64(patches)


task='vessel'
gdth_file_name = '/data/jjia/mt/data/vessel/valid/gdth_ct/SSc/SSc_patient_51.mhd'
pred_file_name = '/data/jjia/practice/SSc_patient_51.mhd'
gdth, _, gdth_spacing = futil.load_itk(gdth_file_name)
pred, _, pred_spacing = futil.load_itk(pred_file_name)
pred = largest_connected_parts(pred)

if task=='vessel':
    labels=[1]
elif task=='lobe':
    labels=[4,5,6,7,8]
gdth_encode = one_hot_encode_3D(gdth, labels=labels)
pred_encode = one_hot_encode_3D(pred, labels=labels)

for i in range(len(labels)):
    print('start a loop')
    pred = pred_encode[..., i]
    gdth = gdth_encode[..., i]

    quality = computeQualityMeasures(gdth,pred, spacing=gdth_spacing[::-1])
    print(quality)
    data_frame1 = pd.DataFrame({'filename': pred_file_name,
                                'Dice': Dice_list,
                                'Jaccard': Jaccard_list,
                                'Hausdorff Distance': Hausdorff_list,
                                'Mean Surface Distance': mean_surface_dis_list,
                                'Median Surface Distance': median_surface_dis_list,
                                'Std Surface Distance': std_surface_dis_list,
                                '95 Surface Distance': nine5_surface_dis_list})
    data_frame1.to_csv("quality.csv", index=False)