import numpy as np
import os
import SimpleITK as sitk
# import nibabel as nib
import pandas as pd
import futils.util as futil




Hausdorff_list = list()
AvgHausdorff_list = list()
Dice_list = list()
Jaccard_list = list()
Volume_list = list()
False_negative_list = list()
False_positive_list = list()
mean_surface_dis_list = list()
median_surface_dis_list = list()
std_surface_dis_list = list()
max_surface_dis_list = list()

def write_dices_to_csv(labels, gdth_path, pred_path, csv_file, gdth_extension='.nrrd', pred_extension='.nrrd'):
    pass


def load_scan(file_name):

    """Load mhd or nrrd 3d scan"""

    extension = file_name.split('.')[-1]
    if extension == 'mhd':
        scan, origin, spacing = futil.load_itk(file_name)

    elif extension == 'nrrd':
        scan, origin, spacing = futil.load_nrrd(file_name)

    return np.expand_dims(scan, axis=-1), spacing


def file_name(file_dir):
    L = []
    path_list = os.listdir(file_dir)
    path_list.sort()  # 对读取的路径进行排序
    for filename in path_list:
        if 'nii.gz' in filename:
            L.append(os.path.join(filename))
    return L


def computeQualityMeasures(lP, lT, spacing=1):
    quality = dict()
    labelPred = sitk.GetImageFromArray(lP, isVector=False)
    labelTrue = sitk.GetImageFromArray(lT, isVector=False)
    # Hausdorff Distance
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    quality["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
    quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()
    AvgHausdorff_list.append(quality["avgHausdorff"])
    Hausdorff_list.append(quality["Hausdorff"])
    # Dice,Jaccard,Volume Similarity..
    dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    quality["dice"] = dicecomputer.GetDiceCoefficient()
    quality["jaccard"] = dicecomputer.GetJaccardCoefficient()
    quality["volume_similarity"] = dicecomputer.GetVolumeSimilarity()
    quality["false_negative"] = dicecomputer.GetFalseNegativeError()
    quality["false_positive"] = dicecomputer.GetFalsePositiveError()
    Dice_list.append(quality["dice"])
    Jaccard_list.append(quality["jaccard"])
    Volume_list.append(quality["volume_similarity"])
    False_negative_list.append(quality["false_negative"])
    False_positive_list.append(quality["false_positive"])

    # Surface distance measures
    label = 1
    ref_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(labelTrue > 0.5, squaredDistance=False))
    ref_surface = sitk.LabelContour(labelTrue > 0.5)
    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(labelTrue > 0.5)
    num_ref_surface_pixels = int(statistics_image_filter.GetSum())

    seg_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(labelPred > 0.5, squaredDistance=False))
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
    quality["max_surface_distance"] = np.max(all_surface_distances)
    mean_surface_dis_list.append(quality["mean_surface_distance"])
    median_surface_dis_list.append(quality["median_surface_distance"])
    std_surface_dis_list.append(quality["std_surface_distance"])
    max_surface_dis_list.append(quality["max_surface_distance"])

    return quality

def one_hot_encode_3D(patch, labels):

    # assert len(patch.shape)==5 # (5, 128, 128, 64, 1)
    labels = np.array(labels)  # i.e. [0,4,5,6,7,8]
    N_classes = labels.size  # 6, similiar with len(labels)
    if len(patch.shape) == 5:  # (5, 128, 128, 64, 1)
        patch = np.reshape(patch, (patch.shape[0], patch.shape[1], patch.shape[2], patch.shape[3]))
    elif len(patch.shape) == 4:  # (128, 128, 64, 1)
        patch = np.reshape(patch, (patch.shape[0], patch.shape[1], patch.shape[2]))
    patches = []
    # print('patch.shape', patch.shape)
    for i, l in enumerate(labels):
        a = np.where(patch != l, 0, 1)
        patches.append(a)

    patches = np.array(patches)
    patches = np.rollaxis(patches, 0, len(patches.shape))  # from [6, 64, 128, 128] to [64, 128, 128, 6]?

    # print('patches.shape after on hot encode 3D', patches.shape)

    return np.float64(patches)
gdth_file_name = '/data/jjia/mt/data/vessel/valid/gdth_ct/SSc_patient_51.mhd'
pred_file_name = '/data/jjia/practice/SSc_patient_51.mhd'
gdth, gdth_spacing = load_scan(gdth_file_name)
pred, pred_spacing = load_scan(pred_file_name)

labels=[1]
gdth_encode = one_hot_encode_3D(gdth, labels=labels)
pred_encode = one_hot_encode_3D(pred, labels=labels)

for i in range(len(labels)):
    print('start a loop')
    pred = pred_encode[..., i]
    gdth = gdth_encode[..., i]

    quality = computeQualityMeasures(gdth,pred, spacing=gdth_spacing)
    print(quality)
    data_frame1 = pd.DataFrame({'filename': pred_file_name, 'Dice': Dice_list, 'Jaccard': Jaccard_list,
                                'False Negative': False_negative_list, 'False Positive': False_positive_list,
                                'Hausdorff Distance': Hausdorff_list, 'avgHausdorff Distance': AvgHausdorff_list,
                                'Mean Surface Distance': mean_surface_dis_list,
                                'Median Surface Distance': median_surface_dis_list,
                                'Std Surface Distance': std_surface_dis_list, 'Max Surface Distance': max_surface_dis_list})
    data_frame1.to_csv("quality.csv", index=False)