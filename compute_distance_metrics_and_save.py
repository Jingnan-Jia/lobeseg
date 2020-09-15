import numpy as np
import os
import SimpleITK as sitk
# import nibabel as nib
import pandas as pd
import futils.util as futil
from find_connect_parts import largest_connected_parts
import copy
import PySimpleGUI as gui
import matplotlib.pyplot as plt
from futils.util import get_gdth_pred_names


def show_itk(itk, idx):
    ref_surface_array = sitk.GetArrayViewFromImage(itk)
    plt.figure()
    plt.imshow(ref_surface_array[idx])
    plt.show()

    return None

def computeQualityMeasures(lP, lT, spacing):
    """

    :param lP: prediction, shape (x, y, z)
    :param lT: ground truth, shape (x, y, z)
    :param spacing: shape order (x, y, z)
    :return: quality: dict contains metircs
    """

    pred = lP.astype(int) # float data does not support bit_and and bit_or
    gdth = lT.astype(int) # float data does not support bit_and and bit_or
    fp_array = copy.deepcopy(pred) # keep pred unchanged
    fn_array = copy.deepcopy(gdth)
    gdth_sum = np.sum(gdth)
    pred_sum = np.sum(pred)
    intersection = gdth & pred
    union = gdth | pred
    intersection_sum = np.count_nonzero(intersection)
    union_sum = np.count_nonzero(union)

    tp_array = intersection

    tmp = pred - gdth
    fp_array[tmp < 1] = 0

    tmp2 = gdth - pred
    fn_array[tmp2 < 1] = 0

    tn_array = np.ones(gdth.shape) - union

    tp, fp, fn, tn = np.sum(tp_array), np.sum(fp_array), np.sum(fn_array), np.sum(tn_array)

    smooth = 0.001
    precision = tp / (pred_sum + smooth)
    recall = tp / (gdth_sum + smooth)

    false_positive_rate = fp / (fp + tn + smooth)
    false_negtive_rate = fn / (fn + tp + smooth)

    jaccard = intersection_sum / (union_sum + smooth)
    dice = 2 * intersection_sum / (gdth_sum + pred_sum + smooth)

    quality = dict()
    labelPred = sitk.GetImageFromArray(lP, isVector=False)
    labelPred.SetSpacing(spacing)
    labelTrue = sitk.GetImageFromArray(lT, isVector=False)
    labelTrue.SetSpacing(spacing) # spacing order (x, y, z)


    # Dice,Jaccard,Volume Similarity..
    dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue > 0.5, labelPred > 0.5)

    quality["dice"] = dice
    quality["jaccard"] = jaccard
    quality["precision"] = precision
    quality["recall"] = recall
    quality["false_negtive_rate"] = false_negtive_rate
    quality["false_positive_rate"] = false_positive_rate
    quality["volume_similarity"] = dicecomputer.GetVolumeSimilarity()


    slice_idx = 300
    # Surface distance measures
    signed_distance_map = sitk.SignedMaurerDistanceMap(labelTrue > 0.5, squaredDistance=False, useImageSpacing=True) # It need to be adapted.
    # show_itk(signed_distance_map, slice_idx)

    ref_distance_map = sitk.Abs(signed_distance_map)
    # show_itk(ref_distance_map, slice_idx)



    ref_surface = sitk.LabelContour(labelTrue > 0.5, fullyConnected=True)
    # show_itk(ref_surface, slice_idx)
    ref_surface_array = sitk.GetArrayViewFromImage(ref_surface)


    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(ref_surface > 0.5)

    num_ref_surface_pixels = int(statistics_image_filter.GetSum())

    signed_distance_map_pred = sitk.SignedMaurerDistanceMap(labelPred > 0.5, squaredDistance=False, useImageSpacing=True)
    # show_itk(signed_distance_map_pred, slice_idx)

    seg_distance_map = sitk.Abs(signed_distance_map_pred)
    # show_itk(seg_distance_map, slice_idx)

    seg_surface = sitk.LabelContour(labelPred > 0.5, fullyConnected=True)
    # show_itk(seg_surface, slice_idx)
    seg_surface_array = sitk.GetArrayViewFromImage(seg_surface)

    seg2ref_distance_map = ref_distance_map * sitk.Cast(seg_surface, sitk.sitkFloat32)
    # show_itk(seg2ref_distance_map, slice_idx)

    ref2seg_distance_map = seg_distance_map * sitk.Cast(ref_surface, sitk.sitkFloat32)
    # show_itk(ref2seg_distance_map, slice_idx)


    statistics_image_filter.Execute(seg_surface > 0.5)

    num_seg_surface_pixels = int(statistics_image_filter.GetSum())

    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
    seg2ref_distances = seg2ref_distances + list(np.zeros(num_seg_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
    ref2seg_distances = ref2seg_distances + list(np.zeros(num_ref_surface_pixels - len(ref2seg_distances))) #

    all_surface_distances = seg2ref_distances + ref2seg_distances
    quality["mean_surface_distance"] = np.mean(all_surface_distances)
    quality["median_surface_distance"] = np.median(all_surface_distances)
    quality["std_surface_distance"] = np.std(all_surface_distances)
    quality["95_surface_distance"] = np.percentile(all_surface_distances, 95)
    quality["Hausdorff"] = np.max(all_surface_distances)

    return quality


def get_metrics_dict_all_labels(labels, gdth, pred, spacing):
    """

    :param labels: not include background, e.g. [4,5,6,7,8] or [1]
    :param gdth: shape: (x, y, z, channels), channels is equal to len(labels) or equal to len(labels)+1 (background)
    :param pred: the same as above
    :param spacing: spacing order should be (x, y, z) !!!
    :return: metrics_dict_all_labels a dict which contain all metrics
    """


    Hausdorff_list = []
    Dice_list = []
    Jaccard_list = []
    Volume_list = []
    mean_surface_dis_list = []
    median_surface_dis_list = []
    std_surface_dis_list = []
    nine5_surface_dis_list = []
    precision_list = []
    recall_list = []
    false_positive_rate_list = []
    false_negtive_rate_list = []

    for i, label in enumerate(labels):
        print('start get metrics for label: ', label)
        pred_per = pred[..., i]  # select onlabel
        gdth_per = gdth[..., i]

        metrics = computeQualityMeasures(pred_per, gdth_per, spacing=spacing)
        print(metrics)

        Dice_list.append(metrics["dice"])
        Jaccard_list.append(metrics["jaccard"])
        precision_list.append(metrics["precision"])
        recall_list.append(metrics["recall"])
        false_negtive_rate_list.append(metrics["false_negtive_rate"])
        false_positive_rate_list.append(metrics["false_positive_rate"])
        Volume_list.append(metrics["volume_similarity"])
        mean_surface_dis_list.append(metrics["mean_surface_distance"])
        median_surface_dis_list.append(metrics["median_surface_distance"])
        std_surface_dis_list.append(metrics["std_surface_distance"])
        nine5_surface_dis_list.append(metrics["95_surface_distance"])
        Hausdorff_list.append(metrics["Hausdorff"])

    metrics_dict_all_labels = {'Dice': Dice_list,
                               'Jaccard': Jaccard_list,
                               'precision': precision_list,
                               'recall': recall_list,
                               'false_positive_rate': false_positive_rate_list,
                               'false_negtive_rate': false_negtive_rate_list,
                               'volume': Volume_list,
                               'Hausdorff Distance': Hausdorff_list,
                               'Mean Surface Distance': mean_surface_dis_list,
                               'Median Surface Distance': median_surface_dis_list,
                               'Std Surface Distance': std_surface_dis_list,
                               '95 Surface Distance': nine5_surface_dis_list}

    return metrics_dict_all_labels

def one_hot_encode_3D(patch, labels):

    labels = np.array(labels)  # i.e. [0,4,5,6,7,8]
    patches = []
    for i, l in enumerate(labels):
        a = np.where(patch != l, 0, 1)
        patches.append(a)

    patches = np.array(patches)
    patches = np.rollaxis(patches, 0, len(patches.shape))  # from [6, 64, 128, 128] to [64, 128, 128, 6]?

    return np.float64(patches)

'''

pred_file_name= '/data/jjia/new/results/lobe/valid/pred/GLUCOLD/1584790285.7812743_1e-051a_o_0.5ds2dr1bn1fs16ptsz144ptzsz64/GLUCOLD_patients_26.mhd'
gdth_file_name = '/data/jjia/mt/data/lobe/valid/gdth_ct/GLUCOLD/GLUCOLD_patients_26.nrrd'


'''


def write_all_metrics(labels, gdth_path, pred_path, csv_file, fissure=False):
    """

    :param labels:  exclude background
    :param gdth_path:
    :param pred_path:
    :param csv_file:
    :return:
    """
    print('start calculate all metrics (volume and distance) and write them to csv')
    gdth_names, pred_names = get_gdth_pred_names(gdth_path, pred_path, fissure=fissure)

    for gdth_name, pred_name in zip(gdth_names, pred_names):

        gdth, gdth_origin, gdth_spacing = futil.load_itk(gdth_name)
        pred, pred_origin, pred_spacing = futil.load_itk(pred_name)

        gdth = one_hot_encode_3D(gdth, labels=labels)
        pred = one_hot_encode_3D(pred, labels=labels)
        print('start calculate all metrics for image: ', pred_name)
        metrics_dict_all_labels = get_metrics_dict_all_labels(labels, gdth, pred, spacing=gdth_spacing[::-1])
        metrics_dict_all_labels['filename'] = pred_name  # add a new key to the metrics
        data_frame = pd.DataFrame(metrics_dict_all_labels)
        data_frame.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

    return None

def main():

    # task='lobe'
    # model = '1584790285.7812743_1e-051a_o_0.5ds2dr1bn1fs16ptsz144ptzsz64'
    # for file_name in ['26', '27', '28', '29', '30']:
    #     pred_file_name= '/data/jjia/new/results/lobe/valid/pred/GLUCOLD/' + model + '/GLUCOLD_patients_' + file_name + '.mhd'
    #     gdth_file_name = '/data/jjia/mt/data/lobe/valid/gdth_ct/GLUCOLD/GLUCOLD_patients_' + file_name + '.nrrd'
    #     gdth, gdth_origin, gdth_spacing = futil.load_itk(gdth_file_name)
    #     pred, pred_origin, pred_spacing = futil.load_itk(pred_file_name)
    #
    task = 'vessel'
    model = '1591438344_307_lr0.0001ld0m6l0m7l0pm0.5no_label_dirSScao0ds0dr1bn1fn16trszNonetrzszNonetrspNonetrzspNoneptch_per_scan500tr_nb18ptsz144ptzsz96'
    for file_name in ['52','53']:
        pred_file_name = '/data/jjia/new/results/vessel/valid/pred/SSc/' + model + '/SSc_patient_' + file_name + '.mhd'
        gdth_file_name = '/data/jjia/mt/data/vessel/valid/gdth_ct/SSc/SSc_patient_' + file_name + '.mhd'
        gdth, gdth_origin, gdth_spacing = futil.load_itk(gdth_file_name)
        pred, pred_origin, pred_spacing = futil.load_itk(pred_file_name)

        # pred, pred_origin, pred_spacing = futil.load_itk('/data/jjia/new/results/SSc_51_lobe_segmentation/loberecon/SSc_patient_51.mhd')

        # connedted_pred = largest_connected_parts(pred, nb_parts_saved=5)
        # pred[connedted_pred == 0] = 0
        # futil.save_itk(
        #     'results/SSc_51_lobe_segmentation/SSc_patient_51_connected.nrrd',
        #     pred, pred_origin, pred_spacing)

        #
        # pred, pred_origin, pred_spacing = futil.load_itk(pred_file_name)
        #
        # connedted_pred = largest_connected_parts(pred, nb_parts_saved=5)
        # pred[connedted_pred==0] = 0
        # futil.save_itk('/data/jjia/new/results/lobe/valid/pred/GLUCOLD/' + model + '/GLUCOLD_patients_' + file_name + '_connected.nrrd', pred, pred_origin, pred_spacing)

        if task=='vessel':
            labels=[1]
        elif task=='lobe':
            labels=[4,5,6,7,8]
        gdth = one_hot_encode_3D(gdth, labels=labels)
        pred = one_hot_encode_3D(pred, labels=labels)

        metrics_dict_all_labels = metrics_dict_all_labels(labels, gdth, pred, spacing=gdth_spacing[::-1])
        metrics_dict_all_labels['filename']=pred_file_name  # add a new key to the metrics
        data_frame = pd.DataFrame(metrics_dict_all_labels)
        data_frame.to_csv('/data/jjia/new/results/vessel/valid/pred/SSc/' + model + '/SSc_patient_' + file_name + 'connected.csv', index=False)


if __name__=='__main__':
    main()