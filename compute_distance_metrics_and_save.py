import copy
import os
import threading
import time

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import futils.util as futil
from futils.util import get_gdth_pred_names, downsample


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

    pred = lP.astype(int)  # float data does not support bit_and and bit_or
    gdth = lT.astype(int)  # float data does not support bit_and and bit_or
    fp_array = copy.deepcopy(pred)  # keep pred unchanged
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
    labelTrue.SetSpacing(spacing)  # spacing order (x, y, z)

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
    signed_distance_map = sitk.SignedMaurerDistanceMap(labelTrue > 0.5, squaredDistance=False,
                                                       useImageSpacing=True)  # It need to be adapted.
    # show_itk(signed_distance_map, slice_idx)

    ref_distance_map = sitk.Abs(signed_distance_map)
    # show_itk(ref_distance_map, slice_idx)

    ref_surface = sitk.LabelContour(labelTrue > 0.5, fullyConnected=True)
    # show_itk(ref_surface, slice_idx)
    ref_surface_array = sitk.GetArrayViewFromImage(ref_surface)

    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(ref_surface > 0.5)

    num_ref_surface_pixels = int(statistics_image_filter.GetSum())

    signed_distance_map_pred = sitk.SignedMaurerDistanceMap(labelPred > 0.5, squaredDistance=False,
                                                            useImageSpacing=True)
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
    ref2seg_distances = ref2seg_distances + list(np.zeros(num_ref_surface_pixels - len(ref2seg_distances)))  #

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


def write_all_metrics_for_one_ct(labels, gdth_name, pred_name, csv_file, lung, fissure):
    gdth, gdth_origin, gdth_spacing = futil.load_itk(gdth_name)
    pred, pred_origin, pred_spacing = futil.load_itk(pred_name)

    if lung:  # gdth is lung, so pred need to convert to lung from lobes, labels need to be [1],
        # size need to be the same the the gdth size (LOLA11 mask resolutin is 1 mm, 1mm, 1mm)
        pred = get_lung_from_lobe(pred)
        labels = [1]
        if not gdth.shape == pred.shape:  # sometimes the gdth size is different with preds.
            pred = downsample(pred, ori_sz=pred.shape, trgt_sz=gdth.shape, order=1,
                              labels=labels)  # use shape to upsampling because the space is errors sometimes in LOLA11
        suffex_len = len(os.path.basename(pred_name).split(".")[-1])
        lung_file_dir = os.path.dirname(pred_name) + "/lung"
        lung_file_fpath = lung_file_dir + "/" + os.path.basename(pred_name)[:-suffex_len-1] + '.mhd'
        if not os.path.exists(lung_file_dir):
            os.makedirs(lung_file_dir)

        futil.save_itk(lung_file_fpath,  pred, pred_origin, pred_spacing)

    elif fissure and ('LOLA11' in gdth_name or "lola11" in gdth_name):  # only have slices annotations
        pred_cp = copy.deepcopy(pred)
        slic_nb=0
        for i in range(gdth.shape[1]):  # gdth.shape=(600, 512, 512)
            gdth_slice = gdth[:, i, :]
            if not gdth_slice.any():  # the slice is all black
                pred_cp[:, i, :] = 0
            else:
                slic_nb+=1
                # print("gdth slice sum"+str(np.sum(gdth_slice)))
                for j in range(gdth.shape[2]):  # some times only one lobe is annotated in the same slice.
                    gdth_line = gdth_slice[:, j]
                    if not gdth_line.any():
                        pred_cp[:, i, j] = 0
        if slic_nb > 30:
            print('slice number of valuable lobe is greater than 30: '+str(slic_nb)+", change to another axis")
            pred_cp = copy.deepcopy(pred)
            slic_nb = 0
            for i in range(gdth.shape[2]):  # gdth.shape=(600, 512, 512)
                gdth_slice = gdth[:, :, i]
                if not gdth_slice.any():  # the slice is all black
                    pred_cp[:, :, i] = 0
                else:
                    slic_nb += 1
                    # print("gdth slice sum" + str(np.sum(gdth_slice)))
                    for j in range(gdth.shape[1]):  # some times only one lobe is annotated in the same slice.
                        gdth_line = gdth_slice[:, j]
                        if not gdth_line.any():
                            pred_cp[:, j, i] = 0
        if slic_nb > 30:
            raise Exception("cannot get fissure points")
        pred = pred_cp
        futil.save_itk(pred_name.split(".mh")[0]+"_points.mha", pred, pred_origin, pred_spacing)
        print('slice number of valuable lobe: ', slic_nb)

    gdth = one_hot_encode_3D(gdth, labels=labels)
    pred = one_hot_encode_3D(pred, labels=labels)
    print('start calculate all metrics for image: ', pred_name)
    metrics_dict_all_labels = get_metrics_dict_all_labels(labels, gdth, pred, spacing=pred_spacing[::-1])
    metrics_dict_all_labels['filename'] = pred_name  # add a new key to the metrics
    data_frame = pd.DataFrame(metrics_dict_all_labels)
    with threading.Lock():
        data_frame.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
        print(threading.current_thread().name + "successfully write metrics to csv " + csv_file)


def get_lung_from_lobe(pred):
    pred[pred >= 1] = 1
    return pred


def write_all_metrics(labels, gdth_path, pred_path, csv_file, fissure=False, fissureradius=1, lung=False, workers=1):
    """

    :param workers:
    :param lung:
    :param fissureradius:
    :param fissure:
    :param labels:  exclude background
    :param gdth_path:
    :param pred_path:
    :param csv_file:
    :return:
    """
    print('start calculate all metrics (volume and distance) and write them to csv')
    gdth_names, pred_names = get_gdth_pred_names(gdth_path, pred_path, fissure=fissure, fissureradius=fissureradius)

    # gdth_names = get_all_ct_names(gdth_path)
    # import csv
    # for gname in gdth_names:
    #     g = sitk.ReadImage(gname)
    #     orientationg = g.GetDirection()
    #     with open("lola11.csv", "a+") as f:
    #         writer = csv.writer(f)
    #         row = [ "gname", gname, "orientation_g", orientationg]
    #         writer.writerow(row)
            
    # for pname, gname in zip(pred_names, gdth_names):
    #     p = sitk.ReadImage(pname)
    #     g = sitk.ReadImage(gname)
    #     orientationp = p.GetDirection()
    #     orientationg = g.GetDirection()
    #     with open("lola11.csv", "a+") as f:
    #         writer = csv.writer(f)
    #         row = ["pname", pname, "gname", gname, "orientation_p", orientationp, "orientation_g", orientationg]
    #         writer.writerow(row)



    def consumer():  # neural network inference needs GPU which can not be computed by multi threads, so the
        # consumer is just the upsampling only.
        while True:
            with threading.Lock():
                pred_name = None
                if len(pred_names):  # if scan_files are empty, then threads should not wait any more
                    pred_name = pred_names.pop()  # wait up to 1 minutes
                    gdth_name = gdth_names.pop()
                    print(threading.current_thread().name + " gets the lock, thread id: " + str(threading.get_ident())
                          +"start handle "+pred_name+"and"+gdth_name)

            if pred_name is not None:
                t1 = time.time()
                write_all_metrics_for_one_ct(labels, gdth_name, pred_name, csv_file, lung, fissure)
                t3 = time.time()
                print("it costs tis seconds to compute the the data " + str(t3 - t1))
            else:
                print(threading.current_thread().name + "scan_files are empty, finish the thread")
                return None

    thd_list = []
    for i in range(workers):
        thd = threading.Thread(target=consumer)
        thd.start()
        thd_list.append(thd)

    for thd in thd_list:
        thd.join()


def main():
    task = 'vessel'
    model = '1591438344_307_lr0.0001ld0m6l0m7l0pm0.5no_label_dirSScao0ds0dr1bn1fn16trszNonetrzszNonetrspNonetrzspNoneptch_per_scan500tr_nb18ptsz144ptzsz96'
    for file_name in ['52', '53']:
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

        if task == 'vessel':
            labels = [1]
        elif task == 'lobe':
            labels = [4, 5, 6, 7, 8]
        gdth = one_hot_encode_3D(gdth, labels=labels)
        pred = one_hot_encode_3D(pred, labels=labels)

        metrics_dict_all_labels = metrics_dict_all_labels(labels, gdth, pred, spacing=gdth_spacing[::-1])
        metrics_dict_all_labels['filename'] = pred_file_name  # add a new key to the metrics
        data_frame = pd.DataFrame(metrics_dict_all_labels)
        data_frame.to_csv(
            '/data/jjia/new/results/vessel/valid/pred/SSc/' + model + '/SSc_patient_' + file_name + 'connected.csv',
            index=False)


if __name__ == '__main__':
    main()
