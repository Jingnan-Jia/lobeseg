import numpy as np
import futils.util as futil
import SimpleITK as sitk
import copy

def metrics_per_chn(gdth, pred):
    gdth = gdth.astype(int)
    pred = pred.astype(int)
    fp_array = copy.deepcopy(pred)
    fn_array = copy.deepcopy(gdth)

    gdth_sum = np.sum(gdth)
    pred_sum = np.sum(pred)
    intersection = np.bitwise_and(gdth, pred)
    union = np.bitwise_or(gdth, pred)
    intersection_sum = np.sum(intersection)
    union_sum = np.sum(union)

    tp_array = intersection

    tmp = pred - gdth
    fp_array[tmp<1]=0

    tmp2 = gdth - pred
    fn_array[tmp2<1]=0

    tn_array = np.ones(gdth.shape) - union

    tp, fp, fn, tn = np.sum(tp_array), np.sum(fp_array), np.sum(fn_array), np.sum(tn_array)

    smooth = 0.001
    precision_per_chn = tp / (pred_sum + smooth)
    recall_per_chn = tp / (gdth_sum + smooth)
    false_positive_rate_per_chn = fp / (fp + tn + smooth)

    jaccard_per_chn = intersection_sum / (union_sum + smooth)
    dice_per_chn = 2 * intersection_sum / (gdth_sum + pred_sum + smooth)

    return recall_per_chn, precision_per_chn, false_positive_rate_per_chn, jaccard_per_chn, dice_per_chn


def metrics_all(gdth, pred):
    recalls, precisions, false_positive_rates, jaccards, dices = [], [], [], [], []
    for gdth_per_chn, pred_per_chn in zip(gdth, pred):
        recall_per_chn, precision_per_chn, false_positive_rate_per_chn, jaccard_per_chn, dice_per_chn = metrics_per_chn(gdth_per_chn, pred_per_chn)

        recalls.append(recall_per_chn)
        precisions.append(precision_per_chn)
        false_positive_rates.append(false_positive_rate_per_chn)
        jaccards.append(jaccard_per_chn)
        dices.append(dice_per_chn)

    return recalls, precisions, false_positive_rates, jaccards, dices

def metrics_ave(gdth, pred):

    recalls, precisions, false_positive_rates, jaccards, dices = metrics_all(gdth, pred)

    recall_ave, precision_ave, false_positive_rate_ave, jaccard_ave, dice_ave = np.average(recalls, precisions, false_positive_rates, jaccards, dices)

    return recall_ave, precision_ave, false_positive_rate_ave, jaccard_ave, dice_ave


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

def load_scan(file_name):

    """Load mhd or nrrd 3d scan"""

    extension = file_name.split('.')[-1]
    if extension == 'mhd':
        scan, origin, spacing = futil.load_itk(file_name)

    elif extension == 'nrrd':
        scan, origin, spacing = futil.load_nrrd(file_name)

    return np.expand_dims(scan, axis=-1), spacing


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
    metrics = metrics_per_chn(gdth, pred)
    print('recall_per_chn, precision_per_chn, false_positive_rate_per_chn, jaccard_per_chn, dice_per_chn')
    print(metrics)

    overlap_results = overlab_measures(gdth, pred)
    print(overlap_results)

    # metrics_itk = overlab_measures
    # print(metrics_itk)