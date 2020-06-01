import numpy as np
from scipy.ndimage import morphology
import futils.util as futil
import time
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

def surfd(input1, input2, spacing=1, connectivity=3):
    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    input1_erosion = morphology.binary_erosion(input_1, conn).astype(int)
    input2_erosion = morphology.binary_erosion(input_2, conn).astype(int)
    S = input_1 - input1_erosion
    Sprime = input_2 - input2_erosion
    S = S.astype(np.bool)
    Sprime = Sprime.astype(np.bool)

    dta = morphology.distance_transform_edt(~S, spacing)
    dtb = morphology.distance_transform_edt(~Sprime, spacing)
    ds1 = np.ravel(dta[Sprime != 0])
    ds2 = np.ravel(dtb[S != 0])

    count1 = np.count_nonzero(ds1)
    count2 = np.count_nonzero(ds2)

    dta_ = morphology.distance_transform_edt(S, spacing)
    dtb_ = morphology.distance_transform_edt(Sprime, spacing)
    ds1_ = np.ravel(dta[Sprime == 0])
    ds2_ = np.ravel(dtb[S == 0])

    count1_ = np.count_nonzero(ds1_)
    count2_ = np.count_nonzero(ds2_)

    sds = np.concatenate([ds1, ds2])

    return sds

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


gdth_file_name = '/data/jjia/mt/data/vessel/valid/gdth_ct/SSc/SSc_patient_51.mhd'
pred_file_name = '/data/jjia/practice/SSc_patient_51.mhd'
gdth, _, gdth_spacing = futil.load_itk(gdth_file_name)
pred, _, pred_spacing = futil.load_itk(pred_file_name)
gdth = np.expand_dims(gdth, axis=-1)
pred = np.expand_dims(pred, axis=-1)

task = 'vessel'
if task=='lobe':
    labels=[4,5,6,7,8]
else:
    labels=[1]

gdth_encode = one_hot_encode_3D(gdth, labels=labels)
pred_encode = one_hot_encode_3D(pred, labels=labels)

msd_list, rms_list, hd_list, hd95_list, median_list, std_list = [], [], [], [], [], []
for i in range(len(labels)):
    print('start a loop of ', i)
    time1 = time.time()
    pred = pred_encode[..., i]
    gdth = gdth_encode[..., i]

    metrics = metrics_per_chn(gdth, pred)
    print('recall_per_chn, precision_per_chn, false_positive_rate_per_chn, jaccard_per_chn, dice_per_chn')
    print(metrics)

    surface_distance = surfd(pred, gdth, spacing=gdth_spacing)


    msd = surface_distance.mean()
    rms = np.sqrt((surface_distance ** 2).mean())
    hd = surface_distance.max()
    hd95 = np.percentile(surface_distance, 95)
    median = np.median(surface_distance)
    std = np.std(surface_distance)
    msd_list.append(msd)
    rms_list.append(rms)
    hd_list.append(hd)
    hd95_list.append(hd95)
    median_list.append(median)
    std_list.append(std)



    print('with spacing: msd:{}, rms:{}, hd:{}, hd95:{}, median:{}, std:{}'.format(msd, rms, hd, hd95, median, std))

    time2 = time.time()
    print('cost time', time2 - time1)
    print('-------')

print(msd_list, rms_list, hd_list, hd95_list, median_list, std_list)
print(np.array(msd_list).mean(), np.array(rms_list).mean(), np.array(hd_list).mean(), np.array(hd95_list).mean(), np.array(median_list).mean(), np.array(std_list).mean())
    # surface_distance = surfd(pred, gdth, 1)
    #
    #
    # msd = surface_distance.mean()
    # rms = np.sqrt((surface_distance ** 2).mean())
    # hd = surface_distance.max()
    # hd95 = np.percentile(surface_distance, 95)
    # median = np.median(surface_distance)
    # std = np.std(surface_distance)
    # print('without spacing: msd:{}, rms:{}, hd:{}, hd95:{}, median:{}, std:{}'.format(msd, rms, hd, hd95, median, std))
    # time3 = time.time()
    # print('cost time', time3 - time2)
    # print('-------------------------------------------------------')





print('ok')