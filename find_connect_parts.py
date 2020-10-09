import copy
import os
import threading
import time

import numpy as np
from scipy import spatial
from skimage.measure import label
import futils.util as futil
import scipy
from futils.util import get_all_ct_names

1


def nerest_dis_to_center(img):
    sum_img = np.sum(img)
    position = np.where(img > 0)
    coordinates = np.transpose(np.array(position))  # get the coordinates where the voxels is not 0
    cposition = np.array(img.shape) / 2  # center point position/coordinate
    distance, index = spatial.KDTree(coordinates).query(cposition)
    return distance


def not_alone(idx, connect_part_list):
    candidate = copy.deepcopy(connect_part_list[idx])
    struct = scipy.ndimage.generate_binary_structure(3, 1)
    diated = scipy.ndimage.morphology.binary_dilation(candidate, structure=struct).astype(candidate.dtype)
    # t = 0
    for new_idx, part in enumerate(connect_part_list):
        if new_idx != idx:
            product = part * diated
            flag = np.sum(product)
            if flag:
                print("flag: ", flag)
                return 1

    return 0


def find_repeated_label(nb_saved, out, bw_img):
    img = copy.deepcopy(bw_img)
    img[out == 0] = 0
    uniq = np.unique(img)

    if len(np.unique(img)) < nb_saved + 1:  # shhuld be 1 background and 5 lobes
        return 1
    elif len(np.unique(img)) == nb_saved + 1:
        return 0
    else:
        raise Exception("labels are wrong!")


def find_wrong_part(img_1, img_2):
    d1 = nerest_dis_to_center(img_1)
    d2 = nerest_dis_to_center(img_2)
    if d1 > d2:
        return 1
    else:
        return 2


def delete_repeated_part(connect_list, idx_doubt_list, bw_img):


    connect_part_list_ = copy.deepcopy(connect_list)
    img_1 = copy.deepcopy(bw_img)
    img_1[connect_part_list_[idx_doubt_list[-1]] == 0] = 0

    find_flag = False
    for doubt_idx in idx_doubt_list[:-1]:  # exclude last one
        img_2 = copy.deepcopy(bw_img)  # Very important!!!
        out_doubt = np.zeros(bw_img.shape)
        img_2[connect_part_list_[doubt_idx] == 0] = 0
        img_3 = img_1 + img_2
        if len(np.unique(img_3)) == 2:  # ackground and the same foreground
            repeated_idx = find_wrong_part(img_1, img_2)
            if repeated_idx == 1:  # img_1 is wrong part
                idx_doubt_list.pop()
            else:  # img_2 is wrong part
                idx_doubt_list.remove(doubt_idx)
            for nb, idx_believe in enumerate(idx_doubt_list):
                out_doubt = connect_part_list_[idx_believe] * (nb + 1) + out_doubt
            find_flag = True
            return out_doubt, idx_doubt_list
    if not find_flag:
        raise Exception("someting wrong happened")


def largest_connected_parts(bw_img, nb_need_saved=1):
    labeled_img, num = label(bw_img, connectivity=len(bw_img.shape), background=0, return_num=True)
    all_pixel = np.sum((labeled_img >0).astype(int))

    connect_part_list = []
    pixel_count_list = []
    pixel_label_list = []

    i: int
    for i in range(1, num + 1):  # 0 respresent background
        connect_part = (labeled_img == i).astype(int)
        pixel_count = np.sum(connect_part)


        if pixel_count > 30000:

            print('-------*****-------')
            print('step', i, ' pixel number', pixel_count)
            pixel_count_list.append(pixel_count)
            connect_part_list.append(connect_part)
            pixel_label_list.append(i)



    pixel_count_list_index = list(range(len(pixel_count_list)))

    pixel_count_list_sorted, pixel_count_list_index_sorted = zip(
        *sorted(zip(pixel_count_list, pixel_count_list_index), reverse=True))
    # select largest 10
    count_sorted, count_idx_sorted = pixel_count_list_sorted[:10], pixel_count_list_index_sorted[:10]

    out = np.zeros(bw_img.shape)
    nb_saved: int = 1
    idx_doubt_list = []
    sum_part_list = []
    for idx in count_idx_sorted:
        if nb_saved <= nb_need_saved:
            print("nb_saved: " + str(nb_saved))
            sum_part = np.sum(connect_part_list[idx])
            sum_part_list.append(sum_part)
            out = connect_part_list[idx] * nb_saved + out  # to differentiate different parts.

            idx_doubt_list.append(idx)

            if find_repeated_label(nb_saved, out, bw_img):
                print('found repeated part, prepare delete the wrong parts by distance')
                out, idx_doubt_list = delete_repeated_part(connect_part_list, idx_doubt_list, bw_img)
                print("deleted the wrong parts, continue ")
            else:
                nb_saved += 1

    bw_img[out == 0] = 0
    print("all parts are found, prepare write result")
    return bw_img


def main():
    pred_file_dir = "/data/jjia/new/results/lobe/valid/pred/LOLA11/1600908421_801_lrlb0.0001lrvs1e-05mtscale0netnol-nnl-novpm0.5nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96"

    pred_file_dirs = ["/data/jjia/new/results/lobe/valid/pred/LOLA11/1600645190_537_lrlb0.0001lrvs1e-05mtscale0netnol-nnl-novpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                      "/data/jjia/new/results/lobe/valid/pred/LOLA11/1600645190_366_lrlb0.0001lrvs1e-05mtscale0netnolpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                      "/data/jjia/new/results/lobe/valid/pred/LOLA11/1600908421_13_lrlb0.0001lrvs1e-05mtscale0netnolpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                      "/data/jjia/new/results/lobe/valid/pred/LOLA11/1600908421_13_lrlb0.0001lrvs1e-05mtscale0netnolpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                      "/data/jjia/new/results/lobe/valid/pred/LOLA11/1600908421_13_lrlb0.0001lrvs1e-05mtscale0netnolpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                      "/data/jjia/new/results/lobe/valid/pred/LOLA11/1600479252_70_lrlb0.0001lrvs1e-05mtscale1netnolpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                      "/data/jjia/new/results/lobe/valid/pred/LOLA11/1600913687_652_lrlb0.0001lrvs1e-05mtscale1netnol-nnl-novpm0.5nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                      ]
    for pred_file_dir in pred_file_dirs:
        scan_files = get_all_ct_names(pred_file_dir)
        # scan_files = scan_files[36:39]

        def write_connected_lobe():  # neural network inference needs GPU which can not be computed by multi threads, so the
            # consumer is just the upsampling only.
            while True:
                with threading.Lock():
                    ct_fpath = None
                    if len(scan_files):  # if scan_files are empty, then threads should not wait any more
                        print(threading.current_thread().name + " gets the lock, thread id: " + str(
                            threading.get_ident()) + " prepare to compute largest 5 lobes , waiting for the data from queue")
                        ct_fpath = scan_files.pop()  # wait up to 1 minutes
                        print(threading.current_thread().name + " gets the data, thread id: " + str(
                            threading.get_ident()) + " prepare to release the lock.")

                if ct_fpath is not None:
                    t1 = time.time()
                    print(threading.current_thread().name + "is computing ...")
                    pred, pred_origin, pred_spacing = futil.load_itk(ct_fpath)
                    pred = largest_connected_parts(pred, nb_need_saved=5)
                    suffex_len = len(os.path.basename(ct_fpath).split(".")[-1])
                    new_dir = os.path.dirname(ct_fpath) + "/biggest_5_lobe"
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                        print('successfully create directory:', new_dir)
                    write_fpath = new_dir + "/" + os.path.basename(ct_fpath)[:-suffex_len - 1] + '.mhd'
                    futil.save_itk(write_fpath, pred, pred_origin, pred_spacing)
                    t3 = time.time()
                    print("successfully save largest 5 lobes at " + write_fpath)
                    print("it costs tis seconds to compute the largest 5 lobes of the data " + str(t3 - t1))
                else:
                    print(threading.current_thread().name + "scan_files are empty, finish the thread")
                    return None

        futil.execute_the_function_multi_thread(consumer=write_connected_lobe, workers=10)
    print('finish')


if __name__ == '__main__':
    main()
