
import numpy as np
from skimage.measure import label
import futils.util as futil
def largest_connected_parts(bw_img, nb_parts_saved=1):
    labeled_img, num = label(bw_img, connectivity=len(bw_img.shape), background=0, return_num=True)
    max_label = 0
    max_num = 0
    connect_part_list = []
    pixel_count_list = []
    pixel_label_list = []

    for i in range(1, num + 1):  # 0 respresent background
        connect_part = (labeled_img == i).astype(int)


        pixel_count = np.sum(connect_part)
        print('step', i, ' pixel number', pixel_count)
        if pixel_count>30000:
            print('-------*****-------')
            pixel_count_list.append(pixel_count)

            connect_part_list.append(connect_part)

            pixel_label_list.append(i)



    pixel_count_list_index = list(range(len(pixel_count_list)))
    pixel_count_list_sorted, pixel_label_list_sorted = zip(*sorted(zip(pixel_count_list, pixel_label_list), reverse=True))

    pixel_count_list_sorted, pixel_count_list_index_sorted = zip(
        *sorted(zip(pixel_count_list, pixel_count_list_index), reverse=True))
    # pixel_count_list_sorted, connect_part_list_sorted = zip(*sorted(zip(pixel_count_list, connect_part_list), reverse=True))
    out = np.zeros(bw_img.shape)
    for i, idx in zip(range(nb_parts_saved), pixel_count_list_index_sorted):
        print('i, idx', i, idx)
        out += connect_part_list[idx]

    return out

def main():
    gdth_file_name = '/data/jjia/mt/data/vessel/valid/gdth_ct/SSc/SSc_patient_51.mhd'
    pred_file_name = '/data/jjia/practice/SSc_patient_51.mhd'
    gdth, _, gdth_spacing = futil.load_itk(gdth_file_name)
    pred, _, pred_spacing = futil.load_itk(pred_file_name)

    b = largest_connected_parts(a, nb_parts_saved=3)
    print('-')
    print(b.astype(int))



if __name__=='__main__':
    main()