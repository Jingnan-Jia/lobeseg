
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
        connect_part_list.append(connect_part)
        pixel_count = np.sum(connect_part)
        pixel_count_list.append(pixel_count)
        pixel_label_list.append(i)
    pixel_count_list, connect_part_list, pixel_label_list = zip(*sorted(zip(pixel_count_list, connect_part_list, pixel_label_list), reverse=True))
    out = np.zeros(bw_img.shape)
    for i in range(nb_parts_saved):
        out += connect_part_list[i]

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