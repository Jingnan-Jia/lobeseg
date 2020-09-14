import glob
import time
import futils.util as futil
import os
import numpy as np


def write_preds_to_disk(segment, data_dir, preds_dir, number=None, stride=0.25):
    '''
    write predes to disk.
    :param segment: an object or an instance
    :param data_dir: directory where ct data is
    :param preds_dir: directory where prediction result will be saved
    :param number: number of predicted ct
    :param stride: stride or overlap ratio during patching
    :return: None
    '''

    print('Start write_preds_to_disk')
    scan_files = sorted(glob.glob(data_dir + '/' + '*.mhd'))
    if scan_files is None:
        scan_files = sorted(glob.glob(data_dir + '/' + '*.nrrd'))
    if scan_files is None:
        raise Exception('Scan files are None, please check the data directory')
    if isinstance(number, int):
        scan_files = scan_files[:number]
    elif isinstance(number, list):  # number = [3,7]
        scan_files = scan_files[number[0]:number[1]]
    print('Predicted files are:', scan_files)

    t1 = time.time()
    for scan_file in scan_files:

        # ct_scan.shape: (717,, 512, 512), spacing: 0.5, 0.741, 0.741
        ct_scan, origin, spacing = futil.load_itk(filename=scan_file)
        pad_nb = 48
        ct_scan = np.pad(ct_scan, ((pad_nb, pad_nb), (pad_nb, pad_nb), (pad_nb, pad_nb)), mode='constant',
                         constant_values=-3000)
        print('Spacing: ', spacing, 'size', ct_scan.shape)

        # NORMALIZATION
        ct_scan = futil.normalize(ct_scan)
        mask = segment.predict(ct_scan[..., np.newaxis], ori_space_list=spacing,
                               stride=stride, pad_nb=pad_nb)  # shape: (717, 512, 512,1)

        # Save the segmentation
        if not os.path.exists(preds_dir):
            os.makedirs(preds_dir)
        if type(mask) is list:
            futil.save_itk(preds_dir + '/' + scan_file.split('/')[-1], mask[0], origin, spacing)
            print('save ct mask at', preds_dir + '/' + scan_file.split('/')[-1])
            futil.save_itk(preds_dir + '/upsampled_' + scan_file.split('/')[-1], mask[1], origin, spacing)
            print('save ct mask at', preds_dir + '/upsampled_' + scan_file.split('/')[-1])
        else:
            futil.save_itk(preds_dir + '/' + scan_file.split('/')[-1], mask, origin, spacing)
            print('save ct mask at', preds_dir + '/' + scan_file.split('/')[-1])

    t2 = time.time()
    print('finishe a batch of predition, time cost: ', t2 - t1)


def main():
    pass


if __name__ == '__main__':
    main()
