import glob
import time
import futils.util as futil
import os
import numpy as np
import queue
import threading
from futils.util import downsample


class Mask():
    def __init__(self, mask, file_name, pad_nb, preds_dir, origin, spacing, trgt_space_list, original_shape, labels, low_msk, trgt_sz_list):
        self.mask = mask
        self.file_name = file_name
        self.pad_nb = pad_nb
        self.preds_dir = preds_dir
        self.origin = origin
        self.spacing = spacing
        self.trgt_space_list = trgt_space_list
        self.original_shape = original_shape
        self.labels = labels
        self.low_msk = low_msk
        self.trgt_sz_list = trgt_sz_list

    def upsample_crop_save_ct(self):
        masks = self.mask
        scan_file = self.file_name
        pad_nb = self.pad_nb

        if any(self.trgt_space_list) or any(self.trgt_sz_list):
            if self.low_msk:
                print('rescaled to original spacing')
                final_pred = downsample(masks,
                                        ori_space=self.trgt_space_list,
                                        trgt_space=self.spacing,
                                        ori_sz=masks.shape,
                                        trgt_sz=self.original_shape,
                                        order=1,
                                        labels=self.labels)
            else:
                final_pred = masks
        else:
            final_pred = masks

        print('final_pred.shape: ', final_pred.shape)
        mask = final_pred[pad_nb:-pad_nb, pad_nb:-pad_nb, pad_nb:-pad_nb]

        if not os.path.exists(self.preds_dir):
            os.makedirs(self.preds_dir)
        if type(mask) is list:
            futil.save_itk(self.preds_dir + '/' + scan_file.split('/')[-1], mask[0], self.origin, self.spacing)
            print('save ct mask at', self.preds_dir + '/' + scan_file.split('/')[-1])
            futil.save_itk(self.preds_dir + '/upsampled_' + scan_file.split('/')[-1], mask[1], self.origin, self.spacing)
            print('save ct mask at', self.preds_dir + '/upsampled_' + scan_file.split('/')[-1])
        else:
            futil.save_itk(self.preds_dir + '/' + scan_file.split('/')[-1], mask, self.origin, self.spacing)
            print('save ct mask at', self.preds_dir + '/' + scan_file.split('/')[-1])


def get_scan_files(data_dir, number):
    print('Start write_preds_to_disk')
    scan_files = sorted(glob.glob(data_dir + '/' + '*.mhd'))
    scan_files.extend(sorted(glob.glob(data_dir + '/' + '*.nrrd')))
    if scan_files is None:
        raise Exception('Scan files are None, please check the data directory')
    if isinstance(number, int):
        scan_files = scan_files[:number]
    elif isinstance(number, list):  # number = [3,7]
        scan_files = scan_files[number[0]:number[1]]
    print('Predicted files are:', scan_files)
    return scan_files


def write_preds_to_disk(segment, data_dir, preds_dir, number=None, stride=0.25, workers=1, qsize=1):
    '''
    write predes to disk.
    :param segment: an object or an instance
    :param data_dir: directory where ct data is
    :param preds_dir: directory where prediction result will be saved
    :param number: number of predicted ct
    :param stride: stride or overlap ratio during patching
    :return: None
    '''
    scan_files = get_scan_files(data_dir, number)
    pad_nb = 48
    q = queue.Queue(qsize)
    running = True

    def consumer():  # neural network inference needs GPU which can not be computed by multi threads, so the
        # consumer is just the upsampling only.
        while True:
            if len(scan_files):  # if scan_files are empty, then threads should not wait any more
                out_mask = None
                with threading.Lock():
                    if not q.empty():
                        print(threading.current_thread().name + " gets the lock, thread id: " + str(threading.get_ident()) +
                              " prepare to upsample data, waiting for the data from queue")
                        try:
                            out_mask = q.get(timeout=60)  # wait up to 1 minutes
                            t2 = time.time()
                            print(threading.current_thread().name + " gets the data before upsample at time "
                                  + str(t2) + ", the thread releases the lock")
                        except:
                            out_mask = None
                            print(threading.current_thread().name + " does not get the data in 60s, check again if "
                                  + "the scan_files are still not empty, the thread releases the lock")
                if out_mask is not None:
                    t1 = time.time()
                    out_mask.upsample_crop_save_ct()
                    t3 = time.time()
                    print("it costs tis secons to upsample the data " + str(t3 - t1))
            else:
                print(threading.current_thread().name + "scan_files are empty, finish the thread")
                return None

    thd_list = []
    for i in range(workers):
        thd = threading.Thread(target=consumer)
        thd.start()
        thd_list.append(thd)


    # for scan_file in scan_files:
    for i in range(len(scan_files)):
        print('start iterate')
        scan_file = scan_files.pop()

        # ct_scan.shape: (717,, 512, 512), spacing: 0.5, 0.741, 0.741
        ct_scan, origin, spacing = futil.load_itk(filename=scan_file)
        ct_scan = np.pad(ct_scan, ((pad_nb, pad_nb), (pad_nb, pad_nb), (pad_nb, pad_nb)), mode='constant',
                         constant_values=-3000)
        print('Spacing: ', spacing, 'size', ct_scan.shape)

        # NORMALIZATION
        ct_scan = futil.normalize(ct_scan)

        mask, trgt_space_list, original_shape, labels, low_msk, trgt_sz_list = segment.predict(ct_scan[..., np.newaxis],
                               ori_space_list=spacing, stride=stride, pad_nb=pad_nb)  # shape: (717, 512, 512,1)
        mask = Mask(mask, scan_file, pad_nb,  preds_dir, origin, spacing, trgt_space_list, original_shape,
                    labels, low_msk, trgt_sz_list)
        q.put(mask, timeout=6000)

    for thd in thd_list:
        thd.join()


def main():
    pass


if __name__ == '__main__':
    main()
