import os
import queue
import threading
import time

import numpy as np

import futils.util as futil
from futils.util import downsample, get_all_ct_names


class Mask():
    def __init__(self, mask, file_name, pad_nb, preds_dir, origin, spacing,
                 trgt_space_list, original_shape, labels, trgt_sz_list, io, task):
        self.mask = mask
        self.file_name = file_name
        self.pad_nb = pad_nb
        self.preds_dir = preds_dir
        self.origin = origin
        self.spacing = spacing
        self.trgt_space_list = trgt_space_list
        self.original_shape = original_shape
        self.labels = labels
        self.trgt_sz_list = trgt_sz_list
        self.io = io
        self.task = task

    def upsample_crop_save_ct(self):
        """
        for io of "out_hgh", upsample is not necessary. So we need an if/else.
        :return:
        """
        masks = self.mask
        scan_file = self.file_name
        pad_nb = self.pad_nb

        if any(self.trgt_space_list) or any(self.trgt_sz_list):  # only for lobe yet.
            print('start upsampling the output mask to original size/spacing')
            final_pred = downsample(masks, is_mask=True,
                                    ori_space=self.trgt_space_list,
                                    trgt_space=self.spacing,
                                    ori_sz=masks.shape,
                                    trgt_sz=self.original_shape,
                                    order=1,
                                    labels=self.labels)
        else:
            final_pred = masks

        print('final_pred.shape: ', final_pred.shape)
        mask = final_pred[pad_nb:-pad_nb, pad_nb:-pad_nb, pad_nb:-pad_nb]

        if not os.path.exists(self.preds_dir):
            os.makedirs(self.preds_dir)

        futil.save_itk(self.preds_dir + '/' + scan_file.split('/')[-1], mask, self.origin, self.spacing)
        print('successfully save ct mask at', self.preds_dir + '/' + scan_file.split('/')[-1])


def write_preds_to_disk(segment, data_dir, preds_dir, number=None, stride=0.25, workers=1, qsize=1):
    """
    write predes to disk. Divided into 2 parts: predict (require segmentor.py, using GPU, can not use multi threads),
    and upsampling (require upsample_crop_save_ct, using cpu, multi threads).
    :param segment: an object or an instance
    :param data_dir: directory where ct data is
    :param preds_dir: directory where prediction result will be saved
    :param number: number of predicted ct
    :param stride: stride or overlap ratio during patching
    :return: None
    """
    scan_files = get_all_ct_names(data_dir, number=number)
    print("files are: ", scan_files)
    pad_nb = 48
    q = queue.Queue(qsize)
    cooking_flag = False

    def consumer():  # neural network inference needs GPU which can not be computed by multi threads, so the
        # consumer is just the upsampling only.
        while True:
            if len(
                    scan_files) or cooking_flag or not q.empty():  # if scan_files and q are empty, then threads should not wait any more
                with threading.Lock():
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
                print(
                    threading.current_thread().name + "scan_files are empty, cooking flag is False, q is empty, finish the thread")
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
        cooking_flag = True

        # ct_scan.shape: (717,, 512, 512), spacing: 0.5, 0.741, 0.741
        ct_scan, origin, spacing = futil.load_itk(filename=scan_file)
        ct_scan = np.pad(ct_scan, ((pad_nb, pad_nb), (pad_nb, pad_nb), (pad_nb, pad_nb)), mode='constant',
                         constant_values=-3000)
        print('Spacing: ', spacing, 'size', ct_scan.shape)

        # NORMALIZATION
        ct_scan = futil.normalize(ct_scan)

        mask, trgt_space_list, original_shape, labels, trgt_sz_list, io, task = segment.predict(
            ct_scan[..., np.newaxis], ori_space_list=spacing, stride=stride)  # shape: (717, 512, 512,1)
        mask = Mask(mask, scan_file, pad_nb, preds_dir, origin, spacing, trgt_space_list, original_shape,
                    labels, trgt_sz_list, io, task)
        q.put(mask, timeout=6000)
        cooking_flag = False

    for thd in thd_list:
        thd.join()


def main():
    import tensorflow as tf
    from tensorflow.keras import backend as K

    import segmentor as v_seg

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    K.set_session(sess)  # set this TensorFlow session as the default session for Keras

    task = 'lobe'
    str_name = "1600642845_85_lrlb0.0001lrvs1e-05mtscale0netnol-nnl-novpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96"

    # mypath = Mypath(task=task, current_time=str_name)  # set task=vessel to predict the lobe masks of SSc
    model_name = '/data/jjia/new/models/' + task + '/' + str_name + '_valid.hdf5'
    # model_name = mypath.model_fpath_best_whole("valid", str_name)
    tr_sp, tr_z_sp = 1.4, 2.5
    tr_sz, tr_z_sz = None, None
    pt_sz, pt_z_sz = 144, 96
    print('patch_sz', pt_sz, 'patch_z_size', pt_z_sz)
    stride = 0.25

    segment = v_seg.v_segmentor(batch_size=1,
                                model=model_name,
                                ptch_sz=pt_sz, ptch_z_sz=pt_z_sz,
                                trgt_sz=tr_sz, trgt_z_sz=tr_z_sz,
                                trgt_space_list=[tr_z_sp, tr_sp, tr_sp],
                                task=task, attention=False)

    print('stride is', stride)
    sub_dirs = ["COPE", "CONP", "NCPE"]
    for sub_dir in sub_dirs:
        write_preds_to_disk(segment=segment,
                            data_dir="/data/jjia/new/COVID/ori/"+sub_dir+"/001",
                            preds_dir="/data/jjia/new/COVID/results/"+sub_dir+"/001",
                            number=2,
                            stride=stride, workers=1, qsize=1)


if __name__ == '__main__':
    main()
