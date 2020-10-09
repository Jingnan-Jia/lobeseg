# -*- coding: utf-8 -*-
"""
3D Extension of the work in https://github.com/costapt/vess2ret/blob/master/util/data.py.
===========================================================================================
Created on Tue Apr  4 09:35:14 2017
@author: Jingnan
"""

import os
import random
import numpy as np
from image import Iterator
from image import apply_transform
from image import transform_matrix_offset_center
from futils.vpatch import random_patch
import futils.util as futil
from scipy import ndimage
import time
import glob
import sys
from futils.util import downsample, one_hot_encode_3d
from functools import wraps
import queue
import threading
import copy


def random_transform(a, b, c=None, is_batch=True,
                     rotation_range=0.05,
                     height_shift_range=0.05,
                     width_shift_range=0.05,
                     shear_range=0.05,
                     fill_mode='constant',
                     zoom_range=0.05):
    """
    Random dataset augmentation.

    Adapted from https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
    """
    channel_index = 3
    row_index = 1
    col_index = 2

    if np.isscalar(zoom_range):
        zoom_range = [1 - zoom_range, 1 + zoom_range]
    elif len(zoom_range) == 2:
        zoom_range = [zoom_range[0], zoom_range[1]]

    if is_batch is False:
        # a and b are single images, so they don't have image number at index 0
        img_row_index = row_index - 1
        img_col_index = col_index - 1
        img_channel_index = channel_index - 1
    else:
        img_row_index = row_index
        img_col_index = col_index
        img_channel_index = channel_index
    # use composition of homographies to generate final transform that needs to be applied
    if rotation_range:
        theta = np.pi / 180 * np.random.uniform(-rotation_range,
                                                rotation_range)
    else:
        theta = 0
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    if height_shift_range:
        tx = np.random.uniform(-height_shift_range, height_shift_range) \
             * a.shape[img_row_index]
    else:
        tx = 0

    if width_shift_range:
        ty = np.random.uniform(-width_shift_range, width_shift_range) \
             * a.shape[img_col_index]
    else:
        ty = 0

    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    if shear_range:
        shear = np.random.uniform(-shear_range, shear_range)
    else:
        shear = 0
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix),
                              zoom_matrix)

    h, w = a.shape[img_row_index], a.shape[img_col_index]
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)

    if (c is not None):
        A = []
        B = []
        C = []

        for a_, b_, c_ in zip(a, b, c):
            a_ = apply_transform(a_, transform_matrix, img_channel_index - 1,
                                 fill_mode=fill_mode, cval=np.min(a_))
            b_ = apply_transform(b_, transform_matrix, img_channel_index - 1,
                                 fill_mode=fill_mode, cval=0)
            c_ = apply_transform(c_, transform_matrix, img_channel_index - 1,
                                 fill_mode=fill_mode, cval=0)

            A.append(a_)
            B.append(b_)
            C.append(c_)

        a = np.array(A)
        b = np.array(B)
        c = np.array(C)
        return a, b, c

    else:
        A = []
        B = []

        for a_, b_ in zip(a, b):
            a_ = apply_transform(a_, transform_matrix, img_channel_index - 1,
                                 fill_mode=fill_mode, cval=np.min(a_))
            b_ = apply_transform(b_, transform_matrix, img_channel_index - 1,
                                 fill_mode=fill_mode, cval=0)

            A.append(a_)
            B.append(b_)

        a = np.array(A)
        b = np.array(B).astype(np.uint8)

        return a, b


def rp_dcrt(fun):  # decorator to repeat a function until succeessful
    """
    A decorator to repeat this function until no errors raise.
    here, the function normally is a dataloader: iterator.next()
    sometimes, ct scans are empty or pretty smaller than patch size which can lead to errors.

    :param fun: a function which might meet errors
    :return: decorated function
    """

    @wraps(fun)
    def decorated(*args, **kwargs):
        next_fail = True
        while (next_fail):
            try:
                out = fun(*args, **kwargs)
                next_fail = False
            except:
                print('data load or patch failed, pass this data, load next data')
        return out

    return decorated


class QueueWithIndex(queue.Queue):
    def __init__(self, qsize, index_list, name):
        self.index_list = index_list
        self.name = name
        super(QueueWithIndex, self).__init__(qsize)


class ScanIterator(Iterator):
    """Class to iterate A and B 3D scans (mhd or nrrd) at the same time."""

    def __init__(self,
                 directory,
                 task=None,
                 a_dir_name='ori_ct', b_dir_name='gdth_ct',
                 sub_dir=None,
                 a_extension='.mhd', c_extension='.nrrd',
                 ptch_sz=None, ptch_z_sz=None,
                 tszzyx=None,
                 tspzyx=None,
                 data_argum=True,
                 patches_per_scan=5,
                 ds=2,
                 labels=None,
                 batch_size=1,
                 shuffle=True,
                 seed=None,
                 n=None,
                 no_label_dir=None,
                 p_middle=None,
                 aux=None,
                 ptch_seed=None,
                 io=False):
        """
        Iterate through two directories at the same time.

        Files under the directory A and B with the same name will be returned
        at the same time.
        Parameters:
        - directory: base directory of the dataset. Should contain two
        directories with name a_dir_name and b_dir_name;
        - a_dir_name: name of directory under directory that contains the A
        images;
        - b_dir_name: name of directory under directory that contains the B
        images;
        - c_dir_name : this is the auxiliar output folder
        - a/b/c_extension : type of the scan: nrrd or mhd (no dicom available)

        - N: if -1 uses the entire dataset. Otherwise only uses a subset;
        - batch_size: the size of the batches to create;
        - shuffle: if True the order of the images in X will be shuffled;
        - ds: number of deep classifiers
        - labels: output classes

        """
        self.io = io
        self.task = task
        if task == 'lobe':
            self.b_extension = '.nrrd'
        else:
            self.b_extension = '.mhd'

        if "2_in" in self.io or self.task == 'vessel':  # I hope to used p_middle for mtscale
            self.p_middle = p_middle
        else:
            self.p_middle = None

        self.sub_dir = sub_dir
        self.ds = ds
        self.labels = labels
        self.patches_per_scan = patches_per_scan
        self.tszzyx = tszzyx
        self.tspzyx = tspzyx
        self.ptch_sz = ptch_sz
        self.ptch_z_sz = ptch_z_sz
        self.ptch_seed = ptch_seed
        self.aux = aux
        self.running = True
        if self.aux:
            self.c_dir_name = 'aux_gdth'
            self.c_dir = os.path.join(directory, self.c_dir_name, sub_dir)
            self.c_extension = c_extension
        else:
            self.c_dir = None

        if self.task == 'no_label':
            self.a_dir = os.path.join(directory, a_dir_name, no_label_dir)
            self.a_extension = a_extension
            files = set(x.split(a_extension)[0].split(self.a_dir + '/')[-1] for x in
                        sorted(glob.glob(self.a_dir + '/*' + self.a_extension)))
            self.filenames = sorted(list(files))
        else:
            self.a_dir = os.path.join(directory, a_dir_name, sub_dir)
            self.b_dir = os.path.join(directory, b_dir_name, sub_dir)

            self.a_extension = a_extension

            a_files = set(x.split(a_extension)[0].split(self.a_dir + '/')[-1] for x in
                          sorted(glob.glob(self.a_dir + '/*' + self.a_extension)))
            b_files = set(x.split(self.b_extension)[0].split(self.b_dir + '/')[-1] for x in
                          sorted(glob.glob(self.b_dir + '/*' + self.b_extension)))

            # Files inside a and b should have the same name. Images without a pair
            # are discarded.
            self.filenames = sorted(list(a_files.intersection(b_files)))
        if 'train' in self.a_dir:
            self.state = 'train'
        else:
            self.state = 'monitor'
        if n:
            self.filenames = self.filenames[:n]

        print('task:', self.task)
        print('from this directory:', self.a_dir)
        print('these files are used ', self.filenames)
        if self.filenames is []:
            raise Exception('empty dataset')

        self.epoch_nb = 0

        self.data_argum = data_argum
        # self.shuffle = shuffle
        super(ScanIterator, self).__init__(len(self.filenames), batch_size, shuffle, seed)

    def _normal_normalize(self, scan):
        """returns normalized (0 mean 1 variance) scan"""
        scan = (scan - np.mean(scan)) / (np.std(scan))
        return scan

    def load_scan(self, file_name):
        """Load mhd or nrrd 3d scan thread safelly. Output is scan and spacing with shape (z,y,x)"""
        with self.lock:
            print(threading.current_thread().name + " get the lock, thread id: " + str(threading.get_ident()) +
                  " prepare to load data")
            scan, origin, spacing = futil.load_itk(file_name)  # all the output shape are (z, y, z)
            print(threading.current_thread().name + " load data successfully, release the lock")
        return np.expand_dims(scan, axis=-1), spacing  # size=(z,x,y,1)

    def _load_img_pair(self, idx):
        """Get a pair of images after padding and normalization if possible with spacing."""
        a_fname = self.filenames[idx] + self.a_extension
        print('start load file: ', a_fname)

        a, spacing = self.load_scan(file_name=os.path.join(self.a_dir, a_fname))  # (200, 512, 512, 1)
        pad_nb = 48
        a = np.pad(a, ((pad_nb, pad_nb), (pad_nb, pad_nb), (pad_nb, pad_nb), (0, 0)), mode='constant',
                   constant_values=-3000)

        # a = np.array(a)
        a = futil.normalize(a)  # threshold to [-1000,400], then rescale to [0,1]
        a = self._normal_normalize(a)

        if self.task == 'no_label':
            return a, a, spacing
        else:
            b_fname = self.filenames[idx] + self.b_extension
            b, _ = self.load_scan(file_name=os.path.join(self.b_dir, b_fname))  # (200, 512, 512, 1)
            b = np.pad(b, ((pad_nb, pad_nb), (pad_nb, pad_nb), (pad_nb, pad_nb), (0, 0)), mode='constant',
                       constant_values=0)

            if not self.aux:
                return a, b, spacing
            else:
                c_fname = self.filenames[idx] + self.c_extension
                c, _ = self.load_scan(file_name=os.path.join(self.c_dir, c_fname))  # (200, 512, 512, 1)
                c = np.pad(c, ((pad_nb, pad_nb), (pad_nb, pad_nb), (pad_nb, pad_nb), (0, 0)), mode='constant',
                           constant_values=0)
                return a, b, c, spacing

    @rp_dcrt
    def next(self):
        return None

    def get_ct_for_patching(self, idx):
        """
        a_hgh and b_hgh must exist. but:
        if trgt_sz and trgt_sp are None and not self.mtscale, a_low, b_low, c_low would be None
        if not self.aux, c_low, c_hgh would be None
        :param idx: index of ct scan
        :return: a_low, a_hgh, b_low, b_hgh, c_low, c_hgh
        """

        if self.aux:
            # with padding, shape of a,b,c_ori is (z,y,x,1), spacing (z,y,x)
            a_ori, b_ori, c_ori, spacing = self._load_img_pair(idx)
        else:
            # with padding, shape of a,b_ori is (z,y,x,1), spacing is (z,y,x)
            a_ori, b_ori, spacing = self._load_img_pair(idx)
        if self.task != 'no_label':  # encode first, downsample next. Otherwise we need to encode down and ori ct.
            b_ori = one_hot_encode_3d(b_ori, self.labels)  # shape: (z,y,x,chn)
            c_ori = one_hot_encode_3d(c_ori, [0, 1]) if self.aux else None  # shape: (z,y,x,2)

        if self.data_argum:
            if self.aux:
                a_ori, b_ori, c_ori = random_transform(a_ori, b_ori, c_ori)  # shape: (z,y,x,chn)
            else:
                a_ori, b_ori = random_transform(a_ori, b_ori)  # shape: (z,y,x,chn)

        if self.io!="1_in_hgh_1_out_hgh" and not (any(self.tspzyx) or any(self.tszzyx)):
            raise Exception("io is: "+str(self.io)+" but did not set trgt_space_list or trgt_sz_list")

        a_low, a_hgh, b_low, b_hgh, c_low, c_hgh = None, None, None, None, None, None
        if "in_low" in self.io or "2_in" in self.io:
            a_low = downsample(a_ori, is_mask=False,
                               ori_space=spacing, trgt_space=self.tspzyx,
                               ori_sz=a_ori.shape, trgt_sz=self.tszzyx,
                               order=1, labels=self.labels)  # shape: (z,y,x,chn)
        if "out_low" in self.io or "2_out" in self.io:
            b_low = downsample(b_ori,is_mask=True,
                               ori_space=spacing, trgt_space=self.tspzyx,
                               ori_sz=b_ori.shape, trgt_sz=self.tszzyx,
                               order=0, labels=self.labels)  # shape: (z,y,x,chn)
            if self.aux:
                c_low = downsample(c_ori,is_mask=True,
                                   ori_space=spacing, trgt_space=self.tspzyx,
                                   ori_sz=c_ori.shape, trgt_sz=self.tszzyx,
                                   order=0, labels=self.labels)  # shape: (z,y,x,2)

        if "in_hgh" in self.io or "2_in" in self.io:
            a_hgh = a_ori  # shape: (z,y,x,chn)
        if "out_hgh" in self.io or "2_out" in self.io:
            b_hgh = b_ori  # shape: (z,y,x,chn)
            if self.aux:
                c_hgh = c_ori

        return a_low, a_hgh, b_low, b_hgh, c_low, c_hgh  # shape (z,y,x,chn)

    def queue_data(self, q1, q2, i):
        with self.lock:
            print(threading.current_thread().name + " get the lock, thread id: " + str(threading.get_ident()) +
                  " prepare to get the index of data and load data latter")
            if self.state == "monitor" and not len(q1.index_list):
                print("task: " + self.task + " state: " + self.state + "worker_" + str(i) + " prepare put data but ")
                print( "index_list 1 is empty! the monitor thread has finished its job! stop this thread.")
                return True
            else:
                if not len(q1.index_list) and not len(q2.index_list):
                    pass  # all data have been sent, nothing to do now
                else:
                    if len(q1.index_list):
                        index = q1.index_list.pop()
                        current_q = q1
                    elif len(q2.index_list):
                        index = q2.index_list.pop()
                        current_q = q2
                    print("task: " + self.task + " state: " + self.state + "worker_" + str(i) +
                          " start to put data of " + str(index) + " into queue" + current_q.name + " for patching")
                    print("size of the queue: " + str(self.qmaxsize) + "occupied size: " + str(current_q.qsize()) +
                          "remaining index: " + str(current_q.index_list))
            print(threading.current_thread().name + "release the lock")

        ct_for_patching = self.get_ct_for_patching(index)

        with self.lock:
            print(threading.current_thread().name + " get the lock, thread id: " + str(threading.get_ident()) +
                  " prepare to put data to queue")
            t1 = time.time()
            current_q.put(ct_for_patching, timeout=600000)  # 160 hours  greater than the cost time of one epoch
            t2 = time.time()
            t = t2 - t1
            print("It cost this seconds to put the data into queue" + str(t))
            print("task: " + self.task + " state: " + self.state + "worker_" + str(
                i) + " successfully put data of " + str(index) +
                  " into queue" + current_q.name + " for patching")
            print("these index of data are waiting for loading: " + str(current_q.index_list))
            print("size of the queue: ", self.qmaxsize, "occupied size: ", current_q.qsize())
            print(threading.current_thread().name + "release the lock")

    def productor(self, i, q1, q2):
        # product ct scans for patching
        while True:
            if self.running:
                if self.epoch_nb % 2 == 0:  # even epoch number, q1 first
                    exit_flag = self.queue_data(q1, q2, i)
                else:
                    exit_flag = self.queue_data(q2, q1, i)
                if exit_flag:
                    return exit_flag
            else:
                print('running flag is set to false, this thread is stoped')
                return self.running

    def stop(self):
        self.running = False

    def join(self):
        for thd in self.thread_list:
            thd.join()

    def tune_axis(self, img):
        if type(img) is list:  # io is "2_out"
            img1, img2 = img[0], img[1]
            img1, img2 = np.rollaxis(img1, 0, 3), np.rollaxis(img2, 0, 3)
            img1, img2 = img1[np.newaxis, ...], img2 [np.newaxis, ...]
            return [img1, img2]
        else:
            img = np.rollaxis(img, 0, 3)  # (96, 144, 144, 1) -> (144, 144, 96, 1)
            img = img[np.newaxis, ...]  # (144, 144, 96, 1) -> (1, 144, 144, 96, 1)
            return img



    def generator(self, workers=5, qsize=5):
        index_sorted = list(range(self.n))
        if self.shuffle:
            index_list1 = random.sample(index_sorted, len(index_sorted))
            index_list2 = random.sample(index_sorted, len(index_sorted))
        else:
            index_list1 = list(range(self.n))
            index_list2 = list(range(self.n))
        self.qmaxsize = qsize
        q1 = QueueWithIndex(qsize, index_list1, name="q1")
        q2 = QueueWithIndex(qsize, index_list2, name="q2")
        self.thread_list = []
        for i in range(workers):
            thd = threading.Thread(target=self.productor, args=(i, q1, q2,))
            thd.name = self.task + str(i)
            thd.start()
            self.thread_list.append(thd)

            # self.thread_list.append(t)
        while True:
            # try:
            for _step in range(self.n):
                # try:
                q = q1 if self.epoch_nb % 2 == 0 else q2
                print("before q.get, qsize:", q.qsize(), q.index_list)
                time1 = time.time()

                print(threading.current_thread().name + " prepare to get data from queue")
                a_low, a_hgh, b_low, b_hgh, c_low, c_hgh  = q.get(timeout=6000)  # wait for several minitues for loading data
                # a/a2.shape (z,y,x,1) b/b2.shape (z,y,x,chn), c2.shape(z,y,x,2)
                time2 = time.time()
                print("it costs me this seconds to get the data: " + str(time2 - time1))
                print("after q.get, qsize:" + str(q.qsize()) + str(q.index_list))

                for _ in range(self.patches_per_scan):
                    if self.ptch_seed:
                        self.ptch_seed += 1
                    else:
                        self.ptch_seed = None

                    if self.ptch_sz is not None:
                        if self.aux:
                            # a/a2.shape (z,y,x,1) b/b2.shape (z,y,x,chn), c2.shape(z,y,x,2)
                            a_img, b_img, c_img = random_patch(a_low, a_hgh, b_low, b_hgh, c_low, c_hgh,
                                                               patch_shape=(self.ptch_z_sz, self.ptch_sz, self.ptch_sz),
                                                               p_middle=self.p_middle, task=self.task, io=self.io, ptch_seed=self.ptch_seed)
                        else:
                            a_img, b_img = random_patch(a_low, a_hgh, b_low, b_hgh, c_low, c_hgh,
                                                        patch_shape=(self.ptch_z_sz, self.ptch_sz, self.ptch_sz),
                                                        p_middle=self.p_middle, task=self.task, io=self.io, ptch_seed=self.ptch_seed)
                    else:
                        raise Exception("self.ptch_sz is None or self.ptch_sz == self.trgt_sz")

                    a_img = self.tune_axis(a_img)
                    b_img = self.tune_axis(b_img)
                    if self.aux:  # only for one output, b_img is an array
                        c_img = self.tune_axis(c_img)
                        if self.ds==2:
                            yield a_img, [b_img, c_img, b_img, b_img]
                        else:
                            yield a_img, [b_img, c_img]
                    else:
                        if self.ds == 2:  # only for one output, b_img is an array
                            yield a_img, [b_img, b_img, b_img]

                        else:
                            yield a_img, b_img

            self.epoch_nb += 1
            if self.shuffle:
                if not len(q1.index_list):
                    q1.index_list = random.sample(list(range(self.n)),
                                                  len(list(range(self.n))))  # regenerate the nex shuffle index
                elif not len(q2.index_list):
                    q2.index_list = random.sample(list(range(self.n)),
                                                  len(list(range(self.n))))  # regenerate the nex shuffle index
                # else:
                #     raise Exception("two queues are all not empty at the end of the epoch")
            else:
                if not len(q1.index_list):
                    q1.index_list = list(range(self.n))
                elif not len(q2.index_list):
                    q2.index_list = list(range(self.n))

            #             # import matplotlib.pyplot as plt
            #             # plt.figure()
            #             # plt.imshow(x1[0,72,:,:,0])
            #             # plt.savefig('x1_y.png')
            #             # plt.close()
            #             # plt.figure()
            #             # plt.imshow(x2[0,72, :, :, 0])
            #             # plt.savefig('x2_y.png')
            #             # plt.close()

