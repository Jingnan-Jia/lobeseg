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
from futils.util import downsample
from functools import wraps

""""""
def rp_dcrt(fun): # decorator to repeat a function until succeessful
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

class TwoScanIterator(Iterator):
    """Class to iterate A and B 3D scans (mhd or nrrd) at the same time."""

    def __init__(self,
                 directory,
                 task=None,
                 a_dir_name='ori_ct', b_dir_name='gdth_ct',
                 sub_dir=None,
                 a_extension='.mhd', b_extension='.nrrd', c_extension='.nrrd',
                 ptch_sz=None, ptch_z_sz=None,
                 trgt_sz=None, trgt_z_sz=None,
                 trgt_space=None, trgt_z_space=None,
                 data_argum=True,
                 patches_per_scan=5,
                 ds=2,
                 labels=[],
                 batch_size=1,
                 shuffle=True,
                 seed=None,
                 nb=None,
                 nb_no_label=None,
                 no_label_dir=None,
                 p_middle=None,
                 phase='train',
                 aux=None,
                 mtscale=None):
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
        self.mtscale = mtscale
        self.task = task
        self.phase = phase
        if self.mtscale or self.task=='vessel':  # I hope to used p_middle for mtscale
            self.p_middle = p_middle
        else:
            self.p_middle=None

        self.sub_dir = sub_dir
        self.ds = ds
        self.labels = labels
        self.patches_per_scan = patches_per_scan
        self.trgt_sz = trgt_sz
        self.trgt_z_sz = trgt_z_sz
        self.trgt_space = trgt_space
        self.trgt_z_space = trgt_z_space
        self.ptch_sz = ptch_sz
        self.ptch_z_sz = ptch_z_sz
        self.trgt_sp_list = [self.trgt_z_space, self.trgt_space, self.trgt_space]
        self.trgt_sz_list = [self.trgt_z_sz, self.trgt_sz, self.trgt_sz]

        self.aux=aux
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
            if nb_no_label:
                self.filenames = self.filenames[:nb_no_label]
        else:
            self.a_dir = os.path.join(directory, a_dir_name, sub_dir)
            self.b_dir = os.path.join(directory, b_dir_name, sub_dir)

            self.a_extension = a_extension
            self.b_extension = b_extension

            a_files = set(x.split(a_extension)[0].split(self.a_dir + '/')[-1] for x in
                          sorted(glob.glob(self.a_dir + '/*' + self.a_extension)))
            b_files = set(x.split(b_extension)[0].split(self.b_dir + '/')[-1] for x in
                          sorted(glob.glob(self.b_dir + '/*' + self.b_extension)))

            # Files inside a and b should have the same name. Images without a pair
            # are discarded.
            self.filenames = sorted(list(a_files.intersection(b_files)))
            if nb:
                self.filenames = self.filenames[:nb]

        print('task:', self.task, 'phase', self.phase)
        print('from this directory:', self.a_dir)
        print('these files are used ', self.filenames)
        if self.filenames is []:
            raise Exception('empty dataset')

        # ----next part is the common parameters for segmentation tasks and reconstruction task
        # axis order should be like this
        self.channel_index = 3
        self.row_index = 1
        self.col_index = 2

        self.data_argum = data_argum
        if self.data_argum:
            self.rotation_range = 0.05
            self.height_shift_range = 0.05
            self.width_shift_range = 0.05
            self.shear_range = 0.05
            self.fill_mode = 'constant'
            self.zoom_range = 0.05
            if np.isscalar(self.zoom_range):
                self.zoom_range = [1 - self.zoom_range, 1 + self.zoom_range]
            elif len(self.zoom_range) == 2:
                self.zoom_range = [self.zoom_range[0], self.zoom_range[1]]

        super(TwoScanIterator, self).__init__(len(self.filenames), batch_size, seed,
                                              shuffle)

    def _normal_normalize(self, scan):
        """returns normalized (0 mean 1 variance) scan"""
        scan = (scan - np.mean(scan)) / (np.std(scan))
        return scan

    def load_scan(self, file_name):
        """Load mhd or nrrd 3d scan"""

        scan, origin, self.spacing = futil.load_itk(file_name)

        return np.expand_dims(scan, axis=-1)  # size=(z,x,y,1)

    def _load_img_pair(self, idx):
        """Get a pair of images with index idx."""

        a_fname = self.filenames[idx] + self.a_extension
        print('start load file: ', a_fname)

        a = self.load_scan(file_name=os.path.join(self.a_dir, a_fname))  # (200, 512, 512, 1)

        # a = np.array(a)
        a = futil.normalize(a)  # threshold to [-1000,400], then rescale to [0,1]
        a = self._normal_normalize(a)

        if self.task == 'no_label':
            return a, a
        else:
            b_fname = self.filenames[idx] + self.b_extension
            b = self.load_scan(file_name=os.path.join(self.b_dir, b_fname))  # (200, 512, 512, 1)
            # b = np.array(b)

            if not self.aux:
                return a, b
            else:
                c_fname = self.filenames[idx] + self.c_extension
                c = self.load_scan(file_name=os.path.join(self.c_dir, c_fname))  # (200, 512, 512, 1)
                # c = np.array(c, dtype='float')

                return a, b, c

    def _random_transform(self, a, b, c=None, is_batch=True):
        """
        Random dataset augmentation.

        Adapted from https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
        """

        if is_batch is False:
            # a and b are single images, so they don't have image number at index 0
            img_row_index = self.row_index - 1
            img_col_index = self.col_index - 1
            img_channel_index = self.channel_index - 1
        else:
            img_row_index = self.row_index
            img_col_index = self.col_index
            img_channel_index = self.channel_index
        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range,
                                                    self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) \
                 * a.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) \
                 * a.shape[img_col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
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
                                     fill_mode=self.fill_mode, cval=np.min(a_))
                b_ = apply_transform(b_, transform_matrix, img_channel_index - 1,
                                     fill_mode=self.fill_mode, cval=0)
                c_ = apply_transform(c_, transform_matrix, img_channel_index - 1,
                                     fill_mode=self.fill_mode, cval=0)

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
                                     fill_mode=self.fill_mode, cval=np.min(a_))
                b_ = apply_transform(b_, transform_matrix, img_channel_index - 1,
                                     fill_mode=self.fill_mode, cval=0)

                A.append(a_)
                B.append(b_)

            a = np.array(A)
            b = np.array(B)

            return a, b

    def one_hot_encode_3D(self, patch, labels):
        # todo: simplify this function

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

    @rp_dcrt
    def next(self):
        """Get the next pair of the sequence."""

        # Lock the iterator when the index is changed.
        with self.lock:
            index_array, _, current_batch_size = next(self.index_generator)
            print('index_array: ', index_array)

        for i, j in enumerate(index_array):

            if self.aux:
                a_ori, b, c = self._load_img_pair(j)
            else:
                a_ori, b = self._load_img_pair(j)
            if self.task!='no_label':
                b = self.one_hot_encode_3D(b, self.labels)
                c = self.one_hot_encode_3D(c, [0,1]) if self.aux else None

            if self.data_argum:
                if self.aux:
                    a_ori, b, c = self._random_transform(a_ori, b, c)
                else:
                    a_ori, b = self._random_transform(a_ori, b)

            if any(self.trgt_sp_list) or any(self.trgt_sz_list):
                if not self.mtscale or self.task == 'lobe':
                    a = downsample(a_ori,
                                   ori_space=self.spacing, trgt_space=self.trgt_sp_list,
                                   ori_sz=a_ori.shape, trgt_sz=self.trgt_sz_list,
                                   order=1)
                    b = downsample(b,
                                   ori_space=self.spacing, trgt_space=self.trgt_sp_list,
                                   ori_sz=b.shape, trgt_sz=self.trgt_sz_list,
                                   order=0)
                    c = downsample(c,
                                   ori_space=self.spacing, trgt_space=self.trgt_sp_list,
                                   ori_sz=c.shape,trgt_sz=self.trgt_sz_list,
                                   order=0) if self.aux else None
                    a2 = a_ori if self.mtscale else None
                else:
                    a = a_ori
                    a2 = downsample(a_ori,
                                   ori_space=self.spacing, trgt_space=self.trgt_sp_list,
                                   ori_sz=a_ori.shape,trgt_sz=self.trgt_sz_list,
                                   order=1) if self.mtscale else None

            else:
                a = a_ori
                a2 = None  # if trgt_sp or trgt_sz is not assigned, it means that mtscale is False

            print('before patching, the shape of a is ', a.shape)
            print('before patching, the shape of a2 is ', a2.shape) if self.mtscale else print('')


            A = []
            B = []
            C = [] if self.aux else None
            for _ in range(self.patches_per_scan):
                if self.aux:
                    if self.mtscale or ((self.ptch_sz is not None) and (self.ptch_sz != self.trgt_sz)):
                        a_img, b_img, c_img = random_patch(a, b, c, patch_shape=(self.ptch_z_sz, self.ptch_sz, self.ptch_sz), p_middle=self.p_middle, a2=a2)
                    else:
                        a_img, b_img, c_img = a, b, c
                else:
                    if self.mtscale or ((self.ptch_sz is not None) and (self.ptch_sz != self.trgt_sz)):
                        a_img, b_img = random_patch(a, b, patch_shape=(self.ptch_z_sz, self.ptch_sz, self.ptch_sz), p_middle=self.p_middle, a2=a2)
                    else:
                        a_img, b_img = a, b

                batch_a = a_img.copy() # shape: 96, 144, 144, 1
                batch_b = b_img.copy()
                batch_c = c_img.copy() if self.aux else None

                A.append(batch_a)
                B.append(batch_b)
                C.append(batch_c) if self.aux else None
            a_np = np.array(A)
            b_np = np.array(B)
            c_np = np.array(C) if self.aux else None
            print('after patching, the shape is ', a_np.shape)

            if self.aux:
                return [a_np, b_np, c_np]
            else:
                return [a_np, b_np]


    def generator(self):
        x = None
        while 1:

            if self.aux:
                x, y, y_aux = self.next()
                x_b = np.rollaxis(x, 1, 4)
                y_b = np.rollaxis(y, 1, 4)
                y_aux_b = np.rollaxis(y_aux, 1, 4)
                print('prepare feed the data to model, x, y, y_aux', x_b.shape, y_b.shape, y_aux_b.shape)
                for x, y, y_aux in zip(x_b, y_b, y_aux_b):
                    x = x[np.newaxis, ...]
                    y = y[np.newaxis, ...]
                    y_aux = y_aux[np.newaxis, ...]
                    if self.mtscale:
                        x1 = x[..., 0]
                        x1 = x1[..., np.newaxis]
                        x2 = x[..., 1]
                        x2 = x2[..., np.newaxis]
                        if self.ds == 2:
                            yield [x1, x2], [y, y_aux, y, y]

                        else:
                            yield [x1, x2], [y, y_aux]
                    else:

                        if self.ds == 2:
                            yield x, [y, y_aux, y, y]

                        else:
                            yield x, [y, y_aux]
            else:

                x, y = self.next()
                if self.task=='no_label':
                    y = x
                x_b = np.rollaxis(x, 1, 4)
                y_b = np.rollaxis(y, 1, 4)
                print('prepare feed the data to model, x, y', x_b.shape, y_b.shape)

                for x, y in zip(x_b, y_b):
                    x = x[np.newaxis, ...]
                    y = y[np.newaxis, ...]
                    if self.mtscale:
                        x1 = x[..., 0]
                        x1 = x1[..., np.newaxis]
                        x2 = x[..., 1]
                        x2 = x2[..., np.newaxis]

                        # import matplotlib.pyplot as plt
                        # plt.figure()
                        # plt.imshow(x1[0,72,:,:,0])
                        # plt.savefig('x1_y.png')
                        # plt.close()
                        # plt.figure()
                        # plt.imshow(x2[0,72, :, :, 0])
                        # plt.savefig('x2_y.png')
                        # plt.close()

                        if self.ds == 2:
                            yield [x1, x2], [y, y, y]

                        else:
                            yield [x1, x2], y
                    else:

                        if self.ds == 2:
                            yield x, [y, y, y]

                        else:
                            yield x, y



