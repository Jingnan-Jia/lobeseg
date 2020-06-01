"""Auxiliar methods to deal with loading the dataset."""
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

"""3D Extension of the work in https://github.com/costapt/vess2ret/blob/master/util/data.py"""

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
                 no_label_dir=None,
                 p_middle=None,
                 phase='train',
                 aux=None):
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
        self.task = task
        self.phase = phase
        self.p_middle = p_middle
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
            if nb:
                self.filenames = self.filenames[:nb]
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
        print('downsampling ori ct ...', a_fname)
        a = self.downscale_scan(a, order=1)
        if self.task == 'no_label':
            return a, a

        else:
            b_fname = self.filenames[idx] + self.b_extension
            b = self.load_scan(file_name=os.path.join(self.b_dir, b_fname))  # (200, 512, 512, 1)
            # b = np.array(b)
            print('downsampling masks ...', b_fname)
            b = self.downscale_scan(b, order=0)  # for masks, order=0, means nearest neighbor downsampling
            if not self.aux:
                return a, b
            else:
                c_fname = self.filenames[idx] + self.c_extension
                c = self.load_scan(file_name=os.path.join(self.c_dir, c_fname))  # (200, 512, 512, 1)
                # c = np.array(c, dtype='float')
                print('downsampling aux mask ...', c_fname)
                c = self.downscale_scan(c, order=0)
                return a, b, c

    def downscale_scan(self, scan, order=1):  # scan.shape: (200, 512, 512, 1)

        time1 = time.time()
        # print('time before downscale:', time1)
        print('shape before downscale', scan.shape)
        if self.trgt_space and self.trgt_z_space:  # put trgt space more priority
            trgt_sp_list = [self.trgt_z_space, self.trgt_space, self.trgt_space] # (x, y, x)

            zoom_seq = np.array(self.spacing, dtype='float') / np.array(trgt_sp_list, dtype='float')  # order is correct
            zoom_seq = np.append(zoom_seq, 1)  # because here the scan is 4 dimentions
            print('downsample to target space ', [self.trgt_z_space, self.trgt_space, self.trgt_space, 1], 'zoom_seq',
                  zoom_seq)
            s = ndimage.interpolation.zoom(scan, zoom_seq, order=order, prefilter=order)  # (128, 256, 256, 1)

        elif self.trgt_sz and self.trgt_z_sz:
            trgt_sz_list = [self.trgt_z_sz, self.trgt_sz, self.trgt_sz, 1]
            zoom_seq = np.array(trgt_sz_list, dtype='float') / np.array(scan.shape, dtype='float')  # order is correct
            print('downsample to target size', trgt_sz_list, 'zoom_seq', zoom_seq)
            s = ndimage.interpolation.zoom(scan, zoom_seq, order=order, prefilter=order)  # (128, 256, 256, 1)

        else:
            s = scan
            print('do not assign correct trgt space or size, please assign the correct rescale method')

        # zoom_seq = np.array([z_length, sz, sz, 1], dtype='float') / np.array([100, 100, 100, 1], dtype='float')

        time2 = time.time()
        # print('time after downscle:', time2)
        print('shape after downscale', s.shape)
        print('time during downscle:', time2 - time1, file=sys.stdout)
        # print('time during downscle:', time2 - time1, file=sys.stderr)

        return s

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

    def next(self):
        """Get the next pair of the sequence."""

        # Lock the iterator when the index is changed.
        with self.lock:
            index_array, _, current_batch_size = next(self.index_generator)
            print('index_array: ', index_array)

        for i, j in enumerate(index_array):

            if  self.aux:
                a, b, c = self._load_img_pair(j)
                b = self.one_hot_encode_3D(b, self.labels)
                c = self.one_hot_encode_3D(c, [0,1])
                if self.data_argum:
                    a, b, c = self._random_transform(a, b, c)
                print('before patching, the shape is ', a.shape)

                A = []
                B = []
                C = []
                for _ in range(self.patches_per_scan):

                    if (self.ptch_sz is not None) and (self.ptch_sz != self.trgt_sz):
                        a_img, b_img, c_img = random_patch(a, b, c, patch_shape=(self.ptch_z_sz, self.ptch_sz, self.ptch_sz), p_middle=self.p_middle)
                    else:
                        a_img, b_img, c_img = a, b, c

                    batch_a = a_img.copy()
                    batch_b = b_img.copy()
                    batch_c = c_img.copy()

                    A.append(batch_a)
                    B.append(batch_b)
                    C.append(batch_c)
                a_np = np.array(A)
                b_np = np.array(B)
                c_np = np.array(C)
                print('after patching, the shape is ', a_np.shape)


                return [a_np, b_np, c_np]


            else:
                a, b = self._load_img_pair(j)
                if self.task != 'no_label':
                    b = self.one_hot_encode_3D(b, self.labels)
                # apply random affine transformation
                if self.data_argum:
                    a, b = self._random_transform(a, b)

                print('before patching, the shape is ', a.shape)

                A = []
                B = []
                for _ in range(self.patches_per_scan):

                    if (self.ptch_sz is not None) and (self.ptch_sz != self.trgt_sz):
                        a_img, b_img = random_patch(a, b, patch_shape=(self.ptch_z_sz, self.ptch_sz, self.ptch_sz), p_middle=self.p_middle)
                    else:
                        a_img, b_img = a, b


                    batch_a = a_img.copy()
                    batch_b = b_img.copy()

                    A.append(batch_a)
                    B.append(batch_b)
                a_np = np.array(A)
                b_np = np.array(B)
                print('after patching, the shape is ', a_np.shape)

                return [a_np, b_np]




    def generator(self):
        x = None
        while 1:
            if self.model_mt_scales:
                # todo

                x, y = self.next()

                x_b = np.rollaxis(x, 1, 4)
                y_b = np.rollaxis(y, 1, 4)
                print('prepare feed the data to model, x, y', x_b.shape, y_b.shape)

                for x, y in zip(x_b, y_b):
                    x1 = x
                    x2 =
                    if self.task == 'no_label':
                        yield x[np.newaxis, ...], y[np.newaxis, ...]
                    else:
                        if self.ds == 2:
                            yield x[np.newaxis, :, :, :, :], [y[np.newaxis, :, :, :, :], y[np.newaxis, :, :, :, :],
                                                              y[np.newaxis, :, :, :, :]]
                        else:
                            yield x[np.newaxis, :, :, :, :], [y[np.newaxis, :, :, :, :]]

            else:
                if self.aux:
                    # for i in range(5):
                    x, y, y_aux = self.next()
                    #     try:
                    #         x, y, y_aux = self.next()
                    #         break
                    #     except:
                    #         print('fail to generate this ct for ' + self.task + ', pass it', file=sys.stderr)
                    #         pass
                    # if x is None:
                    #     raise Exception('failed 5 times generation of ct, please check dataset or rescale method, like trgt space or trgt size')

                    x_b = np.rollaxis(x, 1, 4)
                    y_b = np.rollaxis(y, 1, 4)
                    y_aux_b = np.rollaxis(y_aux, 1, 4)
                    print('prepare feed the data to model, x, y, y_aux', x_b.shape, y_b.shape, y_aux_b.shape)
                    for x, y, y_aux in zip(x_b, y_b, y_aux_b):
                        if self.ds == 2:
                            yield x[np.newaxis, :, :, :, :], [y[np.newaxis, :, :, :, :], y_aux[np.newaxis, :, :, :, :], y[np.newaxis, :, :, :, :], y[np.newaxis, :, :, :, :]]

                        else:
                            yield x[np.newaxis, :, :, :, :], [y[np.newaxis, :, :, :, :], y_aux[np.newaxis, :, :, :, :]]
                else:
                    # for i in range(5):
                    x, y = self.next()
                        # try:
                        #     x, y = self.next()
                        #     break
                        # except:
                        #     print('fail to generate this ct for ' + self.task + ', pass it', file=sys.stderr)
                        #     pass
                    # if x is None:
                    #     raise Exception('failed 5 times generation of ct, please check dataset or rescale method, like trgt space or trgt size')
                    x_b = np.rollaxis(x, 1, 4)
                    y_b = np.rollaxis(y, 1, 4)
                    print('prepare feed the data to model, x, y', x_b.shape, y_b.shape)

                    for x, y in zip(x_b, y_b):
                        if self.task == 'no_label':
                            yield x[np.newaxis, ...], y[np.newaxis, ...]
                        else:
                            if self.ds == 2:
                                yield x[np.newaxis, :, :, :, :], [y[np.newaxis, :, :, :, :], y[np.newaxis, :, :, :, :], y[np.newaxis, :, :, :, :]]
                            else:
                                yield x[np.newaxis, :, :, :, :], [y[np.newaxis, :, :, :, :]]
