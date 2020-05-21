"""Auxiliar methods to deal with loading the dataset."""
import os
import random
import numpy as np
from tensorflow.keras import backend as K
from image import Iterator, load_img, img_to_array
from image import apply_transform
from image import transform_matrix_offset_center
from skimage import color, transform
from futils.vpatch import random_patch
import futils.util as futil
from scipy import ndimage
import time
import glob
import sys

"""3D Extension of the work in https://github.com/costapt/vess2ret/blob/master/util/data.py"""


#
# class TwoScanIteratorold(Iterator):
#     """Class to iterate A and B 3D scans (mhd or nrrd) at the same time."""
#
#     def __init__(self,
#                  directory,
#                  task = 'lobe',
#                  a_dir_name='ori_ct', b_dir_name='gdth_ct', c_dir_name='aux_gdth',
#                  sub_dir='GLUCOL_isotropic1dot5',
#                  a_extension='.mhd', b_extension='.nrrd', c_extension='.nrrd',
#                  N=-1,
#                  ptch_sz=64, ptch_z_sz=16,
#                  trgt_sz=256, trgt_z_sz=128,
#                  trgt_space=None, trgt_z_space=None,
#                  data_argum=True,
#                  new_spacing=None,
#                  weight_map=10,
#                  patches_per_scan=5,
#
#                  ds=2, labels=[],
#                  batch_size=1, shuffle=True, seed=None, nb=None):
#         """
#         Iterate through two directories at the same time.
#
#         Files under the directory A and B with the same name will be returned
#         at the same time.
#         Parameters:
#         - directory: base directory of the dataset. Should contain two
#         directories with name a_dir_name and b_dir_name;
#         - a_dir_name: name of directory under directory that contains the A
#         images;
#         - b_dir_name: name of directory under directory that contains the B
#         images;
#         - c_dir_name : this is the auxiliar output folder
#         - a/b/c_extension : type of the scan: nrrd or mhd (no dicom available)
#
#         - N: if -1 uses the entire dataset. Otherwise only uses a subset;
#         - batch_size: the size of the batches to create;
#         - shuffle: if True the order of the images in X will be shuffled;
#         - ds: number of deep classifiers
#         - labels: output classes
#         - weight_map = value to give to lung borders
#
#         """
#         self.task = task
#
#         if self.task=='no_label':
#             self.new_spacing = new_spacing
#             self.a_dir = os.path.join(directory, a_dir_name)
#             self.a_extension = a_extension
#             a_files = set(x.split(a_extension)[0].split(self.a_dir + '/')[-1] for x in
#                           sorted(glob.glob(self.a_dir + '/*' + self.a_extension)))
#             self.filenames = list(a_files)
#         else:
#             self.sub_dir = sub_dir
#             if sub_dir is None:
#                 self.a_dir = os.path.join(directory, a_dir_name)
#                 self.b_dir = os.path.join(directory, b_dir_name)
#
#                 if c_dir_name is None:
#                     self.c_dir = None
#                     self.aux = False
#                 else:
#                     self.c_dir = os.path.join(directory, c_dir_name)
#                     self.aux = True
#             else:
#                 self.a_dir = os.path.join(directory, a_dir_name, sub_dir)
#                 self.b_dir = os.path.join(directory, b_dir_name, sub_dir)
#
#                 if c_dir_name is None:
#                     self.c_dir = None
#                     self.aux = False
#                 else:
#                     self.c_dir = os.path.join(directory, c_dir_name, sub_dir)
#                     self.aux = True
#
#             self.a_extension = a_extension
#             self.b_extension = b_extension
#             self.c_extension = c_extension
#
#             a_files = set(x.split(a_extension)[0].split(self.a_dir + '/')[-1] for x in
#                           sorted(glob.glob(self.a_dir + '/*' + self.a_extension)))
#             b_files = set(x.split(b_extension)[0].split(self.b_dir + '/')[-1] for x in
#                           sorted(glob.glob(self.b_dir + '/*' + self.b_extension)))
#
#             # Files inside a and b should have the same name. Images without a pair
#             # are discarded.
#             self.filenames = list(a_files.intersection(b_files))
#             if nb:
#                 self.filenames = self.filenames[:5]
#
#             self.b_fnames = self.filenames
#             if c_dir_name is not None:
#                 self.c_fnames = self.filenames
#
#             self.weight_map = weight_map
#
#             self.ds = ds
#             self.labels = labels
#             if (self.labels == []):
#                 self.is_b_categorical = False
#             else:
#                 self.is_b_categorical = True
#
#         #----next part is the common parameters for segmentation tasks and reconstruction task
#         # axis order should be like this
#         self.channel_index = 3
#         self.row_index = 1
#         self.col_index = 2
#
#         self.data_argum = data_argum
#         if self.data_argum:
#             self.rotation_range = 0.05
#             self.height_shift_range = 0.05
#             self.width_shift_range = 0.05
#             self.shear_range = 0.05
#             self.fill_mode = 'constant'
#             self.zoom_range = 0.05
#             if np.isscalar(self.zoom_range):
#                 self.zoom_range = [1 - self.zoom_range, 1 + self.zoom_range]
#             elif len(zoom_range) == 2:
#                 self.zoom_range = [self.zoom_range[0], self.zoom_range[1]]
#
#         self.patches_per_scan = patches_per_scan
#
#         # Use only a subset of the files. Good to easily overfit the model
#         if N > 0:
#             random.shuffle(self.filenames)
#             self.filenames = self.filenames[:N]
#         self.N = len(self.filenames)
#
#         # sizes
#         self.trgt_sz = trgt_sz
#         self.trgt_z_sz = trgt_z_sz
#         self.trgt_space = trgt_space
#         self.trgt_z_space = trgt_z_space
#         self.ptch_sz = ptch_sz
#         self.ptch_z_sz = ptch_z_sz
#
#         super(TwoScanIterator, self).__init__(len(self.filenames), batch_size, seed,
#                                               shuffle)
#
#
#     def _normal_normalize(self, scan):
#         """returns normalized (0 mean 1 variance) scan"""
#         scan = (scan - np.mean(scan)) / (np.std(scan))
#         return scan
#
#     def load_scan(self, file_name, extension):
#         """Load mhd or nrrd 3d scan"""
#
#         if extension == '.mhd':
#             scan, origin, self.spacing = futil.load_itk(file_name)
#
#         elif extension == '.nrrd':
#             scan, origin, self.spacing = futil.load_nrrd(file_name)
#
#         if self.task=='no_label' and self.new_spacing is not None:
#             time3 = time.time()
#             # print ('time before isotropic:', time3)
#             print('spacing before isotropic', spacing)
#             # time.sleep(120)
#
#             zoom_seq = np.array(spacing, dtype='float') / np.array(self.new_spacing, dtype='float')
#             scan = ndimage.interpolation.zoom(scan, zoom_seq, order=1, prefilter=1)
#             time4 = time.time()
#             # print ('time after isotropic:', time4)
#             print('time during isotropic:', time4 - time3)
#
#         return np.expand_dims(scan, axis=-1)
#
#     def _load_img_pair(self, idx):
#         """Get a pair of images with index idx."""
#
#         a_fname = self.filenames[idx] + self.a_extension
#         a = self.load_scan(file_name=os.path.join(self.a_dir, a_fname),
#                            extension=self.a_extension)  # (200, 512, 512, 1)
#
#         a = np.array(a)
#         a = futil.normalize(a)  # we need to change the name of this
#         a = self._normal_normalize(a)
#         a = self.downscale_scan(a, order=1)
#         if self.task == 'no_label':
#             return a, a
#
#         else:
#             b_fname = self.filenames[idx] + self.b_extension
#             b = self.load_scan(file_name=os.path.join(self.b_dir, b_fname),
#                                extension=self.b_extension)  # (200, 512, 512, 1)
#             b = np.array(b)
#             # print(np.max(b))
#             # print(np.sum(b))
#             # print(b)
#             # for i in range(700):
#             #     i = i + 200
#             #     b_ = b[i,:,:,0]
#             b = self.downscale_scan(b, order=0)
#             if not self.aux:
#                 return a, b
#             else:
#                 c_fname = self.filenames[idx] + self.c_extension
#                 c = self.load_scan(file_name=os.path.join(self.c_dir, c_fname),
#                                    extension=self.c_extension)  # (200, 512, 512, 1)
#                 c = np.array(c, dtype='float')
#                 c = self.downscale_scan(c, order=0)
#                 return a, b, c
#
#
#
#
#
#     def downscale_scan(self, scan, order=1):  # scan.shape: (200, 512, 512, 1)
#
#         time1 = time.time()
#         # print('time before downscale:', time1)
#         print('shape before downscale', scan.shape)
#         if self.trgt_space and self.trgt_z_space:  # put trgt space more priority
#             trgt_sp_list = [self.trgt_z_space, self.trgt_space, self.trgt_space]
#
#             zoom_seq = np.array(self.spacing, dtype='float') / np.array(trgt_sp_list, dtype='float')  # order is correct
#             zoom_seq = np.append(zoom_seq, 1)  # because here the scan is 4 dimentions
#             print('downsample to target space ', [self.trgt_z_space, self.trgt_space, self.trgt_space, 1], 'zoom_seq',
#                   zoom_seq)
#             s = ndimage.interpolation.zoom(scan, zoom_seq, order=order, prefilter=order)  # (128, 256, 256, 1)
#
#
#         elif self.trgt_sz and self.trgt_z_sz:
#             trgt_sz_list = [self.trgt_z_sz, self.trgt_sz, self.trgt_sz, 1]
#             zoom_seq = np.array(trgt_sz_list, dtype='float') / np.array(scan.shape, dtype='float')  # order is correct
#             print('downsample to target size', trgt_sz_list, 'zoom_seq', zoom_seq)
#             s = ndimage.interpolation.zoom(scan, zoom_seq, order=order, prefilter=order)  # (128, 256, 256, 1)
#
#         else:
#             s = scan
#             print('do not rescale, please assign the correct rescale method')
#
#         # zoom_seq = np.array([z_length, sz, sz, 1], dtype='float') / np.array([100, 100, 100, 1], dtype='float')
#
#
#         time2 = time.time()
#         # print('time after downscle:', time2)
#         print('shape after downscale', s.shape)
#         print('time during downscle:', time2 - time1, file=sys.stdout)
#         print('time during downscle:', time2 - time1, file=sys.stderr)
#
#
#         return s
#
#     def _random_transform(self, a, b, is_batch=True):
#         """
#         Random dataset augmentation.
#
#         Adapted from https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
#         """
#
#         if is_batch is False:
#             # a and b are single images, so they don't have image number at index 0
#             img_row_index = self.row_index - 1
#             img_col_index = self.col_index - 1
#             img_channel_index = self.channel_index - 1
#         else:
#             img_row_index = self.row_index
#             img_col_index = self.col_index
#             img_channel_index = self.channel_index
#         # use composition of homographies to generate final transform that needs to be applied
#         if self.rotation_range:
#             theta = np.pi / 180 * np.random.uniform(-self.rotation_range,
#                                                     self.rotation_range)
#         else:
#             theta = 0
#         rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
#                                     [np.sin(theta), np.cos(theta), 0],
#                                     [0, 0, 1]])
#         if self.height_shift_range:
#             tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) \
#                  * a.shape[img_row_index]
#         else:
#             tx = 0
#
#         if self.width_shift_range:
#             ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) \
#                  * a.shape[img_col_index]
#         else:
#             ty = 0
#
#         translation_matrix = np.array([[1, 0, tx],
#                                        [0, 1, ty],
#                                        [0, 0, 1]])
#
#         if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
#             zx, zy = 1, 1
#         else:
#             zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
#         zoom_matrix = np.array([[zx, 0, 0],
#                                 [0, zy, 0],
#                                 [0, 0, 1]])
#
#         if self.shear_range:
#             shear = np.random.uniform(-self.shear_range, self.shear_range)
#         else:
#             shear = 0
#         shear_matrix = np.array([[1, -np.sin(shear), 0],
#                                  [0, np.cos(shear), 0],
#                                  [0, 0, 1]])
#
#         transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix),
#                                   zoom_matrix)
#
#         h, w = a.shape[img_row_index], a.shape[img_col_index]
#         transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
#
#         A = []
#         B = []
#
#         for a_, b_ in zip(a, b):
#             a_ = apply_transform(a_, transform_matrix, img_channel_index - 1,
#                                  fill_mode=self.fill_mode, cval=np.min(a_))
#             b_ = apply_transform(b_, transform_matrix, img_channel_index - 1,
#                                  fill_mode=self.fill_mode, cval=0)
#
#             A.append(a_)
#             B.append(b_)
#
#         a = np.array(A)
#         b = np.array(B)
#
#         return a, b
#
#     def one_hot_encode_3D(self, patch, labels):
#
#         # assert len(patch.shape)==5 # (5, 128, 128, 64, 1)
#         labels = np.array(labels)  # i.e. [0,4,5,6,7,8]
#         N_classes = labels.size  # 6, similiar with len(labels)
#         if len(patch.shape) == 5:  # (5, 128, 128, 64, 1)
#             patch = np.reshape(patch, (patch.shape[0], patch.shape[1], patch.shape[2], patch.shape[3]))
#         elif len(patch.shape) == 4:  # (128, 128, 64, 1)
#             patch = np.reshape(patch, (patch.shape[0], patch.shape[1], patch.shape[2]))
#         patches = []
#         # print('patch.shape', patch.shape)
#         for i, l in enumerate(labels):
#             a = np.where(patch != l, 0, 1)
#             patches.append(a)
#
#         patches = np.array(patches)
#         patches = np.rollaxis(patches, 0, len(patches.shape))  # from [6, 64, 128, 128] to [64, 128, 128, 6]?
#
#         # print('patches.shape after on hot encode 3D', patches.shape)
#
#         return np.float64(patches)
#
#     def next(self):
#         """Get the next pair of the sequence."""
#
#         # Lock the iterator when the index is changed.
#         with self.lock:
#             index_array, _, current_batch_size = next(self.index_generator)
#             print('index_array: ', index_array)
#
#         for i, j in enumerate(index_array):
#             if self.task!='no_label' and self.aux:
#                 a, b, c = self._load_img_pair(j)
#                 # for i in b:
#                 #     for j in i:
#                 #         print(j)
#
#
#                 # give weights for the aux map
#                 if (np.min(c) == 1):
#                     c -= 1
#                 c *= self.weight_map
#                 c += 1
#                 # we include our aux ground truth in the first channel. This is not cool, but it's easier to make the transformations :(
#                 np.copyto(b[..., 0], c[..., 0])
#             else:
#                 a, b = self._load_img_pair(j)
#             # print(b)
#
#             if self.task!='no_label' and self.is_b_categorical:
#                 b = self.one_hot_encode_3D(b, self.labels)
#                 # for i in b:
#                 #     print(i[:,:,0])
#                 # print(np.sum(b))
#                 # for i in b:
#                 #
#                 #     import csv
#                 #     with open('tst.csv', 'w') as f:
#                 #         writer = csv.writer(f)
#                 #         writer.writerow(i[:,:,0])
#
#             # apply random affine transformation
#             if self.data_argum:
#                 a, b = self._random_transform(a, b)
#             print('before patching, the shape is ', a.shape)
#
#             A = []
#             B = []
#             for _ in range(self.patches_per_scan):
#
#                 if self.ptch_sz is not None and self.ptch_sz != self.trgt_sz:
#                     a_img, b_img = random_patch(a, b, patch_shape=(self.ptch_z_sz, self.ptch_sz, self.ptch_sz))
#                 else:
#                     a_img, b_img = a, b
#
#                 if self.task!='no_label' and self.aux:  # why???
#                     b_0 = b_img[..., 0]
#                     b_0[b_0 < 1] = 1
#
#                 batch_a = a_img.copy()
#                 batch_b = b_img.copy()
#
#                 A.append(batch_a)
#                 B.append(batch_b)
#
#         return [np.array(A), np.array(B)]
#
#     def split_output(self, scan):
#         """here we separate again the aux output that was in channel 0"""
#         # print('scan.shape', scan.shape)
#         # output 1- original gt
#         o1 = scan[np.newaxis, ...]
#         # print('o1.shape', o1.shape)
#
#         w = scan[..., 0].copy()
#         # print('w.shape', w.shape)
#
#         # output 2- weight
#         o2 = 1 - (w - 1) / self.weight_map  # background channel
#         # print('o2.shape', o2.shape)
#
#         o2 = o2[..., np.newaxis]
#         # print('o2.shape', o2.shape)
#
#         o2 = np.append(o2, 1 - o2, axis=-1)
#         # print('o2.shape', o2.shape)
#
#         output = [o1, o2[np.newaxis, ...]]
#         # print('output.len', len(output))
#
#         for _ in range(self.ds):
#             output.append(o1)
#         print('split_out.shape:', len(output), output[0].shape, output[1].shape, output[2].shape,
#               output[2].shape
#               )
#
#         return output
#
#     def generator(self):
#
#         while 1:
#             x, y = self.next()
#
#             # adapt to input tensor
#             # print('..........after next.........................')
#
#             # print('.............after rollaxis 1, 4......................')
#             x_b = np.rollaxis(x, 1, 4)
#             y_b = np.rollaxis(y, 1, 4)
#             # print (x_b.shape, y_b.shape)
#             # print ('...................................')
#
#             # y_b = np.reshape(y_b, (y_b.shape[0], y_b.shape[1], y_b.shape[2], y_b.shape[-1]))
#             print('prepare feed the data to model', x_b.shape, y_b.shape)
#
#             for x, y in zip(x_b, y_b):
#                 if self.task=='no_label':
#                     yield x[np.newaxis,...], y[np.newaxis, ...]
#                 else:
#                     if self.ds == 2 and self.aux:
#                         yield x[np.newaxis, :, :, :, :], self.split_output(y)
#                     elif self.ds == 2 and not self.aux:
#                         yield x[np.newaxis, :, :, :, :], [y[np.newaxis, :, :, :, :], y[np.newaxis, :, :, :, :],
#                                                           y[np.newaxis, :, :, :, :]]
#                     else:
#                         yield x[np.newaxis, :, :, :, :], y[np.newaxis, :, :, :, :]


class TwoScanIterator(Iterator):
    """Class to iterate A and B 3D scans (mhd or nrrd) at the same time."""

    def __init__(self,
                 directory,
                 task='lobe',
                 a_dir_name='ori_ct', b_dir_name='gdth_ct', c_dir_name='aux_gdth',
                 sub_dir='GLUCOL_isotropic1dot5',
                 a_extension='.mhd', b_extension='.nrrd', c_extension='.nrrd',
                 N=-1,
                 ptch_sz=64, ptch_z_sz=16,
                 trgt_sz=256, trgt_z_sz=128,
                 trgt_space=None, trgt_z_space=None,
                 data_argum=True,
                 new_spacing=None,
                 weight_map=10,
                 patches_per_scan=5,

                 ds=2, labels=[],
                 batch_size=1, shuffle=True, seed=None, nb=None,
                                   no_label_dir=None):
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
        - weight_map = value to give to lung borders

        """
        self.task = task

        if self.task == 'no_label':
            self.new_spacing = new_spacing
            if no_label_dir:
                self.a_dir = os.path.join(directory, no_label_dir)
            else:
                self.a_dir = os.path.join(directory, a_dir_name)
            self.a_extension = a_extension
            a_files = set(x.split(a_extension)[0].split(self.a_dir + '/')[-1] for x in
                          sorted(glob.glob(self.a_dir + '/*' + self.a_extension)))
            self.filenames = list(a_files)
            self.aux=False
        else:
            self.sub_dir = sub_dir
            if sub_dir is None:
                self.a_dir = os.path.join(directory, a_dir_name)
                self.b_dir = os.path.join(directory, b_dir_name)

                if c_dir_name is None:
                    self.c_dir = None
                    self.aux = False
                else:
                    self.c_dir = os.path.join(directory, c_dir_name)
                    self.aux = True
            else:
                self.a_dir = os.path.join(directory, a_dir_name, sub_dir)
                self.b_dir = os.path.join(directory, b_dir_name, sub_dir)

                if c_dir_name is None:
                    self.c_dir = None
                    self.aux = False
                else:
                    self.c_dir = os.path.join(directory, c_dir_name, sub_dir)
                    self.aux = True

            self.a_extension = a_extension
            self.b_extension = b_extension
            self.c_extension = c_extension

            a_files = set(x.split(a_extension)[0].split(self.a_dir + '/')[-1] for x in
                          sorted(glob.glob(self.a_dir + '/*' + self.a_extension)))
            b_files = set(x.split(b_extension)[0].split(self.b_dir + '/')[-1] for x in
                          sorted(glob.glob(self.b_dir + '/*' + self.b_extension)))

            # Files inside a and b should have the same name. Images without a pair
            # are discarded.
            self.filenames = list(a_files.intersection(b_files))
            if nb:
                self.filenames = self.filenames[:nb]
                print('from this directory:', self.a_dir)
                print('these files in are used ', self.filenames)

            self.b_fnames = self.filenames
            if c_dir_name is not None:
                self.c_fnames = self.filenames

            self.weight_map = weight_map

            self.ds = ds
            self.labels = labels
            if (self.labels == []):
                self.is_b_categorical = False
            else:
                self.is_b_categorical = True

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

        self.patches_per_scan = patches_per_scan

        # Use only a subset of the files. Good to easily overfit the model
        if N > 0:
            random.shuffle(self.filenames)
            self.filenames = self.filenames[:N]
        self.N = len(self.filenames)

        # sizes
        self.trgt_sz = trgt_sz
        self.trgt_z_sz = trgt_z_sz
        self.trgt_space = trgt_space
        self.trgt_z_space = trgt_z_space
        self.ptch_sz = ptch_sz
        self.ptch_z_sz = ptch_z_sz

        super(TwoScanIterator, self).__init__(len(self.filenames), batch_size, seed,
                                              shuffle)

    def _normal_normalize(self, scan):
        """returns normalized (0 mean 1 variance) scan"""
        scan = (scan - np.mean(scan)) / (np.std(scan))
        return scan

    def load_scan(self, file_name):
        """Load mhd or nrrd 3d scan"""

        if file_name.split('.')[-1] == 'mhd':
            scan, origin, self.spacing = futil.load_itk(file_name)

        elif file_name.split('.')[-1] == 'nrrd':
            scan, origin, self.spacing = futil.load_nrrd(file_name)
        # todo: if extensiion == 'mha' or others
        print((''))

        # if self.task=='no_label' and self.new_spacing is not None:
        #     # rescale no_label data to the same new_spacing as segmentation tasks
        #     time3 = time.time()
        #     print('spacing before rescale no_label scan', self.spacing)
        #
        #     zoom_seq = np.array(self.spacing, dtype='float') / np.array(self.new_spacing, dtype='float')
        #     scan = ndimage.interpolation.zoom(scan, zoom_seq, order=1, prefilter=1)
        #     time4 = time.time()
        #     print('time during rescaleno_label scan:', time4 - time3)

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

                    if self.ptch_sz is not None and self.ptch_sz != self.trgt_sz:
                        a_img, b_img, c_img = random_patch(a, b, c, patch_shape=(self.ptch_z_sz, self.ptch_sz, self.ptch_sz))
                    else:
                        a_img, b_img, c_img = a, b, c

                    batch_a = a_img.copy()
                    batch_b = b_img.copy()
                    batch_c = c_img.copy()

                    A.append(batch_a)
                    B.append(batch_b)
                    C.append(batch_c)

                return [np.array(A), np.array(B), np.array(C)]


            else:
                a, b = self._load_img_pair(j)
                if self.task != 'no_label' and self.is_b_categorical:
                    b = self.one_hot_encode_3D(b, self.labels)
                # apply random affine transformation
                if self.data_argum:
                    a, b = self._random_transform(a, b)

                print('before patching, the shape is ', a.shape)

                A = []
                B = []
                for _ in range(self.patches_per_scan):

                    if self.ptch_sz is not None and self.ptch_sz != self.trgt_sz:
                        a_img, b_img = random_patch(a, b, patch_shape=(self.ptch_z_sz, self.ptch_sz, self.ptch_sz))
                    else:
                        a_img, b_img = a, b


                    batch_a = a_img.copy()
                    batch_b = b_img.copy()

                    A.append(batch_a)
                    B.append(batch_b)

                return [np.array(A), np.array(B)]




    def generator(self):
        x = None
        while 1:
            if self.aux:
                for i in range(10):
                    try:
                        x, y, y_aux = self.next()
                        break
                    except:
                        print('fail to generate this ct for ' + self.task + ', pass it', file=sys.stderr)
                        pass
                if x is None:
                    raise Exception('failed 10 times generation of ct, please check dataset or rescale method, like trgt space or trgt size')

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
                for i in range(10):
                    try:
                        x, y = self.next()
                        break
                    except:
                        print('fail to generate this ct for ' + self.task + ', pass it', file=sys.stderr)
                        pass
                if x is None:
                    raise Exception('failed 10 times generation of ct, please check dataset or rescale method, like trgt space or trgt size')
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
                            yield x[np.newaxis, :, :, :, :], y[np.newaxis, :, :, :, :]
