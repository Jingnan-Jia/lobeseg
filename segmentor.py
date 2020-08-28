# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:20:10 2017
@author: fferreira
"""

import numpy as np
from futils.util import downsample, correct_shape
from futils.vpatch import deconstruct_patch, reconstruct_patch, deconstruct_patch_gen, reconstruct_patch_gen
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import tensorflow.keras.backend as K
import time
import sys
from futils.util import one_hot_decoding


class v_segmentor(object):
    def __init__(self, batch_size=1, model='.hdf5', ptch_sz=128, ptch_z_sz=64, trgt_sz=None, trgt_z_sz=None,
                 patching=True,
                 trgt_space_list=[], task='lobe', sr=False):
        self.sr = sr
        self.batch_size = batch_size
        self.model = model
        self.ptch_sz = ptch_sz
        self.z_sz = ptch_z_sz
        self.trgt_sz = trgt_sz
        self.trgt_z_sz = trgt_z_sz
        self.trgt_space_list = trgt_space_list  # 2.5, 1.4, 1.4
        self.trgt_sz_list = [self.trgt_z_sz, self.trgt_sz, self.trgt_sz]
        self.task = task
        if task == 'lobe':

            self.labels = [0, 4, 5, 6, 7, 8]
        elif task == 'vessel':
            self.labels = [0, 1]
        else:
            print('please assign the task name for prediction')

        if (self.trgt_sz != self.ptch_sz) and patching and (self.ptch_sz != None and self.ptch_sz != 0):
            self.patching = True
        else:
            self.patching = False

        if type(self.model) is str:  # if model is loaded from a file
            if model.split(".hdf5")[0][-7] == 'h':  # for models saved according to patch metrics
                model_path = model.split(".hdf5")[0][:-12] + '.json'
            else:
                model_path = model.split(".hdf5")[0][:-6] + '.json'

            self.graph1 = tf.Graph()
            with self.graph1.as_default():
                self.session1 = tf.Session()
                with self.session1.as_default():
                    with open(model_path, "r") as json_file:
                        json_model = json_file.read()
                        self.v = model_from_json(json_model)
                        self.v.load_weights((self.model))


        else:  # model is a tf.keras model directly in RAM
            self.graph1 = tf.Graph()
            with self.graph1.as_default():
                self.session1 = tf.Session()
                with self.session1.as_default():
                    self.v = self.model

    def _normalize(self, scan):
        """returns normalized (0 mean 1 variance) scan"""

        scan = (scan - np.mean(scan)) / (np.std(scan))
        return scan

    # def clear_memory(self):
    #     K.clear_session()
    #     return None

    def save(self, model_fpath):
        with self.graph1.as_default():
            with self.session1.as_default():
                self.v.save(model_fpath)  # output a list if aux or deep supervision

    def predict_gen(self, x_patch_gen):
        with self.graph1.as_default():
            with self.session1.as_default():
                for x_patch in x_patch_gen:
                    pred = self.v.predict(x_patch, verbose=0)
                    yield pred

    def predict(self, x, ori_space_list=None, stride=0.25):
        """

        :param x:  shape (z, x, y, 1)
        :param ori_space_list: (z, x, y)
        :param stride:
        :return:
        """
        self.ori_space_list = ori_space_list  # ori_space_list: 0.5, 0.741, 0.741
        # save shape for further upsample
        original_shape = x.shape  # ct_scan.shape: (717,, 512, 512, 1),
        # normalize input
        x_ori = self._normalize(x)  # ct_scan.shape: (717,, 512, 512, 1)
        print('self.trgt_space_list', self.trgt_space_list)

        layers = [l.name for l in self.v.layers]
        if 'input_2' in layers:
            self.mtscale = True
        else:
            self.mtscale = False

        if any(self.trgt_space_list) or any(self.trgt_sz_list):
            if not self.mtscale or self.task == 'lobe':
                x = downsample(x_ori,
                               ori_space=self.ori_space_list, trgt_space=self.trgt_space_list,
                               ori_sz=x_ori.shape, trgt_sz=self.trgt_sz_list,
                               order=1)
                a2 = x_ori if self.mtscale else None
            else:
                x = x_ori
                a2 = downsample(x_ori,
                                ori_space=self.ori_space_list, trgt_space=self.trgt_space_list,
                                ori_sz=x_ori.shape, trgt_sz=self.trgt_sz_list,
                                order=1) if self.mtscale else None

        else:
            x = x_ori
            a2 = None

        if not self.patching:
            raise Exception('patching is not valid! ')

        patch_shape = (self.z_sz, self.ptch_sz, self.ptch_sz)
        x_patch_gen = deconstruct_patch_gen(x, patch_shape=patch_shape, stride=stride, a2=a2)

        pre_gen = self.predict_gen(x_patch_gen)
        if self.task == 'lobe':
            CHN = 6
        elif self.task == 'no_label':
            CHN = 1
        else:
            CHN = 2

        pred = reconstruct_patch_gen(pre_gen, ptch_shape=patch_shape, original_shape=x.shape, stride=stride, chn=CHN)
        # pred has 5 dims, original_shape has 4 dims

        # one hot decoding
        masks = []
        for p in pred:
            masks.append(one_hot_decoding(p, self.labels))
        masks = np.array(masks, dtype='uint8')

        if any(self.trgt_space_list) or any(self.trgt_sz_list):
            if not self.mtscale or self.task == 'lobe':
                print('rescaled to original spacing  ')
                final_pred = downsample(masks,
                                        ori_space=self.trgt_space_list,
                                        trgt_space=self.ori_space_list,
                                        ori_sz=masks.shape,
                                        trgt_sz=original_shape,
                                        order=1,
                                        labels=self.labels)
            else:
                final_pred = masks
        else:
            final_pred = masks

        final_pred = correct_shape(final_pred, original_shape)  # correct the shape mistakes made by sampling
        print('final_pred.shape: ', final_pred.shape)

        return final_pred

#
