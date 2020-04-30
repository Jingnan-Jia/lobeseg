# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:20:10 2020
@author: jjia
"""

import tensorflow as tf
import tensorflow.keras
import os
from tensorflow.keras.models import load_model # load model architecture and weights
from tensorflow.keras.utils import plot_model

import time
import futils.util as futil
import segmentor as v_seg
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Input
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import os

K.set_learning_phase(0)

#--------------parameters setting----------------------
# task = 'lobe'
# fixed_space = True
# file = 'GLUCOLD_patients_28'
# scan_file = file +'.mhd'
# sub_dir = 'GLUCOLD'
# model_number = '1584789581.1021984_1e-051a_o_0.5ds2dr1bn1fs16ptsz144ptzsz64'
# layer_name = None # if layer_name is none, then plot all the feature maps
# start_x_in_ori = 190
# start_y_in_ori = 40
# center_z_in_ori = 652


#--------------parameters setting----------------------

# #--------------parameters setting----------------------
task = 'vessel'
fixed_space = None
fixed_size = None
file = 'SSc_patient_51'
scan_file = file +'.mhd'
model_number = '1587858645.924413_0.00010a_o_0ds2dr1bn1fs8ptsz144ptzsz96'
layer_name = None # if layer_name is none, then plot all the feature maps
start_x_in_ori = 200
start_y_in_ori = 80
center_z_in_ori = 700
#1587858645.924413_0.00010a_o_0ds2dr1bn1fs8ptsz144ptzsz96
#
# #--------------parameters setting----------------------

ptch_z_sz = int(model_number[-2:])
try:
    ptch_sz = int(model_number[-10:-7])
except:
    ptch_sz = int(model_number[-9:-7])
print('ptch_sz, {}, ptch_z_sz, {}'.format(ptch_sz, ptch_z_sz))

if fixed_space:
    trgt_space = [0.3, 0.5, 0.5]
elif fixed_size:
    trgt_size = [128, 256, 256]
else:
    trgt_size = None

#LOAD THE MODEL
segment = v_seg.v_segmentor(batch_size=1, model='models/' + task + '/' + model_number+'MODEL.hdf5', patching=True)
print('finished first model loading')
model = segment.v

plot_model(model, show_shapes=True, to_file='lobemodel.png')
#
# K.set_learning_phase(0)
# task = 'lobe'
# model_number = '1583105196.4919333_0.00010a_o_0.5ds2dr1bn1fs16ptsz144ptzsz80'
#
# #LOAD THE MODEL
# segment_ = v_seg.v_segmentor(batch_size=1, model='models/' + task + '/' + model_number+'MODEL.hdf5', patching=False)
# print('finished first model loading')
# model_ = segment_.v

# plot_model(model, show_shapes=True, to_file='/exports/lkeb-hpc/jjia/project/e2e_new/amodel.png')

# summarize filter shapes
layer_names = []
# all_layers = []
for layer in model.layers:
    # check for convolutional layer
    # print(layer.name)
    # all_layers.append(layer.name)
    if 'input' in layer.name:
        layer_names.append(layer.name)
        print(layer.name)
    if 'Conv' in layer.name or 'conv' in layer.name:
        filters = layer.get_weights()[0]
        print(layer.name, filters.shape, layer.output.shape)
        layer_names.append(layer.name)
    elif 'out' in layer.name:
        layer_names.append(layer.name)
        print(layer.name)
# print(layer_names)
# model.summary()

def getLayerIndexByName(model, layername):
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx


# scan_file = 'SSc_patient_77.mhd'
# sub_dir = 'GLUCOLD_isotropic1dot5'



if task=='vessel':
    file_name = os.path.dirname (os.path.realpath (__file__)) + '/data/'+ task + '/valid/ori_ct/'  + scan_file
    file_name_gdth = os.path.dirname (os.path.realpath (__file__)) + '/data/' + task + '/valid/gdth_ct/'  + scan_file
elif task=='lobe':
    file_name = os.path.dirname (os.path.realpath (__file__)) + '/data/' + task + '/valid/ori_ct/' + sub_dir + '/' + scan_file
print(file_name)
ct_scan, origin, spacing, orientation = futil.load_itk(filename = file_name, get_orientation=True)
scan_gdth, _, _ = futil.load_itk(filename = file_name_gdth, get_orientation=False)

if (orientation[-1] == -1):
    ct_scan = ct_scan[::-1]
print ('Spacing: ', spacing)

scan = (ct_scan - np.mean(ct_scan))/(np.std(ct_scan))



if fixed_space:
    zoom_seq = np.array (spacing, dtype='float') / np.array (trgt_space, dtype='float') # order is correct
    print('zoom_seq', zoom_seq)

    scan = ndimage.interpolation.zoom(scan, zoom_seq, order=1, prefilter=1)
elif fixed_size:
    zoom_seq = np.array (trgt_size, dtype='float') / np.array (scan.shape, dtype='float') # order is correct
    print('zoom_seq', zoom_seq)

    scan = ndimage.interpolation.zoom(scan, zoom_seq, order=1, prefilter=1)
else:
    zoom_seq = [1, 1, 1]

scan = scan[np.newaxis,:,:,:,np.newaxis]
scan = np.rollaxis(scan,1, 4)
print(scan.shape)



center_z = int(center_z_in_ori * zoom_seq[0])

start_x = int(start_x_in_ori * zoom_seq[1])
end_x = start_x + ptch_sz

start_y = int(start_y_in_ori * zoom_seq[2])
end_y = start_y + ptch_sz

start_z = int(center_z - ptch_z_sz / 2)
end_z = start_z+ptch_z_sz

local_z_position = 0.5

if end_x>scan.shape[1]:
    end_x = scan.shape[1]
    start_x = end_x - ptch_sz

if end_y>scan.shape[2]:
    end_y = scan.shape[2]
    start_y = end_y - ptch_sz

if end_z>scan.shape[3]:
    end_z = scan.shape[3]
    start_z = end_z - ptch_z_sz
    local_z_position = (center_z - start_z)/ptch_z_sz
    print('the index of z is not at the center, its position is ', local_z_position)



# test = start_z+ptch_z_sz
# print(start_x+ptch_sz, start_y+ptch_sz, start_z+ptch_z_sz)

scan = scan[:, start_x:start_x+ptch_sz, start_y:start_y+ptch_sz, start_z:start_z+ptch_z_sz, : ]
print(scan.shape)

scan_gdth = scan_gdth[np.newaxis,:,:,:,np.newaxis]
scan_gdth = np.rollaxis(scan_gdth,1, 4)
scan_gdth = scan_gdth[:, start_x:start_x+ptch_sz, start_y:start_y+ptch_sz, start_z:start_z+ptch_z_sz, : ]




def plot_feature_maps(feature_maps, file, layer_name):
    number = feature_maps.shape[-1]
    plt.ion()
    if number > 256:
        print('feature maps number are greater than 128, can not plot')
    else:
        if number ==1:
            row, col = (1, 1)
        if number == 2:
            row, col = (1, 2)
        elif number == 6:
            row, col = (3, 2)
        elif number == 8:
            row, col = (4, 2)
        elif number == 16:
            row, col = (4, 4)
        elif number == 32:
            row, col = (8, 4)
        elif number == 64:
            row, col = (8, 8)
        elif number == 128:
            row, col = (16, 8)
        elif number == 256:
            row, col = (16, 16)
        else:
            print('feature maps number is {} instead of 2,6,16,32,64,128, can not plot'.format(number))
        ix = 1
        f = plt.figure(figsize=(col * 4, row * 4))
        z = int(local_z_position * feature_maps.shape[-2])
        print(number)
        for _ in range(row):
            for _ in range(col):
                # specify subplot and turn of axis
                ax = f.add_subplot(row, col, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale

                plt.imshow(feature_maps[0, :, :, z, ix - 1], cmap='bwr')
                ix += 1
        # show the figure
        f.tight_layout()

        # plt.show()
        if not os.path.exists('results/' + task + '/feature_maps/' + model_number[:8] ):
            os.makedirs('results/' + task + '/feature_maps/' + model_number[:8] )

        saved_path =  'results/' + task + '/feature_maps/' + model_number[:8] + '/' + file + '_'+ layer_name + '_nb_' + str(number) + \
                      'x_' + str(start_x_in_ori) + 'y_' + str(start_y_in_ori) + 'z_' + str(center_z_in_ori) + '.png'
        plt.savefig( saved_path)
        plt.close()
        print('save png to '+saved_path)


# layer_name = 'vessel_block8_Conv3D_2'
# idx = getLayerIndexByName(model, layer_name)
# print('idx of features', idx)
# t = Model(inputs=model.layers[0].input, outputs=model.layers[idx].output)
#
# feature_maps = t.predict(scan)
# plot_feature_maps(number=feature_maps.shape[-1])
#
#
# feature_maps =  np.where(feature_maps > 0.5, 1, 0)
# plot_feature_maps(number=feature_maps.shape[-1])
if layer_name:
    idx = getLayerIndexByName(model, layer_name)
    print('idx of features', idx)
    t = Model(inputs=model.layers[0].input, outputs=model.layers[idx].output)

    feature_maps = t.predict(scan)
    plot_feature_maps(feature_maps=feature_maps, file=file, layer_name=layer_name)
else:
    print('length of layer_names', len(layer_names))
    for layer_name in layer_names:
        idx = getLayerIndexByName(model, layer_name)
        print('idx of features', idx)
        if 'input' in layer_name:
            feature_maps = scan
        else:
            t = Model(inputs=model.layers[0].input, outputs=model.layers[idx].output)
            feature_maps = t.predict(scan)
        plot_feature_maps(feature_maps=feature_maps, file=file, layer_name=layer_name)

    plot_feature_maps(feature_maps=scan_gdth, file=file, layer_name='gdth')