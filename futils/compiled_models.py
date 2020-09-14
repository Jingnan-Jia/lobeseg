# -*- coding: utf-8 -*-
"""
Compiled models for different tasks.
=============================================================
Created on Tue Apr  4 09:35:14 2017
@author: Jingnan
"""

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Activation, Lambda, Dropout, Conv3D, BatchNormalization, add, concatenate, \
    UpSampling3D, PReLU
from tensorflow import multiply
from tensorflow.keras import backend as K
import tensorflow as tf
import os

# I do not know if this line should be put here or in the train.py, so I put it both
# tf.keras.mixed_precision.experimental.set_policy('infer')
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'


# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
# implemented by E. Moebel, 06/04/18
def dice_coef_every_class(y_true,
                          y_pred):  # not used yet, because one metric must be one scalar from what I experienced
    """
    dice coefficient for every class/label.

    :param y_true: ground truth
    :param y_pred: prediction results
    :return: dice value
    """
    alpha = 0.5
    beta = 0.5

    ones = K.ones(K.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (0, 1, 2, 3))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2, 3)) + beta * K.sum(p1 * g0, (0, 1, 2, 3))

    T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl - T


def dice_coef(y_true, y_pred):  # Not used, because it include background, meaningless
    """

    Returns overall dice coefficient after supressing the background

    TODO : Choose channel(and axis) of background

    :param y_true:
    :param y_pred:
    :return:
    """
    y_true_f = K.flatten(Lambda(lambda y_true: y_true[:, :, :, :, 0:])(y_true))
    y_pred_f = K.flatten(Lambda(lambda y_pred: y_pred[:, :, :, :, 0:])(y_pred))

    smooth = 0.0001
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_wo_bg(y_true, y_pred):  # not used, it is sum, but we need average
    """
    Returns overall dice coefficient after supressing the background

    TODO : Choose channel(and axis) of background
    """
    y_true_f = K.flatten(Lambda(lambda y_true: y_true[:, :, :, :, 1:])(y_true))
    y_pred_f = K.flatten(Lambda(lambda y_pred: y_pred[:, :, :, :, 1:])(y_pred))

    smooth = 0.0001
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_prod(y_true, y_pred):
    """
    completely same with dice_coef but cahieved with different operations

    TODO : Choose channel (and axis) of background
           Choose other merge methods (sum,avg,etc)
    """
    y_true_f = (Lambda(lambda y_true: y_true[:, :, :, :, 0:])(y_true))
    y_pred_f = (Lambda(lambda y_pred: y_pred[:, :, :, :, 0:])(y_pred))

    product = multiply([y_true_f, y_pred_f])

    red_y_true = K.sum(y_true_f, axis=[0, 1, 2, 3])
    red_y_pred = K.sum(y_pred_f, axis=[0, 1, 2, 3])
    red_product = K.sum(product, axis=[0, 1, 2, 3])

    smooth = 0.0001
    dices = (2. * red_product + smooth) / (red_y_true + red_y_pred + smooth)

    return K.prod(dices)


def dice_coef_every_class_old(y_true, y_pred):  # not used
    """
    Returns overall dice coefficient after supressing the background

    TODO : Choose channel(and axis) of background
    """
    y_true_f = (Lambda(lambda y_true: y_true[:, :, :, :, 0:])(y_true))
    y_pred_f = (Lambda(lambda y_pred: y_pred[:, :, :, :, 0:])(y_pred))

    product = multiply([y_true_f, y_pred_f])

    red_y_true = K.sum(y_true_f, axis=[0, 1, 2, 3])
    red_y_pred = K.sum(y_pred_f, axis=[0, 1, 2, 3])
    red_product = K.sum(product, axis=[0, 1, 2, 3])

    smooth = 0.0001
    dices = (2. * red_product + smooth) / (red_y_true + red_y_pred + smooth)

    return dices


def dice_coef_weight_sub(y_true, y_pred):  # what is the difference between tf.multiply and .*??
    """
    Returns the product of dice coefficient for each class
    this dice is designed to increase the ratio of class which include smaller area,
    by the function: ratio_y_pred = 1.0 - ratio_y_pred)
    hope this can make the model pay more attention to the small classes like the right middle lobe
    can be used but not powerful as power weights

    TODO : Choose channel (and axis) of background
           Choose other merge methods (sum,avg,etc)
    """
    y_true_f = (Lambda(lambda y_true: y_true[:, :, :, :, 0:])(y_true))
    y_pred_f = (Lambda(lambda y_pred: y_pred[:, :, :, :, 0:])(y_pred))

    product = multiply([y_true_f, y_pred_f])  # multiply should be import from tf or tf.math

    red_y_true = K.sum(y_true_f, axis=[0, 1, 2, 3])  # shape [None, nb_class]
    red_y_pred = K.sum(y_pred_f, axis=[0, 1, 2, 3])
    red_product = K.sum(product, axis=[0, 1, 2, 3])

    smooth = 0.0001
    dices = (2. * red_product + smooth) / (red_y_true + red_y_pred + smooth)

    ratio = red_y_true / (K.sum(red_y_true) + smooth)
    ratio_y_pred = 1.0 - ratio
    ratio_y_pred = ratio_y_pred / K.sum(ratio_y_pred)  # I do not understand it, K.sum(ratio_y_pred) = 1?

    return K.sum(multiply([dices, ratio_y_pred]))


def dice_coef_weight_pow(y_true, y_pred):
    """
    this dice is designed to increase the ratio of class which include smaller area,
    by the function: K.pow(ratio_y_pred + 0.001, -1.0)
    hope this can make the model pay more attention to the small classes like the right middle lobe

    TODO : Choose channel (and axis) of background
           Choose other merge methods (sum,avg,etc)
    """
    y_true_f = (Lambda(lambda y_true: y_true[:, :, :, :, 0:])(y_true))
    y_pred_f = (Lambda(lambda y_pred: y_pred[:, :, :, :, 0:])(y_pred))

    product = multiply(y_true_f, y_pred_f)

    red_y_true = K.sum(y_true_f, axis=[0, 1, 2, 3])
    red_y_pred = K.sum(y_pred_f, axis=[0, 1, 2, 3])
    red_product = K.sum(product, axis=[0, 1, 2, 3])

    smooth = 0.0001
    dices = (2. * red_product + smooth) / (red_y_true + red_y_pred + smooth)

    ratio_y_pred = red_y_true / (K.sum(red_y_true) + smooth)
    ratio_y_pred = K.pow(ratio_y_pred + 0.001, -1.0)  # Here can I use 1.0/(ratio + 0.001)?
    ratio_y_pred = ratio_y_pred / K.sum(ratio_y_pred)

    return K.sum(multiply(dices, ratio_y_pred))


def dices_all_class(y_true, y_pred):
    '''

    :param y_true:
    :param y_pred:
    :return: dices list
    '''
    y_true_f = (Lambda(lambda y_true: y_true[:, :, :, :, 0:])(y_true))
    y_pred_f = (Lambda(lambda y_pred: y_pred[:, :, :, :, 0:])(y_pred))

    product = multiply(y_true_f, y_pred_f)

    red_y_true = K.sum(y_true_f, axis=[0, 1, 2, 3])
    red_y_pred = K.sum(y_pred_f, axis=[0, 1, 2, 3])
    red_product = K.sum(product, axis=[0, 1, 2, 3])

    smooth = 0.0001
    dices = (2. * red_product + smooth) / (red_y_true + red_y_pred + smooth)

    return dices


def dice_0(y_true, y_pred):
    dices = dices_all_class(y_true, y_pred)
    return dices[0]  # return the average dice over the total classes


def dice_1(y_true, y_pred):
    dices = dices_all_class(y_true, y_pred)
    return dices[1]  # return the average dice over the total classes


def dice_2(y_true, y_pred):
    dices = dices_all_class(y_true, y_pred)
    return dices[2]  # return the average dice over the total classes


def dice_3(y_true, y_pred):
    dices = dices_all_class(y_true, y_pred)
    return dices[3]  # return the average dice over the total classes


def dice_4(y_true, y_pred):
    dices = dices_all_class(y_true, y_pred)
    return dices[4]  # return the average dice over the total classes


def dice_5(y_true, y_pred):
    dices = dices_all_class(y_true, y_pred)
    return dices[5]  # return the average dice over the total classes


def dice_coef_mean(y_true, y_pred):
    """
    Returns the product of dice coefficient for each class
    it assumes channel 0 as background

    TODO : Choose channel (and axis) of background
           Choose other merge methods (sum,avg,etc)
    """

    y_true_f = (Lambda(lambda y_true: y_true[:, :, :, :, 1:])(y_true))
    y_pred_f = (Lambda(lambda y_pred: y_pred[:, :, :, :, 1:])(y_pred))

    product = multiply(y_true_f, y_pred_f)

    red_y_true = K.sum(y_true_f, axis=[0, 1, 2, 3])
    red_y_pred = K.sum(y_pred_f, axis=[0, 1, 2, 3])
    red_product = K.sum(product, axis=[0, 1, 2, 3])

    smooth = 0.0001
    dices = (2. * red_product + smooth) / (red_y_true + red_y_pred + smooth)
    mean = K.sum(dices)

    return K.mean(dices)  # return the average dice over the total classes


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_coef_loss_weight_p(y_true, y_pred):
    return 1 - dice_coef_weight_pow(y_true, y_pred)


def intro(inputs, nf, bn, name='1'):
    """
    introduction convolution + PReLU (+ batch_normalization).

    :param inputs: data from last output
    :param nf: number of filters (or feature maps)
    :param bn: if batch normalization
    :return: convolution results
    """
    # inputs = Input((None, None, None, nch))
    conv = Conv3D(nf, 3, padding='same', name='intro_Conv3D' + name)(inputs)
    conv = PReLU(shared_axes=[1, 2, 3], name='intro_PReLU' + name)(conv)
    if bn:
        conv = BatchNormalization(name='intro_bn' + name)(conv)
    # m = Model(inputs=inputs, outputs=c
    return conv


def down_trans(inputs, nf, nconvs, bn, dr, ty='v', name='block'):
    # inputs = Input((None, None, None, nch))

    downconv = Conv3D(nf, 2, padding='valid', strides=(2, 2, 2), name=name + '_Conv3D_0')(inputs)
    downconv = PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_0')(downconv)
    if bn:
        downconv = BatchNormalization(name=name + '_bn_0')(downconv)
    if dr:
        downconv = Dropout(0.5, name=name + '_dr_0')(downconv)

    conv = downconv
    for i in range(nconvs):
        conv = Conv3D(nf, 3, padding='same', name=name + '_Conv3D_' + str(i + 1))(conv)
        conv = PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_' + str(i + 1))(conv)
        if bn:
            conv = BatchNormalization(name=name + '_bn_' + str(i + 1))(conv)

    if ty == 'v':  # V-Net
        d = add([conv, downconv])
    elif ty == 'u':  # U-Net
        d = conv
    else:
        raise Exception("please assign the model net_type: 'v' or 'u'.")

    # m = Model(inputs=inputs, outputs=d)

    return d


def up_trans(input1, nf, nconvs, bn, dr, ty='v', input2=None, name='block'):
    """
    up transform convolution.

    :param input1: output from last layer
    :param nf: number of filters (or feature maps)
    :param nconvs: number of convolution operations in each level apart from basic downconvolutin or upconvolution.
    :param bn: if batch normalization
    :param dr: if dropout
    :param ty: model type, 'v' means V-Net, 'u' means 'U-Net'
    :param input2: second input, necessary for merge results from short connection
    :param name: block name
    :return: convolution results
    """

    upconv = UpSampling3D((2, 2, 2), name=name + '_Upsampling3D')(input1)  #
    upconv = Conv3D(nf, 2, padding='same', name=name + '_Conv3D_0')(upconv)
    upconv = PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_0')(upconv)
    if bn:
        upconv = BatchNormalization(name=name + '_bn_0')(upconv)
    if dr:
        upconv = Dropout(0.5, name=name + '_dr_0')(upconv)

    if input2 is not None:
        conv = concatenate([upconv, input2])
    else:
        conv = upconv
    for i in range(nconvs):
        conv = Conv3D(nf, 3, padding='same', name=name + '_Conv3D_' + str(i + 1))(conv)
        conv = PReLU(shared_axes=[1, 2, 3], name=name + '_PReLU_' + str(i + 1))(conv)
        if bn:
            conv = BatchNormalization(name=name + '_bn_' + str(i + 1))(conv)

    if ty == 'v':  # V-Net
        d = add([conv, upconv])
    elif ty == 'u':  # U-Net
        d = conv
    else:
        raise Exception("please assign the model net_type: 'v' or 'u'.")
    return d

def get_loss_weights_optim(ao, ds, lr, mot, task=None):
    loss = [dice_coef_loss_weight_p]
    loss_weights = [1]
    if mot:
        if task == "no_label":
            loss = ['mse']
        else:
            loss.append(dice_coef_loss_weight_p)
        loss_weights = [0.5, 0.5]
    if ao:
        loss.append(dice_coef_loss_weight_p)
        loss_weights = [0.5, 0.5]
        if ds == 2:
            loss.append(dice_coef_loss_weight_p)
            loss.append(dice_coef_loss_weight_p)
            loss_weights = [0.375, 0.375, 0.125, 0.125]
    elif ds == 2:
        loss.append(dice_coef_loss_weight_p)
        loss.append(dice_coef_loss_weight_p)
        loss_weights = [0.5, 0.25, 0.25]

    loss_itgt_recon = loss + ['mse']
    loss_itgt_recon_weights = loss_weights + [0.1 * loss_weights[0]]

    optim_tmp = Adam(lr)
    optim = tf.train.experimental.enable_mixed_precision_graph_rewrite(optim_tmp)

    return loss, loss_weights, loss_itgt_recon, loss_itgt_recon_weights, optim


def load_cp_models(model_names, args):
    """
    load compiled models.
    """

    nch = 1
    nf = args.feature_number
    bn = args.batch_norm
    dr = args.dropout
    net_type = args.u_v
    mtscale = args.mtscale
    attention = args.attention

    ## start model
    input_data = Input((None, None, None, nch), name='input')  # input data
    in_tr = intro(input_data, nf, bn)

    if mtscale:
        input_data2 = Input((None, None, None, nch), name='input_2')  # input data 2 with a different scale
        in_tr2 = intro(input_data2, nf, bn, name='2')

        in_tr = concatenate([in_tr, in_tr2])
        input_data = [input_data, input_data2]

    # down_path
    dwn_tr1 = down_trans(in_tr, nf * 2, 2, bn, dr, ty=net_type, name='block1')
    dwn_tr2 = down_trans(dwn_tr1, nf * 4, 2, bn, dr, ty=net_type, name='block2')
    dwn_tr3 = down_trans(dwn_tr2, nf * 8, 2, bn, dr, ty=net_type, name='block3')
    dwn_tr4 = down_trans(dwn_tr3, nf * 16, 2, bn, dr, ty=net_type, name='block4')



    #######################################################-----------------------#####################################
    # decoder for lobe segmentation. up_path for segmentation V-Net, with shot connections
    up_tr4_lobe = up_trans(dwn_tr4, nf * 8, 2, bn, dr, ty=net_type, input2=dwn_tr3, name='lobe_block5')
    up_tr3_lobe = up_trans(up_tr4_lobe, nf * 4, 2, bn, dr, ty=net_type, input2=dwn_tr2, name='lobe_block6')
    up_tr2_lobe = up_trans(up_tr3_lobe, nf * 2, 2, bn, dr, ty=net_type, input2=dwn_tr1, name='lobe_block7')
    up_tr1_lobe = up_trans(up_tr2_lobe, nf * 1, 1, bn, dr, ty=net_type, input2=in_tr, name='lobe_block8')
    # classification
    lobe_out_chn = 6
    res_lobe = Conv3D(lobe_out_chn, 1, padding='same', name='lobe_Conv3D_last')(up_tr1_lobe)
    out_lobe = Activation('softmax', name='lobe_out_segmentation')(res_lobe)

    out_lobe = [out_lobe]  # convert to list to append other outputs
    if args.mot_lb:
        res_lobe2 = Conv3D(lobe_out_chn, 1, padding='same', name='lobe_Conv3D_last2')(up_tr1_lobe)
        out_lobe2 = Activation('softmax', name='lobe_out_segmentation2')(res_lobe2)
        out_lobe.append(out_lobe2)
    if args.ao_lb:
        # aux_output
        aux_res = Conv3D(2, 1, padding='same', name='lobe_aux_Conv3D_last')(up_tr1_lobe)
        aux_out = Activation('softmax', name='lobe_aux')(aux_res)
        out_lobe.append(aux_out)
    if args.ds_lb:
        out_lobe = [out_lobe]
        # deep supervision#1
        deep_1 = UpSampling3D((2, 2, 2), name='lobe_d1_UpSampling3D_0')(up_tr2_lobe)
        res = Conv3D(lobe_out_chn, 1, padding='same', name='lobe_d1_Conv3D_last')(deep_1)
        d_out_1 = Activation('softmax', name='lobe_d1')(res)
        out_lobe.append(d_out_1)

        # deep supervision#2
        deep_2 = UpSampling3D((2, 2, 2), name='lobe_d2_UpSampling3D_0')(up_tr3_lobe)
        deep_2 = UpSampling3D((2, 2, 2), name='lobe_d2_UpSampling3D_1')(deep_2)
        res = Conv3D(lobe_out_chn, 1, padding='same', name='lobe_d2_Conv3D_last')(deep_2)
        d_out_2 = Activation('softmax', name='lobe_d2')(res)
        out_lobe.append(d_out_2)

        #######################################################-----------------------#####################################
    # decoder for lung segmentation. up_path for segmentation V-Net, with shot connections
    up_tr4_lung = up_trans(dwn_tr4, nf * 8, 2, bn, dr, ty=net_type, input2=dwn_tr3, name='lung_block5')
    up_tr3_lung = up_trans(up_tr4_lung, nf * 4, 2, bn, dr, ty=net_type, input2=dwn_tr2, name='lung_block6')
    up_tr2_lung = up_trans(up_tr3_lung, nf * 2, 2, bn, dr, ty=net_type, input2=dwn_tr1, name='lung_block7')
    up_tr1_lung = up_trans(up_tr2_lung, nf * 1, 1, bn, dr, ty=net_type, input2=in_tr,
                           name='lung_block8')  # note filters number
    # classification
    lung_out_chn = 2
    if attention:
        res_lung = Conv3D(lobe_out_chn, 1, padding='same', name='lung_Conv3D_last')(up_tr1_lung)
        out_lung = Activation('softmax', name='lung_out_segmentation')(res_lung)
    else:
        res_lung = Conv3D(lung_out_chn, 1, padding='same', name='lung_Conv3D_last')(up_tr1_lung)
        out_lung = Activation('softmax', name='lung_out_segmentation')(res_lung)

    out_lung = [out_lung]  # convert to list to append other outputs
    if args.mot_lu:
        res_lung2 = Conv3D(lung_out_chn, 1, padding='same', name='lung_Conv3D_last2')(up_tr1_lung)
        out_lung2 = Activation('softmax', name='lung_out_segmentation2')(res_lung2)
        out_lung.append(out_lung2)
    if args.ao_lu:
        # aux_output
        aux_res = Conv3D(2, 1, padding='same', name='lung_aux_Conv3D_last')(up_tr1_lung)
        aux_out = Activation('softmax', name='lung_aux')(aux_res)
        out_lung.append(aux_out)
    if args.ds_lu:
        out_lung = [out_lung]
        # deep supervision#1
        deep_1 = UpSampling3D((2, 2, 2), name='lung_d1_UpSampling3D_0')(up_tr2_lung)
        res = Conv3D(lung_out_chn, 1, padding='same', name='lung_d1_Conv3D_last')(deep_1)
        d_out_1 = Activation('softmax', name='lung_d1')(res)
        out_lung.append(d_out_1)

        # deep supervision#2
        deep_2 = UpSampling3D((2, 2, 2), name='lung_d2_UpSampling3D_0')(up_tr3_lung)
        deep_2 = UpSampling3D((2, 2, 2), name='lung_d2_UpSampling3D_1')(deep_2)
        res = Conv3D(lung_out_chn, 1, padding='same', name='lung_d2_Conv3D_last')(deep_2)
        d_out_2 = Activation('softmax', name='lung_d2')(res)
        out_lung.append(d_out_2)

    #######################################################-----------------------#####################################
    # decoder for vessel segmentation. up_path for segmentation V-Net, with shot connections
    up_tr4_vessel = up_trans(dwn_tr4, nf * 8, 2, bn, dr, ty=net_type, input2=dwn_tr3, name='vessel_block5')
    up_tr3_vessel = up_trans(up_tr4_vessel, nf * 4, 2, bn, dr, ty=net_type, input2=dwn_tr2, name='vessel_block6')
    up_tr2_vessel = up_trans(up_tr3_vessel, nf * 2, 2, bn, dr, ty=net_type, input2=dwn_tr1, name='vessel_block7')
    up_tr1_vessel = up_trans(up_tr2_vessel, nf * 1, 1, bn, dr, ty=net_type, input2=in_tr, name='vessel_block8')
    # classification
    vessel_out_chn = 2
    # out_vessel_attention = out_vessel_tmp[1] * out_lobe_tmp
    if attention:
        res_vessel = Conv3D(lobe_out_chn, 1, padding='same', name='vessel_Conv3D_last')(up_tr1_vessel)
        out_vessel = Activation('softmax', name='vessel_out_segmentation')(res_vessel)
    else:
        res_vessel = Conv3D(vessel_out_chn, 1, padding='same', name='vessel_Conv3D_last')(up_tr1_vessel)
        out_vessel = Activation('softmax', name='vessel_out_segmentation')(res_vessel)

    out_vessel = [out_vessel]  # convert to list to append other outputs
    # vessel_aux=0
    if args.mot_vs:
        res_vessel2 = Conv3D(vessel_out_chn, 1, padding='same', name='vessel_Conv3D_last2')(up_tr1_vessel)
        out_vessel2 = Activation('softmax', name='vessel_out_segmentation2')(res_vessel2)
        out_vessel.append(out_vessel2)
    if args.ao_vs:
        # aux_output
        aux_res = Conv3D(2, 1, padding='same', name='vessel_aux_Conv3D_last')(up_tr1_vessel)
        aux_out = Activation('softmax', name='vessel_aux')(aux_res)
        out_vessel.append(aux_out)
    if args.ds_vs:
        out_vessel = [out_vessel]
        # deep supervision#1
        deep_1 = UpSampling3D((2, 2, 2), name='vessel_d1_UpSampling3D_0')(up_tr2_vessel)
        res = Conv3D(vessel_out_chn, 1, padding='same', name='vessel_d1_Conv3D_last')(deep_1)
        d_out_1 = Activation('softmax', name='vessel_d1')(res)
        out_vessel.append(d_out_1)

        # deep supervision#2
        deep_2 = UpSampling3D((2, 2, 2), name='vessel_d2_UpSampling3D_0')(up_tr3_vessel)
        deep_2 = UpSampling3D((2, 2, 2), name='vessel_d2_UpSampling3D_1')(deep_2)
        res = Conv3D(vessel_out_chn, 1, padding='same', name='vessel_d2_Conv3D_last')(deep_2)
        d_out_2 = Activation('softmax', name='vessel_d2')(res)
        out_vessel.append(d_out_2)

    #######################################################-----------------------#####################################
    # decoder for airway segmentation. up_path for segmentation V-Net, with shot connections
    up_tr4_airway = up_trans(dwn_tr4, nf * 8, 2, bn, dr, ty=net_type, input2=dwn_tr3, name='airway_block5')
    up_tr3_airway = up_trans(up_tr4_airway, nf * 4, 2, bn, dr, ty=net_type, input2=dwn_tr2, name='airway_block6')
    up_tr2_airway = up_trans(up_tr3_airway, nf * 2, 2, bn, dr, ty=net_type, input2=dwn_tr1, name='airway_block7')
    up_tr1_airway = up_trans(up_tr2_airway, nf * 1, 1, bn, dr, ty=net_type, input2=in_tr, name='airway_block8')
    # classification
    airway_out_chn = 2
    if attention:
        res_airway = Conv3D(lobe_out_chn, 1, padding='same', name='airway_Conv3D_last')(up_tr1_airway)
        out_airway = Activation('softmax', name='airway_out_segmentation')(res_airway)
    else:
        res_airway = Conv3D(airway_out_chn, 1, padding='same', name='airway_Conv3D_last')(up_tr1_airway)
        out_airway = Activation('softmax', name='airway_out_segmentation')(res_airway)


    out_airway = [out_airway]  # convert to list to append other outputs
    if args.mot_aw:
        res_airway2 = Conv3D(airway_out_chn, 1, padding='same', name='airway_Conv3D_last2')(up_tr1_airway)
        out_airway2 = Activation('softmax', name='airway_out_segmentation2')(res_airway2)
        out_airway.append(out_airway2)
    if args.ao_aw:
        # aux_output
        aux_res = Conv3D(2, 1, padding='same', name='airway_aux_Conv3D_last')(up_tr1_airway)
        aux_out = Activation('softmax', name='airway_aux')(aux_res)
        out_airway.append(aux_out)
    if args.ds_aw:
        out_airway = [out_airway]
        # deep supervision#1
        deep_1 = UpSampling3D((2, 2, 2), name='airway_d1_UpSampling3D_0')(up_tr2_airway)
        res = Conv3D(airway_out_chn, 1, padding='same', name='airway_d1_Conv3D_last')(deep_1)
        d_out_1 = Activation('softmax', name='airway_d1')(res)
        out_airway.append(d_out_1)

        # deep supervision#2
        deep_2 = UpSampling3D((2, 2, 2), name='airway_d2_UpSampling3D_0')(up_tr3_airway)
        deep_2 = UpSampling3D((2, 2, 2), name='airway_d2_UpSampling3D_1')(deep_2)
        res = Conv3D(airway_out_chn, 1, padding='same', name='airway_d2_Conv3D_last')(deep_2)
        d_out_2 = Activation('softmax', name='airway_d2')(res)
        out_airway.append(d_out_2)

    # decoder for reconstruction
    up_tr4_rec = up_trans(dwn_tr4, nf * 8, 2, bn, dr, ty=net_type, name='rec_block5')
    up_tr3_rec = up_trans(up_tr4_rec, nf * 4, 2, bn, dr, ty=net_type, name='rec_block6')
    up_tr2_rec = up_trans(up_tr3_rec, nf * 2, 2, bn, dr, ty=net_type, name='rec_block7')
    up_tr1_rec = up_trans(up_tr2_rec, nf * 1, 1, bn, dr, ty=net_type, name='rec_block8')
    # classification
    rec_out_chn = 1
    if attention:
        out_recon = Conv3D(lobe_out_chn, 1, padding='same', name='out_recon')(up_tr1_rec)
    else:
        out_recon = Conv3D(rec_out_chn, 1, padding='same', name='out_recon')(up_tr1_rec)

    if args.mot_rc:
        out_recon2 = Conv3D(rec_out_chn, 1, padding='same', name='out_recon2')(up_tr1_rec)
        out_recon = [out_recon, out_recon2]

    # out_rec = Activation ('softmax', name='rec_out_segmentation') (res_rec) # no activation for reconstruction

    out_itgt_vessel_recon = out_vessel + [out_recon]
    out_itgt_airway_recon = out_airway + [out_recon]
    out_itgt_lobe_recon = out_lobe + [out_recon]
    out_itgt_lung_recon = out_lung + [out_recon]

    metrics_seg_6_classes = [dice_coef_mean, dice_0, dice_1, dice_2, dice_3, dice_4, dice_5]
    metrics_seg_2_classes = [dice_coef_mean, dice_0, dice_1]

    ###################----------------------------------#########################################
    # compile lobe models
    metrics_lobe = {'lobe_out_segmentation': metrics_seg_6_classes}
    if args.mot_lb:
        metrics_lobe['lobe_out_segmentation2'] = metrics_seg_6_classes
    if args.ao_lb:
        metrics_lobe['lobe_aux'] = metrics_seg_2_classes
    if args.ds_lb == 2:
        metrics_lobe['lobe_d1'] = metrics_seg_6_classes
        metrics_lobe['lobe_d2'] = metrics_seg_6_classes

    loss, loss_weights, loss_itgt_recon, loss_itgt_recon_weights, optim = get_loss_weights_optim(args.ao_lb, args.ds_lb,
                                                                                                 args.lr_lb,
                                                                                                 args.mot_lb)
    net_only_lobe = Model(input_data, out_lobe, name='net_only_lobe')
    net_only_lobe.compile(optimizer=optim,
                          loss=loss,
                          loss_weights=loss_weights,
                          metrics=metrics_lobe)

    net_itgt_lobe_recon = Model(input_data, out_itgt_lobe_recon, name='net_itgt_lobe_recon')
    net_itgt_lobe_recon.compile(optimizer=optim,
                                loss=loss_itgt_recon,
                                loss_weights=loss_itgt_recon_weights,
                                metrics=metrics_lobe.update({'out_recon': 'mse'}))

    ###################----------------------------------#########################################
    # compile vessel models
    if attention:
        metrics_vessel = {'vessel_out_segmentation': metrics_seg_6_classes}
    else:
        metrics_vessel = {'vessel_out_segmentation': metrics_seg_2_classes}
    if args.mot_vs:
        metrics_vessel['vessel_out_segmentation2'] = metrics_seg_2_classes
    if args.ao_vs:
        metrics_vessel['vessel_aux'] = metrics_seg_2_classes
    if args.ds_vs == 2:
        metrics_vessel['vessel_d1'] = metrics_seg_2_classes
        metrics_vessel['vessel_d2'] = metrics_seg_2_classes

    loss, loss_weights, loss_itgt_recon, loss_itgt_recon_weights, optim = get_loss_weights_optim(args.ao_vs, args.ds_vs,
                                                                                                 args.lr_vs,
                                                                                                 args.mot_vs)
    net_only_vessel = Model(input_data, out_vessel, name='net_only_vessel')
    net_only_vessel.compile(optimizer=optim,
                            loss=loss,
                            loss_weights=loss_weights,
                            metrics=metrics_vessel)

    net_itgt_vessel_recon = Model(input_data, out_itgt_vessel_recon, name='net_itgt_vessel_recon')
    net_itgt_vessel_recon.compile(optimizer=optim,
                                  loss=loss_itgt_recon,
                                  loss_weights=loss_itgt_recon_weights,
                                  metrics=metrics_vessel.update({'out_recon': 'mse'}))

    ###################----------------------------------#########################################
    # compile airway models
    if attention:
        metrics_airway = {'airway_out_segmentation': metrics_seg_2_classes}
    else:
        metrics_airway = {'airway_out_segmentation': metrics_seg_2_classes}

    if args.mot_aw:
        metrics_airway['airway_out_segmentation2'] = metrics_seg_2_classes
    if args.ao_aw:
        metrics_airway['airway_aux'] = metrics_seg_2_classes
    if args.ds_aw == 2:
        metrics_airway['airway_d1'] = metrics_seg_2_classes
        metrics_airway['airway_d2'] = metrics_seg_2_classes

    loss, loss_weights, loss_itgt_recon, loss_itgt_recon_weights, optim = get_loss_weights_optim(args.ao_aw, args.ds_aw,
                                                                                                 args.lr_aw,
                                                                                                 args.mot_aw)
    net_only_airway = Model(input_data, out_airway, name='net_only_airway')
    net_only_airway.compile(optimizer=optim,
                            loss=loss,
                            loss_weights=loss_weights,
                            metrics=metrics_airway)

    net_itgt_airway_recon = Model(input_data, out_itgt_airway_recon, name='net_itgt_airway_recon')
    net_itgt_airway_recon.compile(optimizer=optim,
                                  loss=loss_itgt_recon,
                                  loss_weights=loss_itgt_recon_weights,
                                  metrics=metrics_airway.update({'out_recon': 'mse'}))
    ###################----------------------------------#########################################
    # compile lung models
    if attention:
        metrics_lung = {'lung_out_segmentation': metrics_seg_6_classes}
    else:
        metrics_lung = {'lung_out_segmentation': metrics_seg_2_classes}
    if args.mot_lu:
        metrics_lung['lung_out_segmentation2'] = metrics_seg_2_classes
    if args.ao_lu:
        metrics_lung['lung_aux'] = metrics_seg_2_classes
    if args.ds_lu == 2:
        metrics_lung['lung_d1'] = metrics_seg_2_classes
        metrics_lung['lung_d2'] = metrics_seg_2_classes

    loss, loss_weights, loss_itgt_recon, loss_itgt_recon_weights, optim = get_loss_weights_optim(args.ao_lu, args.ds_lu,
                                                                                                 args.lr_lu,
                                                                                                 args.mot_lu)
    net_only_lung = Model(input_data, out_lung, name='net_only_lung')
    net_only_lung.compile(optimizer=optim,
                          loss=loss,
                          loss_weights=loss_weights,
                          metrics=metrics_lung)

    net_itgt_lung_recon = Model(input_data, out_itgt_lung_recon, name='net_itgt_lung_recon')
    net_itgt_lung_recon.compile(optimizer=optim,
                                loss=loss_itgt_recon,
                                loss_weights=loss_itgt_recon_weights,
                                metrics=metrics_lung.update({'out_recon': 'mse'}))

    # configeration and compilization for network in recon task
    optim, loss_weights, _, __, optim = get_loss_weights_optim(args.ao_rc, args.ds_rc, args.lr_rc, args.mot_rc,
                                                               task='no_label')
    net_no_label = Model(input_data, out_recon, name='net_no_label')
    net_no_label.compile(optimizer=optim, loss='mse', loss_weights=loss_weights, metrics=['mse'])

    models_dict = {
        "net_itgt_lu_rc": net_itgt_lung_recon,
        "net_itgt_aw_rc": net_itgt_airway_recon,
        "net_itgt_lb_rc": net_itgt_lobe_recon,
        "net_itgt_vs_rc": net_itgt_vessel_recon,

        "net_no_label": net_no_label,

        "net_only_lobe": net_only_lobe,
        "net_only_vessel": net_only_vessel,
        "net_only_lung": net_only_lung,
        "net_only_airway": net_only_airway,
    }

    return list(map(models_dict.get, model_names))

def main():
    print('this is main function')


if __name__ == '__main__':
    main()
