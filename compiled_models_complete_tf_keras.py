
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Activation, Reshape, Lambda, Dropout, Conv3D, BatchNormalization, add, concatenate, RepeatVector, UpSampling3D, PReLU
from tensorflow import multiply
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import sys
import numpy as np
import tensorflow as tf

# I do not know if this line should be put here or in the train.py, so I put it both
tf.keras.mixed_precision.experimental.set_policy('infer')


# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
# implemented by E. Moebel, 06/04/18
def dice_coef_every_class(y_true, y_pred): # not used yet, because one metric must be one scalar from what I experienced
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


def dice_coef(y_true, y_pred): # Not used, because it include background, meaningless
    """
    Returns overall dice coefficient after supressing the background

    TODO : Choose channel(and axis) of background
    """
    y_true_f = K.flatten(Lambda(lambda y_true: y_true[:, :, :, :, 0:])(y_true))
    y_pred_f = K.flatten(Lambda(lambda y_pred: y_pred[:, :, :, :, 0:])(y_pred))

    smooth = 0.0001
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_wo_bg(y_true, y_pred): # not used, it is sum, but we need average
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

def dice_coef_weight_sub(y_true, y_pred): # what is the difference between tf.multiply and .*??
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

    product = multiply([y_true_f, y_pred_f]) # multiply should be import from tf or tf.math

    red_y_true = K.sum(y_true_f, axis=[0, 1, 2, 3]) # shape [None, nb_class]
    red_y_pred = K.sum(y_pred_f, axis=[0, 1, 2, 3])
    red_product = K.sum(product, axis=[0, 1, 2, 3])

    smooth = 0.0001
    dices = (2. * red_product + smooth) / (red_y_true + red_y_pred + smooth)

    ratio= red_y_true / (K.sum(red_y_true) + smooth)
    ratio_y_pred = 1.0 - ratio
    ratio_y_pred = ratio_y_pred / K.sum(ratio_y_pred) # I do not understand it, K.sum(ratio_y_pred) = 1?

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
    ratio_y_pred = K.pow(ratio_y_pred + 0.001, -1.0) # Here can I use 1.0/(ratio + 0.001)?
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
    return dices[0] # return the average dice over the total classes

def dice_1(y_true, y_pred):
    dices = dices_all_class(y_true, y_pred)
    return dices[1] # return the average dice over the total classes

def dice_2(y_true, y_pred):
    dices = dices_all_class(y_true, y_pred)
    return dices[2] # return the average dice over the total classes

def dice_3(y_true, y_pred):
    dices = dices_all_class(y_true, y_pred)
    return dices[3] # return the average dice over the total classes

def dice_4(y_true, y_pred):
    dices = dices_all_class(y_true, y_pred)
    return dices[4] # return the average dice over the total classes

def dice_5(y_true, y_pred):
    dices = dices_all_class(y_true, y_pred)
    return dices[5] # return the average dice over the total classes


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

    return K.mean(dices) # return the average dice over the total classes

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def dice_coef_loss_weight_p(y_true, y_pred):
    return 1-dice_coef_weight_pow(y_true, y_pred)


## Start defining our building blocks

def intro(inputs, nf, nch, bn):
    # inputs = Input((None, None, None, nch))
    conv = Conv3D(nf, 3, padding='same', name='intro_Conv3D')(inputs)
    conv = PReLU(shared_axes=[1, 2, 3], name='intro_PReLU')(conv)
    if bn:
        conv = BatchNormalization(name='intro_bn')(conv)
    # m = Model(inputs=inputs, outputs=c
    return conv


def down_trans(
        inputs, nf, nconvs, bn, dr, ty='v', name='block'):
    # inputs = Input((None, None, None, nch))

    downconv = Conv3D(nf, 2, padding='valid', strides=(2, 2, 2), name=name+'_Conv3D_0')(inputs)
    downconv = PReLU(shared_axes=[1, 2, 3], name=name+'_PReLU_0')(downconv)
    if bn:
        downconv = BatchNormalization( name=name+'_bn_0')(downconv)
    if dr:
        downconv = Dropout(0.5, name=name+'_dr_0')(downconv)

    conv = downconv
    for i in range(nconvs):
        conv = Conv3D(nf, 3, padding='same', name=name+'_Conv3D_'+str(i+1))(conv)
        conv = PReLU(shared_axes=[1, 2, 3], name=name+'_PReLU_'+str(i+1))(conv)
        if bn:
            conv = BatchNormalization( name=name+'_bn_'+str(i+1))(conv)

    if ty == 'v':  # V-Net
        d = add([conv, downconv])
    elif ty == 'u':  # U-Net
        d = conv
    else:
        raise Exception ("please assign the model net_type: 'v' or 'u'.")

    # m = Model(inputs=inputs, outputs=d)

    return d


def up_trans(input1, nf, nconvs, bn, dr, wt_cnt=True, ty='v', input2=None, name='block'):
    '''

    :param nf:
    :param nconvs: number of convolutions
    :param nch:
    :param nch2:
    :param bn:
    :param dr:
    :param wt_cnt: true if with connection
    :param ty: net type: 'u' or 'v'
    :return:
    '''
# #     input1 = Input((None, None, None, nch))
#     if wt_cnt:
#         input2 = Input((None, None, None, nch2))

    upconv = UpSampling3D((2, 2, 2), name=name+'_Upsampling3D')(input1) # todo: if upsampling 3D can be applied?


    if input2 is not None:
        upconv = Conv3D(nf, 2, padding='same', name=name+'_Conv3D_0')(upconv)
    else:
        upconv = Conv3D(nf *2, 2, padding='same', name=name+'_Conv3D_0')(upconv)  # make sure  d = add([conv, merged]) works
    upconv = PReLU(shared_axes=[1, 2, 3], name=name+'_PReLU_0')(upconv)

    if bn:
        upconv = BatchNormalization( name=name+'_bn_0')(upconv)
    if dr:
        upconv = Dropout(0.5, name=name+'_dr_0')(upconv)

    if input2 is not None:
        merged = concatenate([upconv, input2])
    else:
        merged = upconv
    conv = merged
    for i in range(nconvs-1 ):
        conv = Conv3D(nf * 2, 3, padding='same', name=name+'_Conv3D_'+str(i+1))(merged)
        conv = PReLU(shared_axes=[1, 2, 3], name=name+'_PReLU_'+str(i+1))(conv)
        if bn:
            conv = BatchNormalization( name=name+'_vn_'+str(i+1))(conv)

    if ty == 'v':  # V-Net
        d = add([conv, merged])
    elif ty == 'u':  # U-Net
        d = conv
    else:
        raise Exception ("please assign the model net_type: 'v' or 'u'.")

    return d

def connect(a, b):
    if isinstance(type(a), list):
        return a.append(b)
    else:
        return [a, b]

def load_cp_models(model_names,
                   nch=1,
                   lr=0.0001,
                   nf=16,
                   bn=1,
                   dr=1,
                   ds=2,
                   aux=0.5,
                   net_type='v',
                   deeper_vessel=0):

    ## start model
    inputs_data_wt_gt = Input ((None, None, None, nch), name='input_jia')  # input data with ground truth
    in_tr = intro (inputs_data_wt_gt, nf, nch, bn) 

    # down_path
    dwn_tr1 = down_trans (in_tr, nf * 2, 2, bn, dr, ty=net_type, name='block1')
    dwn_tr2 = down_trans (dwn_tr1, nf * 4, 2,  bn, dr, ty=net_type, name='block2')
    dwn_tr3 = down_trans (dwn_tr2,nf * 8, 2,  bn, dr, ty=net_type, name='block3')
    dwn_tr4 = down_trans (dwn_tr3, nf * 16, 2,  bn, dr, ty=net_type, name='block4')

    #######################################################-----------------------#####################################
    # decoder for lung segmentation. up_path for segmentation V-Net, with shot connections
    up_tr4_lung = up_trans (dwn_tr4, nf * 8, 2, bn, dr, ty=net_type, input2=dwn_tr3, name='lung_block5')
    up_tr3_lung = up_trans (up_tr4_lung, nf * 4, 2, bn, dr, ty=net_type, input2=dwn_tr2, name='lung_block6')
    up_tr2_lung = up_trans (up_tr3_lung, nf * 2, 2, bn, dr, ty=net_type, input2=dwn_tr1, name='lung_block7')
    up_tr1_lung = up_trans (up_tr2_lung, nf * 1, 2, bn, dr, ty=net_type, input2=in_tr, name='lung_block8')
    # classification
    lung_out_chn = 2
    res_lung = Conv3D (lung_out_chn, 1, padding='same', name='lung_Conv3D_last') (up_tr1_lung)
    out_lung = Activation ('softmax', name='lung_out_segmentation') (res_lung)

    out_lung = [out_lung]  # convert to list to append other outputs
    if aux:
        # aux_output
        aux_res = Conv3D (2, 1, padding='same', name='lung_aux_Conv3D_last') (up_tr1_lung)
        aux_out = Activation ('softmax', name='lung_aux') (aux_res)
        out_lung.append (aux_out)
    if ds:
        out_lung = [out_lung]
        # deep supervision#1
        deep_1 = UpSampling3D ((2, 2, 2), name='lung_d1_UpSampling3D_0') (up_tr2_lung)
        res = Conv3D (lung_out_chn, 1, padding='same', name='lung_d1_Conv3D_last') (deep_1)
        d_out_1 = Activation ('softmax', name='lung_d1') (res)
        out_lung.append (d_out_1)

        # deep supervision#2
        deep_2 = UpSampling3D ((2, 2, 2), name='lung_d2_UpSampling3D_0') (up_tr3_lung)
        deep_2 = UpSampling3D ((2, 2, 2), name='lung_d2_UpSampling3D_1') (deep_2)
        res = Conv3D (lung_out_chn, 1, padding='same', name='lung_d2_Conv3D_last') (deep_2)
        d_out_2 = Activation ('softmax', name='lung_d2') (res)
        out_lung.append (d_out_2)

    #######################################################-----------------------#####################################
    # decoder for lobe segmentation. up_path for segmentation V-Net, with shot connections
    up_tr4_lobe = up_trans (dwn_tr4, nf * 8, 2, bn, dr, ty=net_type, input2=dwn_tr3, name='lobe_block5')
    up_tr3_lobe = up_trans (up_tr4_lobe, nf * 4, 2, bn, dr, ty=net_type, input2=dwn_tr2, name='lobe_block6')
    up_tr2_lobe = up_trans (up_tr3_lobe, nf * 2, 2, bn, dr, ty=net_type, input2=dwn_tr1, name='lobe_block7')
    up_tr1_lobe = up_trans (up_tr2_lobe, nf * 1, 2, bn, dr, ty=net_type, input2=in_tr, name='lobe_block8')
    # classification
    lobe_out_chn = 6
    res_lobe = Conv3D (lobe_out_chn, 1, padding='same', name='lobe_Conv3D_last') (up_tr1_lobe)
    out_lobe = Activation ('softmax', name='lobe_out_segmentation') (res_lobe)

    out_lobe = [out_lobe]  # convert to list to append other outputs
    if aux:
        # aux_output
        aux_res = Conv3D (2, 1, padding='same', name='lobe_aux_Conv3D_last') (up_tr1_lobe)
        aux_out = Activation ('softmax', name='lobe_aux') (aux_res)
        out_lobe.append (aux_out)
    if ds:
        out_lobe = [out_lobe]
        # deep supervision#1
        deep_1 = UpSampling3D ((2, 2, 2), name='lobe_d1_UpSampling3D_0') (up_tr2_lobe)
        res = Conv3D (lobe_out_chn, 1, padding='same', name='lobe_d1_Conv3D_last') (deep_1)
        d_out_1 = Activation ('softmax', name='lobe_d1') (res)
        out_lobe.append (d_out_1)

        # deep supervision#2
        deep_2 = UpSampling3D ((2, 2, 2), name='lobe_d2_UpSampling3D_0') (up_tr3_lobe)
        deep_2 = UpSampling3D ((2, 2, 2), name='lobe_d2_UpSampling3D_1') (deep_2)
        res = Conv3D (lobe_out_chn, 1, padding='same', name='lobe_d2_Conv3D_last') (deep_2)
        d_out_2 = Activation ('softmax', name='lobe_d2') (res)
        out_lobe.append (d_out_2)


    #######################################################-----------------------#####################################
    # decoder for vessel segmentation. up_path for segmentation V-Net, with shot connections
    up_tr4_vessel = up_trans (dwn_tr4, nf * 8, 2, bn, dr, ty=net_type, input2=dwn_tr3, name='vessel_block5')
    up_tr3_vessel = up_trans (up_tr4_vessel, nf * 4, 2, bn, dr, ty=net_type, input2=dwn_tr2, name='vessel_block6')
    up_tr2_vessel = up_trans (up_tr3_vessel, nf * 2, 2, bn, dr, ty=net_type, input2=dwn_tr1, name='vessel_block7')
    up_tr1_vessel = up_trans (up_tr2_vessel, nf * 1, 2, bn, dr, ty=net_type, input2=in_tr, name='vessel_block8')
    # classification
    vessel_out_chn = 2
    res_vessel = Conv3D (vessel_out_chn, 1, padding='same', name='vessel_Conv3D_last') (up_tr1_vessel)
    out_vessel = Activation ('softmax', name='vessel_out_segmentation') (res_vessel)

    out_vessel = [out_vessel]  # convert to list to append other outputs
    # vessel_aux=0
    if aux:
        # aux_output
        aux_res = Conv3D (2, 1, padding='same', name='vessel_aux_Conv3D_last') (up_tr1_vessel)
        aux_out = Activation ('softmax', name='vessel_aux') (aux_res)
        out_vessel.append (aux_out)
    if ds:
        out_vessel = [out_vessel]
        # deep supervision#1
        deep_1 = UpSampling3D ((2, 2, 2), name='vessel_d1_UpSampling3D_0') (up_tr2_vessel)
        res = Conv3D (vessel_out_chn, 1, padding='same', name='vessel_d1_Conv3D_last') (deep_1)
        d_out_1 = Activation ('softmax', name='vessel_d1') (res)
        out_vessel.append (d_out_1)

        # deep supervision#2
        deep_2 = UpSampling3D ((2, 2, 2), name='vessel_d2_UpSampling3D_0') (up_tr3_vessel)
        deep_2 = UpSampling3D ((2, 2, 2), name='vessel_d2_UpSampling3D_1') (deep_2)
        res = Conv3D (vessel_out_chn, 1, padding='same', name='vessel_d2_Conv3D_last') (deep_2)
        d_out_2 = Activation ('softmax', name='vessel_d2') (res)
        out_vessel.append (d_out_2)

        # ------------------------------------------------------------------------------
    if deeper_vessel:
        inputs_data_wt_gt_deeper = Input((None, None, None, nch), name='input_jia_deeper')  # input data with ground truth
        in_tr_deeper = intro(inputs_data_wt_gt_deeper, nf/2, nch, bn)
        dwn_tr0_deeper = down_trans(in_tr_deeper, nf, 2, bn, dr, ty=net_type, name='block0')

        # down_path
        dwn_tr1_deeper = down_trans(in_tr_deeper, nf * 2, 2, bn, dr, ty=net_type, name='block1')
        dwn_tr2_deeper = down_trans(dwn_tr1_deeper, nf * 4, 2, bn, dr, ty=net_type, name='block2')
        dwn_tr3_deeper = down_trans(dwn_tr2_deeper, nf * 8, 2, bn, dr, ty=net_type, name='block3')
        dwn_tr4_deeper = down_trans(dwn_tr3_deeper, nf * 16, 2, bn, dr, ty=net_type, name='block4')
        # decoder for vessel segmentation. up_path for segmentation V-Net, with shot connections
        up_tr4_vessel_deeper = up_trans(dwn_tr4_deeper, nf * 8, 2, bn, dr, ty=net_type, input2=dwn_tr3_deeper, name='vessel_block5')
        up_tr3_vessel_deeper = up_trans(up_tr4_vessel_deeper, nf * 4, 2, bn, dr, ty=net_type, input2=dwn_tr2_deeper, name='vessel_block6')
        up_tr2_vessel_deeper = up_trans(up_tr3_vessel_deeper, nf * 2, 2, bn, dr, ty=net_type, input2=dwn_tr1_deeper, name='vessel_block7')
        up_tr1_vessel_deeper = up_trans(up_tr2_vessel_deeper, nf * 1, 2, bn, dr, ty=net_type, input2=in_tr_deeper, name='vessel_block8')
        up_tr0_vessel_deeper = up_trans(up_tr2_vessel_deeper, nf, 2, bn, dr, ty=net_type, input2=in_tr_deeper, name='vessel_block9')
        # classification
        vessel_out_chn = 2
        res_vessel_deeper = Conv3D(vessel_out_chn, 1, padding='same', name='vessel_Conv3D_last')(up_tr0_vessel_deeper)
        out_vessel_deeper = Activation('softmax', name='vessel_out_segmentation')(res_vessel_deeper)

        out_vessel_deeper = [out_vessel_deeper]  # convert to list to append other outputs
        # vessel_aux=0
        # if aux:
        #     # aux_output
        #     aux_res_deeper = Conv3D(2, 1, padding='same', name='vessel_aux_Conv3D_last')(up_tr1_vessel_deeper)
        #     aux_out_deeper = Activation('softmax', name='vessel_aux')(aux_res_deeper)
        #     out_vessel_deeper.append(aux_out_deeper)
        # if ds:
        #     out_vessel_deeper = [out_vessel_deeper]
        #     # deep supervision#1
        #     deep_1_deeper = UpSampling3D((2, 2, 2), name='vessel_d1_UpSampling3D_0')(up_tr2_vessel_deeper)
        #     res = Conv3D(vessel_out_chn, 1, padding='same', name='vessel_d1_Conv3D_last')(deep_1_deeper)
        #     d_out_1 = Activation('softmax', name='vessel_d1')(res)
        #     out_vessel_deeper.append(d_out_1)
        #
        #     # deep supervision#2
        #     deep_2 = UpSampling3D((2, 2, 2), name='vessel_d2_UpSampling3D_0')(up_tr3_vessel)
        #     deep_2 = UpSampling3D((2, 2, 2), name='vessel_d2_UpSampling3D_1')(deep_2)
        #     res = Conv3D(vessel_out_chn, 1, padding='same', name='vessel_d2_Conv3D_last')(deep_2)
        #     d_out_2 = Activation('softmax', name='vessel_d2')(res)
        #     out_vessel_deeper.append(d_out_2)

        net_only_vessel_deeper = Model(inputs_data_wt_gt, out_vessel_deeper, name='net_only_vessel_deeper')
        net_only_vessel_deeper.compile(optimizer=optim,
                                loss=dice_coef_loss_weight_p,
                                metrics=metrics_vessel)

    #######################################################-----------------------#####################################
    # decoder for airway segmentation. up_path for segmentation V-Net, with shot connections
    up_tr4_airway = up_trans (dwn_tr4, nf * 8, 2, bn, dr, ty=net_type, input2=dwn_tr3, name='airway_block5')
    up_tr3_airway = up_trans (up_tr4_airway, nf * 4, 2, bn, dr, ty=net_type, input2=dwn_tr2, name='airway_block6')
    up_tr2_airway = up_trans (up_tr3_airway, nf * 2, 2, bn, dr, ty=net_type, input2=dwn_tr1, name='airway_block7')
    up_tr1_airway = up_trans (up_tr2_airway, nf * 1, 2, bn, dr, ty=net_type, input2=in_tr, name='airway_block8')
    # classification
    airway_out_chn = 2
    res_airway = Conv3D (airway_out_chn, 1, padding='same', name='airway_Conv3D_last') (up_tr1_airway)
    out_airway = Activation ('softmax', name='airway_out_segmentation') (res_airway)

    out_airway = [out_airway]  # convert to list to append other outputs
    if aux:
        # aux_output
        aux_res = Conv3D (2, 1, padding='same', name='airway_aux_Conv3D_last') (up_tr1_airway)
        aux_out = Activation ('softmax', name='airway_aux') (aux_res)
        out_airway.append (aux_out)
    if ds:
        out_airway = [out_airway]
        # deep supervision#1
        deep_1 = UpSampling3D ((2, 2, 2), name='airway_d1_UpSampling3D_0') (up_tr2_airway)
        res = Conv3D (airway_out_chn, 1, padding='same', name='airway_d1_Conv3D_last') (deep_1)
        d_out_1 = Activation ('softmax', name='airway_d1') (res)
        out_airway.append (d_out_1)

        # deep supervision#2
        deep_2 = UpSampling3D ((2, 2, 2), name='airway_d2_UpSampling3D_0') (up_tr3_airway)
        deep_2 = UpSampling3D ((2, 2, 2), name='airway_d2_UpSampling3D_1') (deep_2)
        res = Conv3D (airway_out_chn, 1, padding='same', name='airway_d2_Conv3D_last') (deep_2)
        d_out_2 = Activation ('softmax', name='airway_d2') (res)
        out_airway.append (d_out_2)

            

    # # decoder for integrated segmentation. up_path for segmentation V-Net, with shot connections
    # up_tr4_integrated = up_trans(nf * 8, 3, int(dwn_tr4.shape[4]),
    #                              int(dwn_tr3.shape[4]), bn, dr, ty=net_type)([dwn_tr4, dwn_tr3])
    # up_tr3_integrated = up_trans(nf * 4, 3,
    #                              int(up_tr4_integrated.shape[4]),
    #                              int(dwn_tr2.shape[4]), bn, dr, ty=net_type)([up_tr4_integrated, dwn_tr2])
    # up_tr2_integrated = up_trans(nf * 2, 2,
    #                              int(up_tr3_integrated.shape[4]),
    #                              int(dwn_tr1.shape[4]), bn, dr, ty=net_type)([up_tr3_integrated, dwn_tr1])
    # up_tr1_integrated = up_trans(nf * 1, 1,
    #                              int(up_tr2_integrated.shape[4]),
    #                              int(in_tr.shape[4]), bn, dr, ty=net_type)([up_tr2_integrated, in_tr])
    # # classification
    # res_integrated_lobe = Conv3D(lung_out_chn, 1, padding='same')(up_tr1_integrated)
    # out_integrated_lung = Activation('softmax', name='out_integrated_lung_segmentation')(res_integrated_lung)
    #
    # # classification
    # res_integrated_lobe = Conv3D(lobe_out_chn, 1, padding='same')(up_tr1_integrated)
    # out_integrated_lobe = Activation('softmax', name='out_integrated_lobe_segmentation')(res_integrated_lobe)
    #
    # # classification
    # res_integrated_airway = Conv3D(airway_out_chn, 1, padding='same')(up_tr1_integrated)
    # out_integrated_airway = Activation('softmax', name='out_integrated_airway_segmentation')(res_integrated_airway)
    #
    # # classification
    # res_integrated_vessel = Conv3D(vessel_out_chn, 1, padding='same')(up_tr1_integrated)
    # out_integrated_vessel = Activation('softmax', name='out_integrated_vessel_segmentation')(res_integrated_vessel)
    #
    # # classification
    # res_integrated_bronchi = Conv3D(bronchi_out_chn, 1, padding='same')(up_tr1_integrated)
    # out_integrated_bronchi = Activation('softmax', name='out_integrated_bronchi_segmentation')(res_integrated_bronchi)
    #
    # out_integrated_dict = {out_integrated_lung:lung_out_chn,
    #                        out_integrated_lobe:lobe_out_chn,
    #                        out_integrated_airway: airway_out_chn,
    #                       out_integrated_vessel:vessel_out_chn,
    #                       out_integrated_bronchi:bronchi_out_chn}
    # for out, out_chn in out_integrated_dict.items():
    #     if ds:
    #         out = [out]
    #         # deep supervision#1
    #         deep_1 = UpSampling3D((2, 2, 2))(up_tr2_bronchi)
    #         res = Conv3D(out_chn, 1, padding='same')(deep_1)
    #         d_out_1 = Activation('softmax', name='d1')(res)
    #         out.append(d_out_1)
    #
    #         # deep supervision#2
    #         deep_2 = UpSampling3D((2, 2, 2))(up_tr3_bronchi)

    #         deep_2 = UpSampling3D((2, 2, 2))(deep_2)
    #         res = Conv3D(out_chn, 1, padding='same')(deep_2)
    #         d_out_2 = Activation('softmax', name='d2')(res)
    #         out.append(d_out_2)

    # decoder for airway segmentation. up_path for segmentation V-Net, with shot connections
    up_tr4_rec = up_trans (dwn_tr4, nf * 8, 3, bn, dr, ty=net_type, name='rec_block5')
    up_tr3_rec = up_trans (up_tr4_rec, nf * 4, 3, bn, dr, ty=net_type,  name='rec_block6')
    up_tr2_rec = up_trans (up_tr3_rec, nf * 2, 2, bn, dr, ty=net_type,  name='rec_block7')
    up_tr1_rec = up_trans (up_tr2_rec, nf * 1, 2, bn, dr, ty=net_type, name='rec_block8')
    # classification
    rec_out_chn = 1
    out_recon = Conv3D (rec_out_chn, 1, padding='same', name='rec_Conv3D_last') (up_tr1_rec)
    # out_rec = Activation ('softmax', name='rec_out_segmentation') (res_rec)


    loss = [dice_coef_loss_weight_p]
    loss_weights = [1]
    if aux:
        loss.append(dice_coef_loss_weight_p)
        loss_weights = [0.5, 0.5]
        if ds==2:
            loss.append(dice_coef_loss_weight_p)
            loss.append(dice_coef_loss_weight_p)
            loss_weights = [0.375, 0.375, 0.125, 0.125]
    else:
        if ds==2:
            loss.append(dice_coef_loss_weight_p)
            loss.append(dice_coef_loss_weight_p)
            loss_weights = [0.5, 0.25, 0.25]
        else:
            loss = dice_coef_loss_weight_p #right?? todo: check it
            loss_weights = 1

    optim = Adam (lr=lr)

    out_vessel_mt = out_vessel+ [out_recon]
    out_airway_mt = out_airway+ [out_recon]
    out_lobe_mt = out_lobe+ [out_recon]
    out_lung_mt = out_lung+ [out_recon]
    loss_mt = loss + ['mse']
    loss_mt_weights = loss_weights + [0.1]



    # if ds==0:
    #     # configeration and compilization for network in segmentation tasks
    #     loss = [dice_coef_loss_weight_p,'mse' ]
    #     loss_weights = [1, 0.3]
    #     optim = Adam(lr=0.0001)
    #     out_bronchi_mt = [out_bronchi, out_recon]
    #     out_vessel_mt = [out_vessel, out_recon]
    #     out_airway_mt = [out_airway, out_recon]
    #     out_lobe_mt = [out_lobe, out_recon]
    #     out_lung_mt = [out_lung, out_recon]
    # else:
    #     # configeration and compilization for network in segmentation tasks
    #     loss = [dice_coef_loss_weight_p, dice_coef_loss_weight_p, dice_coef_loss_weight_p, 'mse']
    #     loss_weights = [1, 0.5, 0.5, 0.3]
    #     optim = Adam(lr=0.0001)
    #
    #
    #     out_bronchi_mt = [out_bronchi[0], out_bronchi[1], out_bronchi[2], out_recon]
    #     out_vessel_mt = [out_vessel[0], out_vessel[1], out_vessel[2], out_recon]
    #     out_airway_mt = [out_airway[0], out_airway[1], out_airway[2], out_recon]
    #     out_lobe_mt = [out_lobe[0], out_lobe[1], out_lobe[2], out_recon]
    #     out_lung_mt = [out_lung[0], out_lung[1], out_lung[2], out_recon]
    metrics_seg_6_classes = [dice_coef_mean, dice_0, dice_1, dice_2, dice_3, dice_4, dice_5]
    metrics_seg_2_classes = [dice_coef_mean, dice_0, dice_1]

    ###################----------------------------------#########################################
    # compile lobe models
    metrics_lobe = {'lobe_out_segmentation':metrics_seg_6_classes}
    if aux:
        metrics_lobe['lobe_aux'] = metrics_seg_2_classes
    if ds==2:
        metrics_lobe['lobe_d1'] = metrics_seg_6_classes
        metrics_lobe['lobe_d2'] = metrics_seg_6_classes

    net_only_lobe = Model (inputs_data_wt_gt, out_lobe, name='net_only_lobe')
    net_only_lobe.compile (optimizer=optim,
                           loss=dice_coef_loss_weight_p,
                           metrics=metrics_lobe)

    net_lobe_recon = Model (inputs_data_wt_gt, out_lobe_mt, name='net_lobe')
    net_lobe_recon.compile(optimizer=optim,
                           loss=loss_mt,
                           loss_weights=loss_mt_weights,
                           metrics=metrics_lobe.update({'out_recon': 'mse'}))

    ###################----------------------------------#########################################
    # compile vessel models
    metrics_vessel = {'vessel_out_segmentation': metrics_seg_2_classes}
    if aux:
        metrics_vessel['vessel_aux'] = metrics_seg_2_classes
    if ds == 2:
        metrics_vessel['vessel_d1'] = metrics_seg_2_classes
        metrics_vessel['vessel_d2'] = metrics_seg_2_classes

    net_only_vessel = Model (inputs_data_wt_gt, out_vessel, name='net_only_vessel')
    net_only_vessel.compile (optimizer=optim,
                           loss=dice_coef_loss_weight_p,
                           metrics=metrics_vessel)

    net_vessel_recon = Model (inputs_data_wt_gt, out_vessel_mt, name='net_vessel')
    net_vessel_recon.compile (optimizer=optim,
                            loss=loss_mt,
                            loss_weights=loss_mt_weights,
                            metrics=metrics_vessel.update ({'out_recon': 'mse'}))

    ###################----------------------------------#########################################
    # compile airway models
    metrics_airway = {'airway_out_segmentation': metrics_seg_2_classes}
    if aux:
        metrics_airway['airway_aux'] = metrics_seg_2_classes
    if ds == 2:
        metrics_airway['airway_d1'] = metrics_seg_2_classes
        metrics_airway['airway_d2'] = metrics_seg_2_classes

    net_only_airway = Model (inputs_data_wt_gt, out_airway, name='net_only_airway')
    net_only_airway.compile (optimizer=optim,
                           loss=dice_coef_loss_weight_p,
                           metrics=metrics_airway)

    net_airway_recon = Model (inputs_data_wt_gt, out_airway_mt, name='net_airway')
    net_airway_recon.compile (optimizer=optim,
                            loss=loss_mt,
                            loss_weights=loss_mt_weights,
                            metrics=metrics_airway.update ({'out_recon': 'mse'}))
    ###################----------------------------------#########################################
    # compile lung models
    metrics_lung = {'lung_out_segmentation': metrics_seg_2_classes}
    if aux:
        metrics_lung['lung_aux'] = metrics_seg_2_classes
    if ds == 2:
        metrics_lung['lung_d1'] = metrics_seg_2_classes
        metrics_lung['lung_d2'] = metrics_seg_2_classes

    net_only_lung = Model (inputs_data_wt_gt, out_lung, name='net_only_lung')
    net_only_lung.compile (optimizer=optim,
                           loss=dice_coef_loss_weight_p,
                           metrics=metrics_lung)

    net_lung_recon = Model (inputs_data_wt_gt, out_lung_mt, name='net_lung')
    net_lung_recon.compile (optimizer=optim,
                            loss=loss_mt,
                            loss_weights=loss_mt_weights,
                            metrics=metrics_lung.update ({'out_recon': 'mse'}))


    # configeration and compilization for network in recon task
    net_no_label = Model(inputs_data_wt_gt, out_recon, name='net_no_label')
    net_no_label.compile(optimizer= Adam(lr=0.00001), loss='mse', metrics=['mse'])

    # net_itgt_lobe_recon = Model(inputs_data_wt_gt, [out_integrated_lobe, out_recon], name = 'net_integrated_lobe_recon')
    # net_itgt_lobe_recon.compile(optimizer=optim, loss=[dice_coef_loss_weight_p, 'mse'], loss_weights=[1,0.2],
    #                             metrics={'out_integrated_lobe_segmentation': metrics_seg_6_classes,
    #                                      'out_recon': 'mse'})
    # net_itgt_vessel_recon = Model(inputs_data_wt_gt, [out_integrated_vessel, out_recon], name = 'net_integrated_vessel_recon')
    # net_itgt_vessel_recon.compile(optimizer=optim, loss=[dice_coef_loss_weight_p, 'mse'], loss_weights=[1,0.2],
    #                               metrics={'out_integrated_vessel_segmentation': metrics_seg_2_classes,
    #                                        'out_recon': ['mse']})
    #

    # whole network (just for show, not for working)
    # net_whole = Model([inputs_data_wt_gt, task], [out_airway, out_bronchi, out_lobe, out_lung, out_vessel, out_recon])
    # net_whole.compile(optimizer= Adam(lr=0.0001), loss=['mse', 'mse', 'mse', 'mse', 'mse', 'mse'])
    # plot_model(net_whole, to_file='results/model_pictures/net_whole.png', show_layer_names=True, show_shapes=True)

    # plot_model(net_lung, to_file='results/model_pictures/net_lung.png', show_layer_names=True, show_shapes=True)
    # plot_model(net_lobe, to_file='results/model_pictures/net_lobe.png', show_layer_names=True, show_shapes=True)
    # plot_model(net_airway, to_file='results/model_pictures/net_airway.png', show_layer_names=True, show_shapes=True)
    # plot_model(net_bronchi, to_file='results/model_pictures/net_bronchhi.png', show_layer_names=True, show_shapes=True)
    # plot_model(net_vessel, to_file='results/model_pictures/net_vessel.png', show_layer_names=True, show_shapes=True)
    # plot_model(net_no_label, to_file='results/model_pictures/net_no_label.png', show_layer_names=True, show_shapes=True)
    #

    # configeration and compilization for network in segmentation tasks
    # metrics = [['accuracy'], ['accuracy'], ['accuracy'], ['accuracy'], ['accuracy'], ['accuracy']]
    # loss = [dice_coef_loss, dice_coef_loss, dice_coef_loss, dice_coef_loss, dice_coef_loss, 'mse']
    # loss_weights = [1, 1, 1, 1, 1, 1]
    # optim = Adam(lr=0.01)
    # tf.keras.mixed_precision.experimental.set_policy('infer')
    #
    # # the integrated net is for future work
    # net_integrated = Model(inputs_data_wt_gt,
    #                        [out_integrated_airway, out_integrated_bronchi, out_integrated_lobe, out_integrated_lung,
    #                         out_integrated_vessel, out_recon], name='net_bronchi')
    # net_integrated.compile(optimizer=optim, loss=loss, loss_weights=loss_weights)





    # plot_model(net_integrated, to_file='results/model_pictures/net_integrated.png', show_layer_names=True, show_shapes=True)

    if deeper_vessel:
        models_dict = {
            "net_airway": net_airway_recon,
            "net_lung": net_lung_recon,
            "net_lobe": net_lobe_recon,
            "net_vessel": net_vessel_recon,
            "net_no_label": net_no_label,

            "net_only_lobe": net_only_lobe,
            "net_only_vessel": net_only_vessel_deeper,
            "net_only_lung": net_only_lung,
            "net_only_airway": net_only_airway,
            #
            # "net_itgt_lobe_recon": net_itgt_lobe_recon,
            # "net_itgt_vessel_recon": net_itgt_vessel_recon
        }
    else:
        models_dict = {
            "net_airway": net_airway_recon,
            "net_lung": net_lung_recon,
            "net_lobe": net_lobe_recon,
            "net_vessel": net_vessel_recon,
            "net_no_label": net_no_label,

            "net_only_lobe": net_only_lobe,
            "net_only_vessel": net_only_vessel,
            "net_only_lung": net_only_lung,
            "net_only_airway": net_only_airway,
            #
            # "net_itgt_lobe_recon": net_itgt_lobe_recon,
            # "net_itgt_vessel_recon": net_itgt_vessel_recon
        }

    results = []
    for i in model_names:
        net_name = models_dict[i]
        results.append(net_name)


    return results




def main():
    print('this is main function')

if __name__ == '__main__':
    main()


