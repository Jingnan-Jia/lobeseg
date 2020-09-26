# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:20:10 2020
@author: jjia
"""

from write_batch_preds import write_preds_to_disk
from write_dice import write_dices_to_csv
import segmentor as v_seg
import os
import re
from mypath import Mypath
import futils.util as futil
from compute_distance_metrics_and_save import write_all_metrics
import sys
import nvidia_smi
import tensorflow as tf
from tensorflow.keras import backend as K
from generate_fissure_from_masks import gntFissure

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
K.set_session(sess)  # set this TensorFlow session as the default session for Keras

'''
'1585000573.7211952_0.00011a_o_0ds2dr1bn1fs16ptsz144ptzsz64',
             '1584924602.9965076_0.00010a_o_0ds2dr1bn1fs16ptsz144ptzsz64',
             '1584925363.1298258_0.00010a_o_0ds2dr1bn1fs16ptsz96ptzsz64'
             1587041504.5222292_0.00010a_o_0ds2dr1bn1fs16ptsz144ptzsz64
             1587846165.2829812_0.00010a_o_0ds2dr1bn1fs16ptsz144ptzsz64
             1587858645.924413_0.00010a_o_0ds2dr1bn1fs8ptsz144ptzsz96
             1587858294.826981_0.00010a_o_0ds2dr1bn1fs8ptsz144ptzsz96
             1587857822.602289_0.00010a_o_0ds2dr1bn1fs8ptsz144ptzsz96
             1587852304.1056986_0.00010a_o_0ds2dr1bn1fs4ptsz144ptzsz96
             1587852304.1056986_0.00010a_o_0ds2dr1bn1fs4ptsz144ptzsz96
             1587848974.2342067_0.00010a_o_0ds2dr1bn1fs8ptsz144ptzsz96
             1587848927.819794_0.00010a_o_0ds2dr1bn1fs16ptsz144ptzsz96
             1587846165.2829812_0.00010a_o_0ds2dr1bn1fs16ptsz144ptzsz64
             
             '1588287353.8400497_0.00010a_o_0ds2dr1bn1fs8ptsz144ptzsz96'
             '1588287759.0237665_0.00010a_o_0ds2dr1bn1fs8ptsz144ptzsz96'
             '1588287839.3979223_0.00010a_o_0ds2dr1bn1fs8ptsz144ptzsz96'
             '1588288125.8417192_0.00010a_o_0ds2dr1bn1fs8ptsz144ptzsz144'
             '1588288409.620666_0.00010a_o_0ds2dr1bn1fs8ptsz144ptzsz64'
             '1588288526.236716_0.00010a_o_0ds2dr1bn1fs4ptsz144ptzsz64'
             '1588288618.4858303_0.00010a_o_0ds2dr1bn1fs2ptsz144ptzsz64'
             '1588289073.84265_0.00010a_o_0ds2dr1bn1fs4ptsz144ptzsz96'
             '1588289144.719585_0.00010a_o_0ds2dr1bn1fs16ptsz144ptzsz96'
             '1588287353.8400497_0.00010a_o_0ds2dr1bn1fs8ptsz144ptzsz96',
             '1588287759.0237665_0.00010a_o_0ds2dr1bn1fs8ptsz144ptzsz96',
             '1588287839.3979223_0.00010a_o_0ds2dr1bn1fs8ptsz144ptzsz96',
             
             '1588717256_lr0.0001ld0ao0ds2dr1bn1fn16trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb5ptsz96ptzsz96',
             '1588718049_lr0.001ld0ao0ds2dr1bn1fn16trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb50ptsz64ptzsz64',
             '1588717836_lr0.001ld0ao0ds2dr1bn1fn16trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb50ptsz96ptzsz64',
             '1588717764_lr0.001ld0ao0ds2dr1bn1fn16trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb50ptsz96ptzsz144',
             '1588717689_lr0.001ld0ao0ds2dr1bn1fn16trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb50ptsz144ptzsz96',
             '1588717666_lr0.0001ld0ao0ds2dr1bn1fn16trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb20ptsz144ptzsz96',
             '1588717641_lr0.0001ld0ao0ds2dr1bn1fn16trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb10ptsz144ptzsz96',
             '1588717429_lr0.0001ld0ao0ds2dr1bn1fn2trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb5ptsz144ptzsz96',
             '1588717407_lr0.0001ld0ao0ds2dr1bn1fn12trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb5ptsz144ptzsz96',
             '1588717381_lr0.0001ld0ao0ds2dr1bn1fn8trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb5ptsz144ptzsz96',
             '1588717334_lr0.0001ld0ao0ds2dr1bn1fn16trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb5ptsz144ptzsz32',
             '1588717312_lr0.0001ld0ao0ds2dr1bn1fn16trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb5ptsz144ptzsz64',
             '1588717256_lr0.0001ld0ao0ds2dr1bn1fn16trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb5ptsz96ptzsz96',
             '1588717176_lr0.0001ld0ao0ds2dr1bn1fn16trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb5ptsz128ptzsz96',
             '1588716830_lr0.0001ld0ao0ds2dr1bn1fn16trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb5ptsz144ptzsz96',
             '1588710600_lr0.0001ld0ao0ds2dr1bn1fn16trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb5ptsz144ptzsz96'
             
             '1588717256_lr0.0001ld0ao0ds2dr1bn1fn16trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb5ptsz96ptzsz96',
             '1588887205_lr0.0001ld0ao0ds2dr1bn1fn16trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb50ptsz96ptzsz144',
             '1588886816_lr0.0001ld0ao0ds2dr1bn1fn16trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb50ptsz64ptzsz64',
             '1588886682_lr0.0001ld0ao0ds2dr1bn1fn16trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb50ptsz144ptzsz96'
             ''
              '1585000103.6170883_0.00011a_o_0ds2dr1bn1fs16ptsz96ptzsz64',
    '1584925363.1298258_0.00010a_o_0ds2dr1bn1fs16ptsz96ptzsz64'
    
    following 6 models contain lobe task:
         '1592676170_448_lr0.0001ld0m6l0m7l0pm0.5no_label_dirNoneao0ds0dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan100tr_nb18ptsz144ptzsz96', # lobe+vessel 18 cts
         '1592685339_438_lr0.0001ld0m6l0m7l0pm0.5no_label_dirGLUCOLDao0ds0dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan100tr_nb18ptsz144ptzsz96', # lobe+recon 18 cts
         '1592685598_920_lr0.0001ld0m6l0m7l0pm0.5no_label_dirSScao0ds0dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan100tr_nb18ptsz144ptzsz96', # lobe+vessel+recon 18 cts
         '1592689121_655_lr0.0001ld0mtscale1m6l0m7l0pm0.5no_label_dirNoneao0ds0dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan100tr_nb5ptsz144ptzsz96',  # lobe 5 cts
         '1592689233_654_lr0.0001ld0mtscale1m6l0m7l0pm0.5no_label_dirGLUCOLDao0ds0dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan100tr_nb5ptsz144ptzsz96',
         '1592689179_995_lr0.0001ld0mtscale1m6l0m7l0pm0.5no_label_dirNoneao0ds0dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan100tr_nb5ptsz144ptzsz96'
             
    following 6 models contain vessel task:
         '1592689179_711_lr0.0001ld0mtscale1m6l0m7l0pm0.5no_label_dirNoneao0ds0dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan100tr_nb5ptsz144ptzsz96', # lobe+vessel 5 cts
         '1592689254_580_lr0.0001ld0mtscale1m6l0m7l0pm0.5no_label_dirSScao0ds0dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan100tr_nb5ptsz144ptzsz96', # vessel+recon 5 cts
         '1592689158_187_lr0.0001ld0mtscale1m6l0m7l0pm0.5no_label_dirNoneao0ds0dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan100tr_nb5ptsz144ptzsz96', #vessel_only 5 cts
         
         '1592685598_203_lr0.0001ld0m6l0m7l0pm0.5no_label_dirSScao0ds0dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan100tr_nb18ptsz144ptzsz96', #lobe+vessel+recon 18 cts
         '1592676170_464_lr0.0001ld0m6l0m7l0pm0.5no_label_dirNoneao0ds0dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan100tr_nb18ptsz144ptzsz96', #lobe+vessel 18 cts
         '1592685256_247_lr0.0001ld0m6l0m7l0pm0.5no_label_dirSScao0ds0dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan100tr_nb18ptsz144ptzsz96',  # vessel + recon 18 cts
         '1592676155_848_lr0.0001ld0m6l0m7l0pm0.5no_label_dirNoneao0ds0dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan100tr_nb18ptsz144ptzsz96'  #vessel_only, 18 cts
         
    follow 4 models are the continuely trained  models:
    for lobe:
        '1592867691_416_lr1e-05ld1mtscale1m6l0m7l0pm0.0no_label_dirNoneao0ds0dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan100tr_nb18ptsz144ptzsz96', # mtscale, 18 cts
        '1592867085_582_lr1e-05ld1mtscale1m6l0m7l0pm0.0no_label_dirNoneao0ds0dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan100tr_nb5ptsz144ptzsz96', # mtscale, 5 cts
    for vessel:
        '1592867294_862_lr1e-05ld1mtscale1m6l0m7l0pm0.5no_label_dirNoneao0ds0dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan100tr_nb18ptsz144ptzsz96', # mtscale, 18 cts
        '1592867477_914_lr1e-05ld1mtscale1m6l0m7l0pm0.5no_label_dirNoneao0ds0dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan100tr_nb5ptsz144ptzsz96', # mtscale, 5 cts
         
    
    for lobe:
    '1589547509_lr0.0001ld0ao1ds2dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan10tr_nb5ptsz144ptzsz96',
    '1589547535_lr0.0001ld0ao1ds2dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan10tr_nb19ptsz144ptzsz96',
    '1589547545_lr0.0001ld0ao1ds2dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan10tr_nb19ptsz144ptzsz96',
    '1589550315_lr0.0001ld0ao1ds2dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan10tr_nb19ptsz144ptzsz96',
    '1589550811_lr0.0001ld0ao1ds2dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan10tr_nb5ptsz144ptzsz96',
    '1589550817_lr0.0001ld0ao1ds2dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan10tr_nb5ptsz144ptzsz96'
    
    for vessel:
    ['1590413648_783_lr0.0001ld0m6l0m7l0pm0.5no_label_dirSScao0ds0dr1bn1fn16trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb3ptsz144ptzsz96',
     '1590414183_839_lr0.0001ld0m6l0m7l0pm0.5no_label_dirSScao0ds0dr1bn1fn16trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb1ptsz144ptzsz96',
     '1590413385_685_lr0.0001ld0m6l0m7l0pm0.5no_label_dirSScao0ds0dr1bn1fn16trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb3ptsz144ptzsz96',
     '1590413385_284_lr0.0001ld0m6l0m7l0pm0.5no_label_dirSScao0ds0dr1bn1fn16trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb1ptsz144ptzsz96'


    ]
    
    '1595875612_613_lr0.0001ld0mtscale1netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb0ptsz144ptzsz96',
    '1595878407_302_lr0.0001ld0mtscale1netnol-nnlpm0.5nldGLUCOLDao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb5ptsz144ptzsz96',
    '1595875648_331_lr0.0001ld0mtscale1netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb18nlnb0ptsz144ptzsz96',
    '1595878305_786_lr0.0001ld0mtscale1netnol-nnlpm0.5nldGLUCOLDao0ds0bn1fn16tsp1.4z2.5pps100trnb18nlnb18ptsz144ptzsz96',
    
        '1595877571_251_lr0.0001ld0mtscale0netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb0ptsz144ptzsz96',
    '1595878666_42_lr0.0001ld0mtscale0netnol-nnlpm0.5nldGLUCOLDao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb5ptsz144ptzsz96',
    '1595876197_377_lr0.0001ld0mtscale0netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb18nlnb0ptsz144ptzsz96',
    '1595888909_592_lr0.0001ld0mtscale0netnol-nnlpm0.5nldGLUCOLDao0ds0bn1fn16tsp1.4z2.5pps100trnb18nlnb18ptsz144ptzsz96',
    
    '1595934933_835_lr0.0001ld0mtscale1netnol-novpm0.5nldGLUCOLDao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb5ptsz144ptzsz96',
    '1595934933_727_lr0.0001ld0mtscale1netnol-nov-nnlpm0.5nldGLUCOLDao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb5ptsz144ptzsz96',
    '1595934933_562_lr0.0001ld0mtscale1netnol-novpm0.5nldGLUCOLDao0ds0bn1fn16tsp1.4z2.5pps100trnb18nlnb18ptsz144ptzsz96',
    '1595934933_588_lr0.0001ld0mtscale1netnol-nov-nnlpm0.5nldGLUCOLDao0ds0bn1fn16tsp1.4z2.5pps100trnb18nlnb18ptsz144ptzsz96'
    
    
    
    
'1596148486_486_lr0.0001ld1mtscale0netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb0ptsz144ptzsz96',
'1596148055_918_lr0.0001ld1mtscale0netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb0ptsz144ptzsz96',
'1596282280_398_lr0.0001ld1mtscale0netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb0ptsz144ptzsz96',
'1596282736_533_lr0.0001ld1mtscale0netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb18nlnb0ptsz144ptzsz96',
'1596282782_472_lr0.0001ld1mtscale0netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb18nlnb0ptsz144ptzsz96',
'1596282864_156_lr0.0001ld1mtscale1netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb0ptsz144ptzsz96',
'1596282909_196_lr0.0001ld1mtscale1netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb0ptsz144ptzsz96',
'1596282954_794_lr0.0001ld1mtscale1netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb18nlnb0ptsz144ptzsz96',
'1596282987_709_lr0.0001ld1mtscale1netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb18nlnb0ptsz144ptzsz96',

'1596492575_641_lr0.0001ld1mtscale1netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb0ptsz144ptzsz96'.
'1596492575_57_lr0.0001ld1mtscale1netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb0ptsz144ptzsz96'.
'1596492575_540_lr0.0001ld1mtscale1netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb18nlnb0ptsz144ptzsz96'.
'1596492575_956_lr0.0001ld1mtscale1netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb18nlnb0ptsz144ptzsz96'.

'1596750032_237_lr0.0001ld0mtscale0netnovpm0.5nldSScao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb5ptsz144ptzsz96',
'1596750032_417_lr0.0001ld0mtscale0netnov-nnlpm0.5nldSScao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb5ptsz144ptzsz96',
'1596750032_566_lr0.0001ld0mtscale0netnovpm0.5nldSScao0ds0bn1fn16tsp1.4z2.5pps100trnb18nlnb18ptsz144ptzsz96',
'1596750032_130_lr0.0001ld0mtscale0netnov-nnlpm0.5nldSScao0ds0bn1fn16tsp1.4z2.5pps100trnb18nlnb18ptsz144ptzsz96',



'1596926882_451_lr0.0001lrvs0.0001ld1mtscale1netnovpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb0ptsz144ptzsz96',
'1596926882_96_lr0.0001lrvs0.0001ld1mtscale1netnovpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb0ptsz144ptzsz96',
'1596926882_784_lr0.0001lrvs0.0001ld1mtscale1netnovpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb18nlnb0ptsz144ptzsz96',
'1596926882_209_lr0.0001lrvs0.0001ld1mtscale1netnovpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb18nlnb0ptsz144ptzsz96',

'1597066656_855_lr0.0001lrvs0.0001ld1mtscale0netnovpm0.5nldNoneao0ds0bn1fn16tsp0.0z0.0pps100trnb5nlnb0ptsz144ptzsz96',
'1597066656_482_lr0.0001lrvs0.0001ld1mtscale0netnovpm0.5nldNoneao0ds0bn1fn16tsp0.0z0.0pps100trnb5nlnb0ptsz144ptzsz96',
'1597066656_632_lr0.0001lrvs0.0001ld1mtscale0netnovpm0.5nldNoneao0ds0bn1fn16tsp0.0z0.0pps100trnb18nlnb0ptsz144ptzsz96',
'1597066656_951_lr0.0001lrvs0.0001ld1mtscale0netnovpm0.5nldNoneao0ds0bn1fn16tsp0.0z0.0pps100trnb18nlnb0ptsz144ptzsz96',

'1597013371_607_lr0.0001lrvs0.0001ld1mtscale1netnovpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb0ptsz144ptzsz96',
'1597013371_561_lr0.0001lrvs0.0001ld1mtscale1netnovpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb0ptsz144ptzsz96',
'1597013371_569_lr0.0001lrvs0.0001ld1mtscale1netnovpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb18nlnb0ptsz144ptzsz96',
'1597013371_780_lr0.0001lrvs0.0001ld1mtscale1netnovpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb18nlnb0ptsz144ptzsz96',
    
    
   
'1597011428_532_lr0.0001lrvs0.0001ld1mtscale1netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb0ptsz144ptzsz96',
'1597011428_999_lr0.0001lrvs0.0001ld1mtscale1netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb0ptsz144ptzsz96',
'1597011428_927_lr0.0001lrvs0.0001ld1mtscale1netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb18nlnb0ptsz144ptzsz96',
'1597011428_590_lr0.0001lrvs0.0001ld1mtscale1netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb18nlnb0ptsz144ptzsz96',

'1597011627_773_lr0.0001lrvs0.0001ld1mtscale1netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb0ptsz144ptzsz96',
'1597011627_868_lr0.0001lrvs0.0001ld1mtscale1netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb5nlnb0ptsz144ptzsz96',
'1597011627_520_lr0.0001lrvs0.0001ld1mtscale1netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb18nlnb0ptsz144ptzsz96',
'1597011627_353_lr0.0001lrvs0.0001ld1mtscale1netnolpm0.5nldNoneao0ds0bn1fn16tsp1.4z2.5pps100trnb18nlnb0ptsz144ptzsz96',

'1599428838_623_lrlb0.0001lrvs1e-05mtscale0netnolpm0.5nldLUNA16ao0ds0tsp1.4z2.5pps100lbnb17vsnb50nlnb400ptsz144ptzsz96',
             '1599475109_771_lrlb0.0001lrvs1e-05mtscale1netnol-novpm0.5nldLUNA16ao1ds2tsp1.4z2.5pps100lbnb17vsnb50nlnb400ptsz144ptzsz96',
             '1599475109_302_lrlb0.0001lrvs1e-05mtscale1netnol-nnlpm0.5nldLUNA16ao1ds2tsp1.4z2.5pps100lbnb17vsnb50nlnb400ptsz144ptzsz96',


'''
""","""
task='lobe'
sub_dir="LOLA11"

for lung, fissure in zip([1, 0], [0, 1]):
    str_names = ["1600642845_843_lrlb0.0001lrvs1e-05mtscale0netnolpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1599948441_65_lrlb0.0001lrvs1e-05mtscale1netnolpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1599948441_432_lrlb0.0001lrvs1e-05mtscale1netnol-novpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1599948441_216_lrlb0.0001lrvs1e-05mtscale1netnol-nnlpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600476675_493_lrlb0.0001lrvs1e-05mtscale0netnol-nnlpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600476675_894_lrlb0.0001lrvs1e-05mtscale0netnol-novpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600476675_636_lrlb0.0001lrvs1e-05mtscale0netnol-nnl-novpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600642845_843_lrlb0.0001lrvs1e-05mtscale0netnolpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600642845_949_lrlb0.0001lrvs1e-05mtscale0netnol-nnlpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600642845_85_lrlb0.0001lrvs1e-05mtscale0netnol-nnl-novpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600643885_62_lrlb0.0001lrvs1e-05mtscale1netnolpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600643885_972_lrlb0.0001lrvs1e-05mtscale1netnol-nnlpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600643885_770_lrlb0.0001lrvs1e-05mtscale1netnol-novpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600643885_753_lrlb0.0001lrvs1e-05mtscale1netnol-nnl-novpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600645190_366_lrlb0.0001lrvs1e-05mtscale0netnolpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600645190_465_lrlb0.0001lrvs1e-05mtscale0netnol-nnlpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600645190_458_lrlb0.0001lrvs1e-05mtscale0netnol-novpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600645190_537_lrlb0.0001lrvs1e-05mtscale0netnol-nnl-novpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600645872_556_lrlb0.0001lrvs1e-05mtscale0netnolpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600645872_657_lrlb0.0001lrvs1e-05mtscale0netnol-nnlpm0.5nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600645872_801_lrlb0.0001lrvs1e-05mtscale0netnol-nnl-novpm0.5nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600478665_637_lrlb0.0001lrvs1e-05mtscale0netnol-nnl-novpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600478665_204_lrlb0.0001lrvs1e-05mtscale0netnol-nnlpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600478665_586_lrlb0.0001lrvs1e-05mtscale0netnolpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600479252_877_lrlb0.0001lrvs1e-05mtscale1netnol-nnl-novpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600479252_612_lrlb0.0001lrvs1e-05mtscale1netnol-nnlpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96",
                 "1600479252_70_lrlb0.0001lrvs1e-05mtscale1netnolpm0.0nldLUNA16ao0ds0pps100lbnb17vsnb50nlnb400ptsz144ptzsz96"

                 ]
    print(str_names)

    for str_name in str_names:
        mypath = Mypath(task=task, current_time=str_name) # set task=vessel to predict the lobe masks of SSc
        model_name1 = '/data/jjia/new/models/' + task + '/' + str_name + '_valid.hdf5'
        model_name = mypath.best_model_fpath("valid", str_name)

        tr_sp, tr_z_sp = 1.4, 2.5
        tr_sz, tr_z_sz = None, None
        pt_sz, pt_z_sz = 144, 96

        print('patch_sz', pt_sz, 'patch_z_size', pt_z_sz)

        for phase in ['valid']:
            if fissure:
                labels = [0, 1]
                stride = 0.5
            else:

                if task=='lobe':
                    labels = [0, 4, 5, 6, 7, 8]
                    stride = 0.5
                elif task=='vessel':
                    labels = [0, 1]
                    stride = 0.5
            if fissure:
                gntFissure(mypath.pred_path(phase, sub_dir=sub_dir), radiusValue=3, workers=10, qsize=20)
            else:
                segment = v_seg.v_segmentor(batch_size=1,
                                            model=model_name,
                                            ptch_sz=pt_sz, ptch_z_sz=pt_z_sz,
                                            trgt_sz=tr_sz, trgt_z_sz=tr_z_sz,
                                            trgt_space_list=[tr_z_sp, tr_sp, tr_sp],
                                            # 2.5, 1.4, 1.4 [2.5, 1.4, 1.4],[0.5, 0.6, 0.6]
                                            task=task, low_msk=True, attention=False)

                print('stride is', stride)
                write_preds_to_disk(segment=segment,
                                    data_dir=mypath.ori_ct_path(phase, sub_dir=sub_dir),
                                    preds_dir=mypath.pred_path(phase, sub_dir=sub_dir),
                                    number=40,
                                    stride=stride, workers=10, qsize=20)
            # #
            # write_dices_to_csv (step_nb=0,
            #                     labels=labels,
            #                     gdth_path=mypath.gdth_path(phase),
            #                     pred_path=mypath.pred_path(phase),
            #                     csv_file=mypath.dices_fpath(phase))
            #
            write_all_metrics(labels=labels[1:], # exclude background
                                gdth_path=mypath.gdth_path(phase),
                                pred_path=mypath.pred_path(phase),
                                csv_file=mypath.all_metrics_fpath(phase, fissure=fissure),
                              fissure=fissure,
                              lung=lung)

