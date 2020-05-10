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
from train_ori_fit_rec_epoch import Mypath

# from train_ori_fit_rec_epoch import Mypath
#1583105196.4919333_0.00010a_o_0.5ds2dr1bn1fs16ptsz144ptzsz80
#1583105073.503702_0.00010a_o_0.5ds2dr1bn1fs16ptsz144ptzsz80
#1583019947.7521048_0.00010a_o_0.5ds2dr1bn1fs16ptsz144ptzsz80
#1579084103.1121035_0.00010a_o_0.5ds2dr1bn1fs16lobesMODEL
#1582578489.061947_0.00011a_o_0.5ds2dr1bn1fs16ptsz144ptzsz80
#1584789581.1021984_1e-051a_o_0.5ds2dr1bn1fs16ptsz144ptzsz64
#1585046531.9177752_0.00011a_o_0.5ds2dr1bn1fs16ptsz144ptzsz64
#1585000573.7211952_0.00011a_o_0ds2dr1bn1fs16ptsz144ptzsz64
#1584924633.4853625_0.00010a_o_0ds2dr1bn1fs16ptsz144ptzsz64
#1585046531.9177752_0.00011a_o_0.5ds2dr1bn1fs16ptsz144ptzsz64
'''
'1585000573.7211952_0.00011a_o_0ds2dr1bn1fs16ptsz144ptzsz64',
             '1584924602.9965076_0.00010a_o_0ds2dr1bn1fs16ptsz144ptzsz64',
             '1584925363.1298258_0.00010a_o_0ds2dr1bn1fs16ptsz96ptzsz64'
             1587041504.5222292_0.00010a_o_0ds2dr1bn1fs16ptsz144ptzsz64MODEL
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
'''
task='vessel'
str_names = ['1588717256_lr0.0001ld0ao0ds2dr1bn1fn16trszNonetrzszNonetrsp0.6trzsp0.3ptch_per_scan500tr_nb5ptsz96ptzsz96',
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

             ]

print(str_names)

mypath = Mypath(task)

for str_name in str_names:
    # str_name = '1585000573.7211952_0.00011a_o_0ds2dr1bn1fs16ptsz144ptzsz64'
    model_name = os.path.dirname (os.path.realpath (__file__)) +'/models/' + task + '/' + str_name + '_tr_best.hdf5'
    # ptch_z_sz = int(str_name.split('ptzsz')[-1])
    # ptch_sz = int(str_name.split('ptsz')[-1].split('ptzsz')[0])

    ptch_z_sz = int(re.findall(r'\d+', str_name.split('ptzsz')[-1])[0])
    ptch_sz = int(re.findall(r'\d+', str_name.split('ptsz')[-1])[0])
    tr_sp = float(re.findall(r'\d+', str_name.split('trsp')[-1])[0])
    tr_z_sp = float(re.findall(r'\d+', str_name.split('trzsp')[-1])[0])

    print('patch_sz', ptch_sz, 'patch_z_size', ptch_z_sz)

    for phase in ['valid']:
        if task=='lobe':
            labels = [0, 4, 5, 6, 7, 8]
        elif task=='vessel':
            labels = [0, 1]

        segment = v_seg.v_segmentor(batch_size=1,
                                    model=model_name,
                                    ptch_sz=ptch_sz, ptch_z_sz=ptch_z_sz,
                                    trgt_sz=None, trgt_z_sz=None,
                                    trgt_space_list=[tr_z_sp, tr_sp, tr_sp],  # 2.5, 1.4, 1.4 [2.5, 1.4, 1.4],[0.5, 0.6, 0.6]
                                    task=task)

        write_preds_to_disk(segment=segment,
                            data_dir=mypath.ori_ct_path(phase),
                            preds_dir=mypath.pred_path(phase),
                            number=3, stride=0.5)


        write_dices_to_csv (labels=labels,
                            gdth_path=mypath.gdth_path(phase),
                            pred_path=mypath.pred_path(phase),
                            csv_file=mypath.dices_location(phase))
