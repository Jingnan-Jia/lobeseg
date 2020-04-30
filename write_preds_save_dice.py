# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:20:10 2020
@author: jjia
"""

from write_batch_preds import write_preds_to_disk
from write_dice import write_dices_to_csv
import segmentor as v_seg
import os
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
'''
task='vessel'
str_names = ['1587041504.5222292_0.00010a_o_0ds2dr1bn1fs16ptsz144ptzsz64']
for str_name in str_names:
    # str_name = '1585000573.7211952_0.00011a_o_0ds2dr1bn1fs16ptsz144ptzsz64'
    model_name = os.path.dirname (os.path.realpath (__file__)) +'/models/' + task + '/' + str_name + 'MODEL.hdf5'
    ptch_z_sz = int(str_name[-2:])
    try:
        ptch_sz = int(str_name[-10:-7])
    except:
        ptch_sz = int(str_name[-9:-7])
    print(ptch_sz, ptch_z_sz)


    for phase in ['valid']:
        if task=='lobe':
            labels = [0, 4, 5, 6, 7, 8]
            data_dir = os.path.dirname (os.path.realpath (__file__)) + '/data/lobe/' + phase + '/ori_ct/GLUCOLD'
            gdth_dir = os.path.dirname (os.path.realpath (__file__)) + '/data/lobe/'  + phase + '/gdth_ct/GLUCOLD'
            preds_dir = os.path.dirname (os.path.realpath (__file__)) + '/data/lobe/preds/' + phase + '/GLUCOLD/' + \
                        model_name.split ('/')[-1][:8]
            # print (preds_dir)
        elif task=='vessel':
            labels = [0, 1]
            data_dir = os.path.dirname (os.path.realpath (__file__)) + '/data/vessel/' + phase + '/ori_ct'
            gdth_dir = os.path.dirname (os.path.realpath (__file__)) + '/data/vessel/'  + phase + '/gdth_ct'
            preds_dir = os.path.dirname (os.path.realpath (__file__)) + '/results/vessel/'+ \
                        model_name.split ('/')[-1][:8]
            # print(preds_dir)
            # preds_dir = '/data/jjia/e2e_new/results/vessel/test_dice'

        segment = v_seg.v_segmentor(batch_size=1,
                                    model=model_name,
                                    ptch_sz=ptch_sz, ptch_z_sz=ptch_z_sz,
                                    trgt_sz=None, trgt_z_sz=None,
                                    trgt_space_list=[0.5, 0.6, 0.6],  # 2.5, 1.4, 1.4 [2.5, 1.4, 1.4]
                                    task=task)

        write_preds_to_disk(segment=segment,
                            data_dir=data_dir,
                            preds_dir=preds_dir,
                            number=5, stride=0.5)


        write_dices_to_csv (labels=labels,
                            gdth_path=gdth_dir,
                            pred_path=preds_dir,
                            csv_file=preds_dir+'/dices.csv')
