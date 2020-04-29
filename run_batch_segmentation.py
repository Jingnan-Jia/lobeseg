import time
import futils.util as futil
import segmentor as v_seg
import tensorflow.keras.backend as K
import glob
import os
K.set_learning_phase(1)
import numpy as np
from  scipy import ndimage


task = 'vessel'
model_number = '1583108450.566679_0.00010a_o_0ds2dr1bn1fs16ptsz144ptzsz80'
model_extension = 'MODEL.hdf5'
model_file = model_number + model_extension
#LOAD THE MODEL
do_isotropic = False
if do_isotropic:
    segment = v_seg.v_segmentor (batch_size=1, model='models/' + task + '/' + model_file, ptch_sz=144, ptch_z_sz=80,
                                 trgt_sz=256, trgt_z_sz=512)
    #new_spacing=[1.5, 1.5, 1.5]
else:
    segment = v_seg.v_segmentor(batch_size=1, model='models/' + task + '/' + model_file, ptch_sz=144, ptch_z_sz=80,
                                trgt_sz=256, trgt_z_sz=512)

#LOAD THE CT_SCAN

valid_dir = '/exports/lkeb-hpc/jjia/project/mt/data/'+task+'/valid/ori_ct/'
if task=='lobe':
    valid_dir = valid_dir + 'GLUCOLD_isotropic1dot5' # GLUCOLD or GLUCOLD_isotropic0dot7, GLUCOLD_isotropic1dot5,



scan_files = sorted(glob.glob(valid_dir  + '/' + '*.mhd'))
for scan_file in scan_files:
    print('strart segment a ct')
    ct_scan, origin, spacing, orientation = futil.load_itk(filename = scan_file, get_orientation=True)
    if (orientation[-1] == -1):
        ct_scan = ct_scan[::-1]
    print ('Spacing: ', spacing, 'size', ct_scan.shape)

    #NORMALIZATION
    ct_scan = futil.normalize(ct_scan)


    #PREDICT the segmentation
    t1 = time.time()
    lobe_mask = segment.predict(ct_scan, spacing)
    t2 = time.time()
    print ('finishe one, time cost: ', t2-t1)
    
    #upsample back to original shape
# #     print()
#     zoom_seq = np.array(ct_scan.shape,dtype='float')/np.array(lobe_mask.shape,dtype='float')
#     print('lobe_mask.shape', lobe_mask.shape, 'zoom_seq',zoom_seq)
#
#     lobe_mask = ndimage.interpolation.zoom(lobe_mask,zoom_seq,order=0,prefilter=False)
#     print('lobe_mask.shape',lobe_mask.shape)

    #Save the segmentation
    save_dir = 'results/' + task + '/' + model_number[:8]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    futil.save_itk( save_dir + '/' + scan_file.split('/')[-1], lobe_mask, origin, spacing)
    print('save ct mask at', save_dir + '/' + scan_file.split('/')[-1])
