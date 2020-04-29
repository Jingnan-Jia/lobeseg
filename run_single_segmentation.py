import time
import futils.util as futil
import segmentor as v_seg
import keras.backend as K

K.set_learning_phase(1)

isotropic=False

#LOAD THE CT_SCAN
scan_file = '1.3.6.1.4.1.14519.5.2.1.6279.6001.413896555982844732694353377538.mhd'
ct_scan, origin, spacing, orientation = futil.load_itk(filename = '/exports/lkeb-hpc/jjia/project/mt/data/lobe/valid/ori_ct/luna16/'+scan_file, get_orientation=True)
if (orientation[-1] == -1):
    ct_scan = ct_scan[::-1]
print ('Origem: ')
print (origin)
print ('Spacing: ')
print (spacing)
print ('Orientation: ')
print (orientation)

#NORMALIZATION
ct_scan = futil.normalize(ct_scan)

if isotropic:
    #LOAD THE MODEL
    segment = v_seg.v_segmentor(batch_size=1, model='models/lobe/1583104581.8410566_0.00010a_o_0.5ds2dr1bn1fs16ptsz144ptzsz80MODEL.hdf5',
                                ptch_sz=128, ptch_z_sz=64, patching=True, spacing=spacing, new_spacing=[1.5, 1.5, 1.5])

else:
    # LOAD THE MODEL
    segment = v_seg.v_segmentor (batch_size=1,
                                 model='models/lobe/1583104581.8410566_0.00010a_o_0.5ds2dr1bn1fs16ptsz144ptzsz80MODEL.hdf5',
                                 ptch_sz=128, ptch_z_sz=64, patching=True)

#PREDICT the segmentation
t1 = time.time()
lobe_mask = segment.predict(ct_scan)
t2 = time.time()
print (t2-t1)


#Save the segmentation
futil.save_itk('results/lobe/'+scan_file, lobe_mask, origin, spacing)
