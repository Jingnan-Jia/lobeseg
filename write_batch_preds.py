import glob
import segmentor as v_seg
import time
import futils.util as futil
import os
import numpy as np




def write_preds_to_disk(segment, data_dir, preds_dir, number=None, stride=0.25):


    print('start write_preds_to_disk, this model is loaded from disk')
    scan_files = sorted (glob.glob (data_dir + '/' + '*.mhd'))
    if scan_files is None:
        scan_files = sorted (glob.glob (data_dir + '/' + '*.nrrd'))
    if scan_files is None:
        raise Exception('Scan files are None, please check the data directory')
    if isinstance(number, int):
        scan_files = scan_files[:number]
    elif isinstance(number, list): # number = [3,7]
        scan_files = scan_files[number[0]:number[1]]
    print('predicted files are:', scan_files)

    # segment = v_seg.v_segmentor (batch_size=batch_size, model=model_name,
    #                              ptch_sz=ptch_sz, ptch_z_sz=ptch_z_sz,
    #                              trgt_sz=trgt_sz, trgt_z_sz=trgt_z_sz,
    #                              trgt_space_list=trgt_space_list,
    #                              task=task)
    t1 = time.time ()
    for scan_file in scan_files:

        # ct_scan.shape: (717,, 512, 512), spacing: 0.5, 0.741, 0.741
        ct_scan, origin, spacing, orientation = futil.load_itk (filename=scan_file, get_orientation=True)

        # noise = np.random.randint(low = -90, high = 90, size = ct_scan.shape)
        # ct_scan = ct_scan + noise

        if (orientation[-1] == -1):
            ct_scan = ct_scan[::-1]
        print ('Spacing: ', spacing, 'size', ct_scan.shape)

        # NORMALIZATION
        ct_scan = futil.normalize (ct_scan)


        mask = segment.predict (ct_scan, ori_space_list=spacing, stride=stride) # shape: 717, 512, 512


        # Save the segmentation

        if not os.path.exists (preds_dir):
            os.makedirs (preds_dir)
        futil.save_itk (preds_dir + '/' + scan_file.split ('/')[-1], mask, origin, spacing)
        print ('save ct mask at', preds_dir + '/' + scan_file.split ('/')[-1])


    # segment.clear_memory()
    # del segment

    t2 = time.time()
    print ('finishe a batch of predition, time cost: ', t2 - t1)

