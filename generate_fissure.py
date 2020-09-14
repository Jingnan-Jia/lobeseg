
import glob
import time
import futils.util as futil
import os

def getFileNames(data_dir, data_nb):
    scan_files = sorted(glob.glob(data_dir + '/' + '*.mhd'))
    if scan_files is None:
        scan_files = sorted(glob.glob(data_dir + '/' + '*.nrrd'))
    if scan_files is None:
        raise Exception('Scan files are None, please check the data directory')
    if isinstance(data_nb, int):
        scan_files = scan_files[:data_nb]
    elif isinstance(data_nb, list):  # number = [3,7]
        scan_files = scan_files[data_nb[0]:data_nb[1]]
    print('Predicted files are:', scan_files)

def generate_batch_fissure(segment, data_dir, preds_dir, data_nb=None, stride=0.25):
    scan_files = getFileNames(data_dir, data_nb)

    t1 = time.time()
    for scan_file in scan_files:
        ct_scan, origin, spacing = futil.load_itk(filename=scan_file)
        fissure =
        # Save the segmentation
        if not os.path.exists(preds_dir):
            os.makedirs(preds_dir)
        futil.save_itk(preds_dir + '/' + scan_file.split('/')[-1], fissure, origin, spacing)
        print('save ct mask at', preds_dir + '/' + scan_file.split('/')[-1])

    t2 = time.time()
    print('finishe a batch of predition, time cost: ', t2 - t1)

