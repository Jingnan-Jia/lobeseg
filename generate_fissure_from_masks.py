import glob
import futils.util as futil
import numpy as np
import itk
import SimpleITK as sitk
import matplotlib as plt
import copy
import threading
import queue
import time

def get_fissure(scan, radiusValue=3):
    lobe_4 = copy.deepcopy(scan)
    lobe_4[lobe_4 != 4] = 0
    lobe_4[lobe_4 == 4] = 1
    lobe_5 = copy.deepcopy(scan)
    lobe_5[lobe_5 != 5] = 0
    lobe_5[lobe_5 == 5] = 1
    lobe_6 = copy.deepcopy(scan)
    lobe_6[lobe_6 != 6] = 0
    lobe_6[lobe_6 == 6] = 1
    lobe_7 = copy.deepcopy(scan)
    lobe_7[lobe_7 != 7] = 0
    lobe_7[lobe_7 == 7] = 1
    lobe_8 = copy.deepcopy(scan)
    lobe_8[lobe_8 != 8] = 0
    lobe_8[lobe_8 == 8] = 1
    right_lung = lobe_4 + lobe_5 + lobe_6
    left_lung = lobe_7 + lobe_8

    f_dilate = sitk.BinaryDilateImageFilter()
    f_dilate.SetKernelRadius(radiusValue)
    f_dilate.SetForegroundValue(1)
    f_subtract = sitk.SubtractImageFilter()

    right_lung = sitk.GetImageFromArray(right_lung.astype('int16'))
    right_lung_diated = f_dilate.Execute(right_lung)
    rightlungBorder = f_subtract.Execute(right_lung_diated, right_lung)
    rightlungBorder = sitk.GetArrayFromImage(rightlungBorder)

    left_lung = sitk.GetImageFromArray(left_lung.astype('int16'))
    left_lung_diated = f_dilate.Execute(left_lung)
    leftlungBorder = f_subtract.Execute(left_lung_diated, left_lung)
    leftlungBorder = sitk.GetArrayFromImage(leftlungBorder)

    border = np.zeros((scan.shape))
    for lobe in [lobe_4, lobe_5, lobe_6, lobe_7, lobe_8]:
        lobe = sitk.GetImageFromArray(lobe.astype('int16'))
        lobe_dilated = f_dilate.Execute(lobe)
        lobe_border = f_subtract.Execute(lobe_dilated, lobe)
        lobe_border = sitk.GetArrayFromImage(lobe_border)
        border += lobe_border
    fissure_left = border - leftlungBorder - leftlungBorder
    fissure_right = border - rightlungBorder - rightlungBorder
    fissure = fissure_left + fissure_right
    fissure[fissure <1] = 0
    fissure[fissure >=1] = 1

    return fissure

def writeFissure(ctFpath, fissureFpath, radiusValue=3, Absdir=None):
    scan, origin, spacing = futil.load_itk(ctFpath)
    fissure = get_fissure(scan, radiusValue=radiusValue)
    futil.save_itk(fissureFpath, fissure, origin, spacing)
    print('save ct mask at', fissureFpath)

    # f_dilate = sitk.BinaryDilateImageFilter()
    # f_dilate.SetKernelRadius(radiusValue)
    # f_dilate.SetForegroundValue(1)
    # f_subtract = sitk.SubtractImageFilter()
    #
    # scan_list = []
    # for label in [4,5,6,7,8]:  #exclude background, right lung, 3 lobes
    #     # Threshold the value [label, label+1), results in values inside the range 1, 0 otherwise
    #     # itkimageOneLabel = sitk.BinaryThreshold(itkimage, float(label), float(label+1), 1, 0)
    #     scanOneLabel = copy.deepcopy(scan)
    #     scanOneLabel[scanOneLabel != label] = 0  # note the order of the two lines
    #     scanOneLabel[scanOneLabel==label] = 1
    #     itkimageOneLabel = sitk.GetImageFromArray(scanOneLabel.astype('int16'))
    #     dilatedOneLabel = f_dilate.Execute(itkimageOneLabel)
    #     image = f_subtract.Execute(dilatedOneLabel, itkimageOneLabel)
    #     scanOneLabel = sitk.GetArrayFromImage(image)
    #     scan_list.append(scanOneLabel)
    # scanOneLung = np.array(scan_list) # shape (6/2, 144, 144, 600)
    #
    # for i in range(scanOneLung.shape[0]):
    #     scanOneLung[i] -= lungBorder_array
    #     scanOneLung[i][scanOneLung[i] < 1] = 0  # avoid -1 values
    #     scanOneLung[i][scanOneLung[i] >= 1] = 1
    #
    # scanOneLung = np.rollaxis(scanOneLung, 0, 4)  # shape (144, 144, 600, 6/2)
    # scanOne = np.sum(scanOneLung, axis=-1) # shape (144, 144, 600)
    # scanOne [scanOne < 1] = 0 # note the order of the two lines!
    # scanOne [scanOne >= 1] = 1

def execute_the_function_multi_thread(workers=10, file_list=[], function=None, *args, **kwargs):
    """

    :param workers:
    :param file_list:
    :param function:
    :param args:
    :return:
    """
    def consumer():  # neural network inference needs GPU which can not be computed by multi threads, so the
        # consumer is just the upsampling only.
        while True:
            with threading.Lock():
                ctFpath = None
                if len(file_list):  # if scan_files are empty, then threads should not wait any more
                    print(threading.current_thread().name + " gets the lock, thread id: " + str(
                        threading.get_ident()) + " prepare to execute the function , waiting for the data from queue")
                    ctFpath = file_list.pop()  # wait up to 1 minutes

            if ctFpath is not None:
                t1 = time.time()
                function(*args, **kwargs)
                t3 = time.time()
                print("it costs tis seconds to execute the function of the data " + str(t3 - t1))
            else:
                print(threading.current_thread().name + "scan_files are empty, finish the thread")
                return None

    thd_list = []
    for i in range(workers):
        thd = threading.Thread(target=consumer)
        thd.start()
        thd_list.append(thd)

    for thd in thd_list:
        thd.join()

import numpy as np
import threading
import time




def gntFissure(Absdir, radiusValue=3, workers=10, number=None, qsize=20):
    scan_files = sorted(glob.glob(Absdir + '/*' + '.nrrd'))
    scan_files.extend(sorted(glob.glob(Absdir + '/*' + '.mhd')))
    scan_files.extend(sorted(glob.glob(Absdir + '/*' + '.mha')))
    scan_files = sorted(scan_files)
    if len(scan_files) == 0:
        raise Exception(' predicted files are None, Please check the directories', Absdir)
    if number is not None:
        if type(number) is list:
            scan_files = scan_files[number[0], number[1]]
        else:
            scan_files = scan_files[:number]


    def consumer():  # neural network inference needs GPU which can not be computed by multi threads, so the
        # consumer is just the upsampling only.
        while True:
            with threading.Lock():
                ctFpath = None
                if len(scan_files):  # if scan_files are empty, then threads should not wait any more
                    print(threading.current_thread().name + " gets the lock, thread id: " + str(
                        threading.get_ident()) + " prepare to compute fissure , waiting for the data from queue")
                    ctFpath = scan_files.pop()  # wait up to 1 minutes

            if ctFpath is not None:
                t1 = time.time()
                fissureFpath = Absdir + '/fissure_' + str(radiusValue) + '_' + ctFpath.split('/')[-1]
                writeFissure(ctFpath, fissureFpath, radiusValue, Absdir)
                t3 = time.time()
                print("it costs tis seconds to compute the fissure of the data " + str(t3 - t1))
            else:
                print(threading.current_thread().name + "scan_files are empty, finish the thread")
                return None

    thd_list = []
    for i in range(workers):
        thd = threading.Thread(target=consumer)
        thd.start()
        thd_list.append(thd)

    for thd in thd_list:
        thd.join()



'''
'1599475109_302_lrlb0.0001lrvs1e-05mtscale1netnol-nnlpm0.5nldLUNA16ao1ds2tsp1.4z2.5pps100lbnb17vsnb50nlnb400ptsz144ptzsz96',
                '1599428838_623_lrlb0.0001lrvs1e-05mtscale0netnolpm0.5nldLUNA16ao0ds0tsp1.4z2.5pps100lbnb17vsnb50nlnb400ptsz144ptzsz96',
                '1599479049_663_lrlb0.0001lrvs1e-05mtscale0netnolpm0.5nldLUNA16ao1ds2tsp1.4z2.5pps100lbnb17vsnb50nlnb400ptsz144ptzsz96',
                '1599479049_59_lrlb0.0001lrvs1e-05mtscale1netnolpm0.5nldLUNA16ao1ds2tsp1.4z2.5pps100lbnb17vsnb50nlnb400ptsz144ptzsz96',
                '1599475109_771_lrlb0.0001lrvs1e-05mtscale1netnol-novpm0.5nldLUNA16ao1ds2tsp1.4z2.5pps100lbnb17vsnb50nlnb400ptsz144ptzsz96',
                '1599475109_302_lrlb0.0001lrvs1e-05mtscale1netnol-nnlpm0.5nldLUNA16ao1ds2tsp1.4z2.5pps100lbnb17vsnb50nlnb400ptsz144ptzsz96',
                '''
def main():
    pass

if __name__=="__main__":
    main()