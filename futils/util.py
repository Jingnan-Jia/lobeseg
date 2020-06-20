# -*- coding: utf-8 -*-
"""
Possible functions on loading, saving, processing itk files.
=============================================================
Created on Tue Apr  4 09:35:14 2017
@author: fferreira and Jingnan
"""
import os
import pickle
import numpy as np
import collections
from skimage import color,transform
import SimpleITK as sitk
from scipy import ndimage
from skimage.io import imsave
import nrrd
import copy
import nibabel as nib
import glob

def get_gdth_pred_names(gdth_path, pred_path):
    gdth_files = sorted(glob.glob(gdth_path + '/*' + '.nrrd'))
    if len(gdth_files) == 0:
        gdth_files = sorted(glob.glob(gdth_path + '/*' + '.mhd'))

    pred_files = sorted(glob.glob(pred_path + '/*' + '.nrrd'))
    if len(pred_files) == 0:
        pred_files = sorted(glob.glob(pred_path + '/*' + '.mhd'))

    if len(gdth_files) == 0:
        raise Exception('ground truth files  are None, Please check the directories', gdth_path)
    if len(pred_files) == 0:
        raise Exception(' predicted files are None, Please check the directories', pred_path)

    if len(pred_files) < len(gdth_files):  # only predict several ct
        gdth_files = gdth_files[:len(pred_files)]

    return gdth_files, pred_files

#%%
def get_UID(file_name):
    print(file_name)
    if(os.path.isfile(file_name)):
        file = open(file_name,'rb')    
        try:        
            data = pickle.load(file)
            # print(data)
            file.close()
            
            return data
        except Exception as inst:
            # print type(inst)     # the exception instance
      
            # print inst           # __str__ allows args to be printed directly

            # print('no pickle here')
            return [],[],[]
    # print 'nop'
    return[],[],[]
   
#%%    
def get_scan(file_name):
    if(os.path.isfile(file_name)):
        file = open(file_name,'rb')    
        data = pickle.load(file)
        file.close()
        return np.rollaxis(data,2)[::-1]
    else:
        return[],[],[]
#%%
def load_itk(filename):
    '''

    :param filename: absolute file path
    :return: ct, origin, spacing, all of them has coordinate (z,y,x) if filename exists. Otherwise, 3 empty list.
    '''
#     print('start load data')
    # Reads the image using SimpleITK
    if(os.path.isfile(filename) ):
        itkimage = sitk.ReadImage(filename)
        
    else:
        print('nonfound:',filename)
        return [],[],[]

    # Convert the image to a  numpy array first ands then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
        
    #ct_scan[ct_scan>4] = 0 #filter trachea (label 5)
    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin()))) #note: after reverseing, origin=(z,y,x)

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing()))) #note: after reverseing,  spacing =(z,y,x)
    orientation = itkimage.GetDirection()
    if (orientation[-1] == -1):
        ct_scan = ct_scan[::-1]

    return ct_scan, origin, spacing
#%%
def save_itk(filename,scan,origin,spacing,dtype = 'int16'):
    """
    Save a array to itk file.

    :param filename: saved file name, a string.
    :param scan: scan array, shape(z, y, x)
    :param origin: origin of itk file, shape (z, y, x)
    :param spacing: spacing of itk file, shape (z, y, x)
    :param dtype: 'int16' default
    :return: None
    """
    stk = sitk.GetImageFromArray(scan.astype(dtype))
    # origin and spacing 's coordinate are (z,y,x). but for image class,
    # the order shuld be (x,y,z), that's why we reverse the order here.
    stk.SetOrigin(origin[::-1])  # numpy array is reversed after convertion from image, but origin and spacing keep unchanged
    stk.SetSpacing(spacing[::-1])

    writer = sitk.ImageFileWriter()
    writer.Execute(stk,filename,True)
#%%

def load_nrrd(filename):
    """
    Load .nrrd file using package nrrd. Can be replaced by function load_itk().

    :param filename: absolute file path
    :return: array of ct, origin and spacing with shape (z, y, x)
    """
    readdata, options = nrrd.read(filename)
    origin = np.array(options['space origin']).astype(float)
    spacing = np.array(options['space directions']).astype(float)
    spacing = np.sum(spacing,axis=0)
    return np.transpose(np.array(readdata).astype(float)),origin[::-1],spacing[::-1] #all of them has coordinate (z,y,x)
                
#%% Save in _nii.gz format
def save_nii(dirname,savefilename,lung_mask):    
    
    array_img = nib.Nifti1Image(lung_mask, affine=None, header=None)
    nib.save(array_img, os.path.join(dirname,savefilename))
    
def save_slice_img(folder,scan,uid):
    print(uid,scan.shape[0])    
    for i,s in enumerate(scan):
        imsave(os.path.join(folder,uid+'sl_'+str(i)+'.png'),s)
#%%
def normalize(image,min_=-1000.0,max_=400.0):
    '''
    Set the values to [0~1].

    :param image: image array
    :param min_: bottom
    :param max_: top
    :return: convert ct scan to 0~1
    '''
    image = (image - min_) / (max_ - min_)
    image[image>1] = 1.
    image[image<0] = 0.

    return image
#%%
def dice(seg,gt):

    
    im1 = np.asarray(seg).astype(float)
    im2 = np.asarray(gt).astype(float)

    
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    
    intersection = (im1*im2).sum()
   
    return 2. * intersection / (im1.sum() + im2.sum())    
#%%
def dice_mc(seg,gt,labels=[]):
    if(labels==[]):   
        labels= np.unique(gt)
    
    dices = np.zeros(len(labels))    
    for i,l in enumerate(labels):
        im1 = seg==l
        im2 = gt ==l
        
        dices[i] = dice(im1,im2)
    
    return dices    
    
   

#%%
def recall(seg,gt):
    
    im1 = np.asarray(seg>0).astype(np.bool)
    im2 = np.asarray(gt>0).astype(np.bool)

    
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2).astype(float)

    
    if(im2.sum()>0):
        return intersection.sum() / ( im2.sum())    
    else:
        return 1.0    
        

def downsample(scan, ori_space=[], trgt_space=[], ori_sz=[], trgt_sz=[], order=1):
    """


    :param scan: shape(z,x,y,chn)
    :param ori_space: shape(z,x,y)
    :param trgt_space: shape(z,x,y)
    :param ori_sz: shape(z,x,y,chn)
    :param trgt_sz: shape(z,x,y)
    :param order:
    :return:
    """
    if len(scan.shape)==3:
        scan=scan[..., np.newaxis]
    if len(ori_sz)==3:
        ori_sz = list(ori_sz)
        ori_sz.append(1)
    print('scan.shape, ori_space, trgt_space, ori_sz, trgt_sz',scan.shape, ori_space, trgt_space, ori_sz, trgt_sz)
    if any(trgt_space):
        print('rescaled to new spacing  ')
        zoom_seq = np.array(ori_space, dtype='float') / np.array(trgt_space, dtype='float')
        zoom_seq = np.append(zoom_seq, 1)
        print('zoom_seq', zoom_seq)
        x = ndimage.interpolation.zoom(scan, zoom_seq, order=order, prefilter=1)  # 143, 271, 271, 1
        print('size after rescale to trgt spacing:', x.shape)
    elif any(trgt_sz):
        print('rescaled to target size')
        zoom_seq = np.array(trgt_sz.append(ori_sz.shape[-1]), dtype='float') / np.array(ori_sz, dtype='float')
        x = ndimage.interpolation.zoom(scan, zoom_seq, order=order, prefilter=order)
        print('size after rescale to new size:', x.shape)  # 64, 144, 144, 1
    else:
        raise Exception('please assign how to rescale')
        print('please assign how to rescale')

    return x


def _one_hot_enc(patch,input_is_grayscale = False,labels=[]):

        labels = np.array(labels)
        N_classes = labels.size
       
             
        
        ptch_ohe = np.zeros((patch.shape[0],patch.shape[1])+(N_classes,))
        for i,l in enumerate(labels):
            
           m = np.where((patch == l).all(axis=2))
            
           new_val = np.zeros(N_classes)
           new_val[i] = 1.
            
            

           ptch_ohe[m] = new_val           
        
        return ptch_ohe
        
def weight_map(label,labels =[0,4,5,6,7,8]):
    
    
    #one hot encoding of label map
    gt_cat = []
    for gt in label:
        gt_cat.append(_one_hot_enc(gt[:,:,np.newaxis],False,labels))    
    gt_cat = np.array(gt_cat)
   
    
    #fill holes and erode to have borders:
    for i in range(1,gt_cat.shape[-1]):
        #gt_cat[:,:,:,i] = ndimage.morphology.binary_fill_holes(gt_cat[:,:,:,i])    
        gt_cat[:,:,:,i] = ndimage.binary_erosion(gt_cat[:,:,:,i],iterations=5)
    
    #create image back from hot encoding
    new_gt= np.zeros_like(label)
    for i,l in enumerate(labels[1::]):
        new_gt[gt_cat[:,:,:,i+1]==1]=l
   
    
    #create weight map
    borders = np.zeros_like(new_gt)    
    borders[(label>0)&(new_gt<1)] = 1.0
    
    weightmap = borders
    #gaussian filter to smooth
    weightmap = ndimage.filters.gaussian_filter(borders.astype('float'),3)
    
   
    
    return weightmap
#%%
def get_fissures(scan):
    lung = np.zeros_like(scan)
    lung[scan>0] = 1
    
    weightmap = weight_map(scan)
    
    weightmap2 = weight_map(lung,labels=[0,1])
    
    weightmap = weightmap-weightmap2
    
    weightmap[weightmap<0] = 0
    
    return weightmap