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
import csv
def get_gdth_pred_names(gdth_path, pred_path, fissure=False):
    if fissure:
        gdth_files = sorted(glob.glob(gdth_path + '/fissure*' + '.nrrd'))
        if len(gdth_files) == 0:
            gdth_files = sorted(glob.glob(gdth_path + '/fissure*' + '.mhd'))

        pred_files = sorted(glob.glob(pred_path + '/fissure*' + '.nrrd'))
        if len(pred_files) == 0:
            pred_files = sorted(glob.glob(pred_path + '/fissure*' + '.mhd'))

        if len(gdth_files) == 0:
            raise Exception('ground truth files  are None, Please check the directories', gdth_path)
        if len(pred_files) == 0:
            raise Exception(' predicted files are None, Please check the directories', pred_path)
    else:
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

class Upsample_smooth():

    def __init__(self, trgt_sz=[]):
        """
        this function is to upsample a bi-value 3D mask with smooth edge which is adapted from nearest neighbor.

        :param mask: shape(z,x,y)
        :param trgt_sz: shape(z,x,y)
        :return:
        """
        self.trgt_sz = np.array(trgt_sz)

    def up_mask_0_pad(self, mask):
        self.old_sz = mask.shape
        self.ratio_array = np.array(self.trgt_sz) / np.array(self.old_sz)
        new_pst = [list(np.ceil(np.arange(sz) * ratio).astype('int')) for sz, ratio in zip(self.old_sz, self.ratio_array)]
        out = np.zeros(self.trgt_sz)  # out image with all zeros
        out[np.ix_(*new_pst)] = 1 # * can parse the list to 3 sub lists
        out2 = out[:, 200, :]
        return out

    def mask_smooth(self, mask):
        new_mask = copy.deepcopy(mask)
        k_sz1 = list(self.ratio_array.astype('int')+1) # +1 means that our kernel can cover two nearest voxels
        k_sz2 = [k_sz1[0]+1, k_sz1[1], k_sz1[2]] # z axis is always float,
        k_sz3 = [k_sz1[0], k_sz1[1]+1, k_sz1[2]+1] # x, y axis may be  float,
        k_sz4 = [k_sz1[0]+1, k_sz1[1]+1, k_sz1[2]+1] # z, x, y axis may be float

        k_list = [np.ones(k_sz) for k_sz in [k_sz1, k_sz2, k_sz3, k_sz4]]
        for kernel in k_list:
            sz = kernel.shape
            for i,j,k in zip(range(self.old_sz[0]-sz[0]), range(self.old_sz[1]-sz[1]), range(self.old_sz[2]-sz[2])):
                conved_pst = [list(m + np.arange(kernel.shape[n])) for m, n in zip([i, j, k], [0, 1, 2])]
                conv = kernel * mask[np.ix_(*conved_pst)] # * can parse the list to 3 sub lists
                new_conv = kernel * new_mask[np.ix_(*conved_pst)] # * can parse the list to 3 sub lists

                # the following part is for plane
                xy_plane = conv[0, :, :]
                yz_plane = conv[:, 0, :]
                zx_plane = conv[:, :, 0]

                new_xy_plane = new_conv[0, :, :]
                new_yz_plane = new_conv[:, 0, :]
                new_zx_plane = new_conv[:, :, 0]

                for plane, new_plane in zip([xy_plane, yz_plane, zx_plane], [new_xy_plane, new_yz_plane, new_zx_plane]):
                    if any(plane):
                        if plane[0,0] and plane[0,-1] and plane[-1,0] and plane[-1,-1]:
                            new_plane[:,:]=1
                        if plane[0,0] and plane[0,-1]:
                            new_plane[0, :] = 1
                        if plane[0,0] and plane[-1,0]:
                            new_plane[:,0] = 1

                        if plane[0,0] and plane[-1,-1]:
                            a_lenth = plane.shape[0]
                            b_lenth = plane.shape[1]
                            a_index = np.arange(a_lenth)
                            b_index = np.round(a_index * b_lenth/a_lenth).astype('int')  # linear function
                            for a, b in zip(a_index, b_index):
                                new_plane[a, b] = 1


                # the following part is for cubic





                print(conv)

        return None


    def upsample_smooth(self, mask):
        mask_0_pad = self.up_mask_0_pad(mask)
        up_mask = self.mask_smooth(mask_0_pad)

        return up_mask






def one_hot_decoding(img,labels,thresh=[]):
    """
    get the one hot decode of img.

    :param img:
    :param labels:
    :param thresh:
    :return:
    """
    new_img = np.zeros((img.shape[0],img.shape[1]))
    r_img   = img.reshape(img.shape[0],img.shape[1],-1)

    aux = np.argmax(r_img,axis=-1)
    for i,l in enumerate(labels[1::]):
        if(thresh==[]):
            new_img[aux==(i+1)]=l
        else:
            new_img[r_img[:,:,i+1]>thresh] = l

    return new_img


def downsample(scan, ori_space=[], trgt_space=[], ori_sz=[], trgt_sz=[], order=1, labels=[0, 1], ):
    """


    :param scan: shape(z,x,y,chn)
    :param ori_space: shape(z,x,y)
    :param trgt_space: shape(z,x,y)
    :param ori_sz: shape(z,x,y,chn)
    :param trgt_sz: shape(z,x,y)
    :param order:
    :return:
    """
    if len(scan.shape)==3:  # (657, 512, 512)
        scan=scan[..., np.newaxis] # (657, 512, 512, 1)
    if len(ori_sz)==3:
        ori_sz = list(ori_sz)
        ori_sz.append(1) # (657, 512, 512, 1)
    print('scan.shape, ori_space, trgt_space, ori_sz, trgt_sz',scan.shape, ori_space, trgt_space, ori_sz, trgt_sz)
    if any(trgt_space):
        print('rescaled to new spacing  ')
        zoom_seq = np.array(ori_space, dtype='float') / np.array(trgt_space, dtype='float')
        zoom_seq = np.append(zoom_seq, 1)
    elif any(trgt_sz):
        print('rescaled to target size')
        zoom_seq = np.array(trgt_sz.append(ori_sz.shape[-1]), dtype='float') / np.array(ori_sz, dtype='float')
    else:
        raise Exception('please assign how to rescale')
        print('please assign how to rescale')

    print('zoom_seq', zoom_seq)
    if len(labels) <= 2 or all(zoom_seq<=1):  # [0, 1] vesel mask or original ct scans
        x = ndimage.interpolation.zoom(scan, zoom_seq, order=order, prefilter=order)  # 143, 271, 271, 1
    else:  # [0, 1, 2, 3, 4, 5, 6]
        x_onehot = one_hot_encode_3D(scan, labels) # (657, 512, 512, 6/2)
        mask1 = []
        for i in range(x_onehot.shape[-1]):
            one_chn = x_onehot[..., i]# (657, 512, 512)
            one_chn = one_chn[..., np.newaxis] # (657, 512, 512, 1)
            x1 = ndimage.interpolation.zoom(one_chn, zoom_seq, order=order, prefilter=order)
            mask1.append(x1)
        mask1 = np.array(mask1)  # (6/2, 567, 512, 512)
        mask1 = np.rollaxis(mask1, 0, start=4)
        mask3 = []
        for p in mask1:
            mask3.append(one_hot_decoding(p, labels))
        x = np.array(mask3, dtype='uint8')

    print('size after rescale:', x.shape)  # 64, 144, 144, 1

    return x

def one_hot_encode_3D( patch, labels):
    # todo: simplify this function

    # assert len(patch.shape)==5 # (5, 128, 128, 64, 1)
    labels = np.array(labels)  # i.e. [0,4,5,6,7,8]
    N_classes = labels.size  # 6, similiar with len(labels)
    if len(patch.shape) == 5:  # (5, 128, 128, 64, 1)
        patch = np.reshape(patch, (patch.shape[0], patch.shape[1], patch.shape[2], patch.shape[3]))
    elif len(patch.shape) == 4:  # (128, 128, 64, 1)
        patch = np.reshape(patch, (patch.shape[0], patch.shape[1], patch.shape[2]))
    patches = []
    # print('patch.shape', patch.shape)
    for i, l in enumerate(labels):
        a = np.where(patch != l, 0, 1)
        patches.append(a)

    patches = np.array(patches)
    patches = np.rollaxis(patches, 0, len(patches.shape))  # from [6, 64, 128, 128] to [64, 128, 128, 6]?

    # print('patches.shape after on hot encode 3D', patches.shape)

    return np.float64(patches)

def save_model_best(dice_file, segment, model_fpath):
    with open(dice_file, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter=',')
        dice_list = []
        for row in reader:
            dice = float(row['ave_total']) # str is the default type from csv
            dice_list.append(dice)

    max_dice = max(dice_list)
    if dice>=max_dice:
        segment.save(model_fpath)
        print("this 'ave_total' is the best: ", str(dice), "save valid model at: ", model_fpath)
    else:
        print("this 'ave_total' is not the best: ", str(dice), 'we do not save the model')

    return max_dice

def correct_shape(final_pred, original_shape):
    print('after rescale, the shape is: ', final_pred.shape)
    if final_pred.shape[0] != original_shape[0]:

        nb_slice_lost = abs(original_shape[0] - final_pred.shape[0])

        if original_shape[0] > final_pred.shape[0]:
            print(
                'there are {} slices lost along z axis, they will be repeated by the last slice'.format(nb_slice_lost))
            for i in range(nb_slice_lost):
                added_slice = np.expand_dims(final_pred[-1], axis=0)
                final_pred = np.concatenate((final_pred, added_slice))
            print('after repeating, the shape became: ', final_pred.shape)
        else:
            print('there are {} slices more along z axis, they will be cut'.format(nb_slice_lost))
            final_pred = final_pred[:original_shape[0]]  # original shape: (649, 512, 512)
            print('after cutting, the shape became: ', final_pred.shape)

    if final_pred.shape[1] != original_shape[1]:

        nb_slice_lost = abs(original_shape[1] - final_pred.shape[1])

        if original_shape[1] > final_pred.shape[1]:
            print('there are {} slices lost along x,y axis, they will be repeated by the last slice'.format(
                nb_slice_lost))
            for i in range(nb_slice_lost):
                added_slice = final_pred[:, -1, :]
                added_slice = np.expand_dims(added_slice, axis=1)
                print('x axis final_pred.shape', final_pred.shape)
                print('x axis add_slice.shape', added_slice.shape)
                final_pred = np.concatenate((final_pred, added_slice), axis=1)
                print('after first repeating, the shape is: ', final_pred.shape)

                added_slice = np.expand_dims(final_pred[:, :, -1], axis=2)
                print('y axis final_pred.shape', final_pred.shape)
                print('y axis add_slice.shape', added_slice.shape)
                final_pred = np.concatenate((final_pred, added_slice), axis=2)
            print('after repeating, the shape became: ', final_pred.shape)
        else:
            print('there are {} slices more along x,y axis, they will be cut'.format(nb_slice_lost))
            final_pred = final_pred[:, :original_shape[1], :original_shape[1]]  # original shape: (649, 512, 512)
            print('after cutting, the shape became: ', final_pred.shape)
    return final_pred

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