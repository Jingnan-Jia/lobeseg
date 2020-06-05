# -*- coding: utf-8 -*-
"""
patch, deconstruct and reconstruct ct array.
=============================================
Created on  May 06 14:57:47 2020
@author: fferreira and Jingnan
"""


import numpy as np
import random
import sys

def random_patch(scan,gt_scan = None, aux_scan = None, patch_shape=(64,128,128),p_middle=None):
    """
    get ramdom patches from the given ct.

    :param scan: one ct scan array, dimensions: 4, shape order: (z, x, y, chn) todo: check the shape order
    :param gt_scan: ground truth, binary masks of object, shape order: (z, x, y, 2)
    :param aux_scan: auxiliary scan, binary boundary of object, shape order: (z, x, y, 2)
    :param patch_shape: patch shape, shape order: (z, x, y) todo: check the shape order
    :param p_middle: float number between 0~1, probability of patches in the middle parts of ct scan
    :return: one patch of ct scan (with one patch of gdth and aux if gdth and aux are not None), dimensions: 4,shape oreders are the same as their correspoonding input scans

    """
    sh = np.array(scan.shape)  # (z, x, y, chn)
    p_sh = np.array(patch_shape)  # (z, x, y)

    range_vals = sh[0:3] - p_sh  # (z, x, y)
    # print('scan shape', sh)
    # print('patch shape', p_sh)
    # print('range values', range_vals)
    
   
    if p_middle:  # set sampling specific probability on central part
        # print('p_middle, select more big vessels')
        if random.random() < p_middle:
            range_vals_low = list(map(int, (sh[0:3]/3 - p_sh//2) ))
            range_vals_high = list(map(int,(sh[0:3] * 2 /3 - p_sh//2) ))
            # assert range_vals_low > 0 and range_vals_high > 0
            if range_vals_low>0 and range_vals_high>0:
                origin = []
                for low, high in zip(range_vals_low, range_vals_high):
                    origin.append(random.randint(low, high))
            else:
                origin = [random.randint(0, x) for x in range_vals]

        else:  #patch from other parts
            origin = [random.randint(0, x) for x in range_vals]

    else:
        origin = [random.randint(0, x) for x in range_vals] # here, x+1 can avoid lob>high
    finish  = origin + p_sh
    for finish_voxel, scan_size in zip(finish, sh):
        if finish_voxel > scan_size:
            print('warning!:patch size is bigger than scan size', file=sys.stderr)
    
    idx = [np.arange(o_,f_) for o_,f_ in zip(origin,finish)]
    
       
    patch = scan[np.ix_(idx[0],idx[1],idx[2])]

    
    if(gt_scan is not None):
        gt_patch = gt_scan[np.ix_(idx[0],idx[1],idx[2])]
        if (aux_scan is not None):
            aux_patch = aux_scan[np.ix_(idx[0], idx[1], idx[2])]
            return patch, gt_patch, aux_patch
        else:
            return patch,gt_patch
    else:
        return patch
        
def deconstruct_patch(scan,patch_shape=(64,128,128),stride = 0.25):
    """
    deconstruct a ct to a batch of patches.

    :param scan: a ct array which need to be deconstructed to (overlapped) patches. shape (z, x, y, chn)
    :param patch_shape: shape (z, x, y)
    :param stride: overlap ratio along each axil. float between 0~1 means the same overlap ratio along all axils;1 means no overlap, a list means the corresponding different overlap ratio along different axils.
    :return: a batch of patches, shape (nb_ptches, z, x, y, chn)
    """
    
    sh = np.array(scan.shape,dtype=int)
    p_sh = np.array(patch_shape,dtype=int)
    
    if isinstance(stride, float) or stride==1:
        stride  = p_sh * stride
    elif isinstance(stride, list):
        stride  = np.ones(3)*stride
    else:
        raise Exception('the stride is wrong', stride)
    
    stride = stride.astype(int)

    n_patches =   (sh[0:3] - p_sh + 2 * stride)  // stride # number of patches to cover  the whole image
    

    
    patches = []
    
    for z,x,y in np.ndindex(tuple(n_patches)):
        it = np.array([z,x,y],dtype= int)
        origin = it * stride
        finish = it * stride + p_sh

        for i in range(len(finish)): # when we meet the last patch
            if finish[i] >= sh[i]:
                finish[i] = sh[i]- 1
                origin[i] = finish[i] - p_sh[i]
        
        idx = [np.arange(o_,f_) for o_,f_ in zip(origin,finish)]
        
       
        patch = scan[np.ix_(idx[0],idx[1],idx[2])]
        
        patches.append(patch)
    
    patches = np.array(patches)    
    return patches
    
def reconstruct_patch(scan,original_shape=(128,256,256),stride = 0.25):
    """
    reconstruct a ct from a batch of patches.

    :param scan: a batch of patches array, shape(nb, z, x, y, chn)
    :param original_shape: shape of original ct scan, shape (z, x, y)
    :param stride: overlap ratio on each axil
    :return: a ct with original shape (z, x, y)
    """
    
    p_sh = np.array(scan.shape,dtype=int)[1:4]  # shape (z, x, y)
    sh = np.array(original_shape,dtype=int)  # shape (z, x, y)
    
    if isinstance(stride, float) or stride==1:
        stride  = p_sh * stride
    elif isinstance(stride, list):
        stride  = np.ones(3)*stride
    else:
        raise Exception('the stride is wrong', stride)

    stride = stride.astype(int)

    n_patches = (sh - p_sh + 2 * stride) // stride

    #init result
    result = np.zeros(tuple(sh)+(scan.shape[-1],),dtype=float)

    index = 0
    for z,x,y in np.ndindex(tuple(n_patches)):
        it = np.array([z,x,y],dtype= int)
        origin = it * stride
        finish = it * stride + p_sh

        for i in range(len(finish)): # when we meet the last patch
            if finish[i] >= sh[i]:
                finish[i] = sh[i]- 1
                origin[i] = finish[i] - p_sh[i]

        idx = [np.arange(o_,f_) for o_,f_ in zip(origin,finish)]

        result[np.ix_(idx[0],idx[1],idx[2])] += scan[index]

        index +=1
      
    #normalize (0-1) output
    r_sum = np.sum(result,axis=-1)

    r_sum[r_sum==0] = 1 #hmm this should not be necessary - but it is :(
    r_sum = np.repeat(r_sum[:,:,:,np.newaxis],scan.shape[-1],-1) #repeat the sum along channel axis (to allow division)
    
    result = np.divide(result,r_sum) 
    
    return result
    