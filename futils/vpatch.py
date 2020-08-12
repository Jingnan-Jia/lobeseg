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
from functools import wraps
import time

def calculate_time(fun):
    @wraps(fun)
    def decorated(*args, **kwargs):
        time1 = time.time()
        result = fun(*args, **kwargs)
        time2 = time.time()
        s = time2 - time1
        m = s/60
        h = m/60
        print('the time cost by ', fun.__name__ , ' is ', m, ' mints.')
        return result

    return decorated

def get_a2_patch(scan, origin, p_sh, a2):
    """
    output a2_patch given a scan and the patch origin(start position) and  patch shape of the first patch,


    :param scan:  shape: (z, x, y, 1)
    :param origin: (z, x, y)
    :param p_sh: (z, x, y)
    :param a2: shape: (z, x, y, 1)
    :return:
    """
    # get a2_patch according scan and its ori, patchshape
    scale_ratio = np.array(a2.shape) / np.array(scan.shape)  # shape: (z, x,y,1)
    center_idx = origin + p_sh // 2 # (z, x, y)
    center_idx = center_idx * scale_ratio[:-1] # (z, x, y)
    center_idx = center_idx.astype(int)  # (z, x, y)
    origin_a2 = center_idx - p_sh // 2  # the patch size in a2 is still p_sh
    finish_a2 = center_idx + p_sh // 2  # (z, x, y)

    if scan.shape[0] < a2.shape[0] or all(i >= 0 for i in origin_a2) and all(m < n for m, n in zip(finish_a2, a2.shape)):
        # scan is downsampled from a2 or
        # a2 is downsampled from scan, original_a2 is positive, and finish_a2 is smaller than a2.shape
        idx_a2 = [np.arange(o_, f_) for o_, f_ in zip(origin_a2, finish_a2)]
        a2_patch = a2[np.ix_(idx_a2[0], idx_a2[1], idx_a2[2])]

    else:
        # origin_a2 is negative or finish_a2 is greater than a2.shape
        pad_origin = np.zeros_like(origin_a2)
        pad_finish = np.zeros_like(finish_a2)
        for i in range(len(origin_a2)):
            if origin_a2[i] < 0:  # patch is out of the left or top of scan
                pad_origin[i] = abs(origin_a2[i])
                origin_a2[i] = 0

            if finish_a2[i] > a2.shape[i]:  # patch is out of the right or bottom of scan
                pad_finish[i] = finish_a2[i] - a2.shape[i]
                finish_a2[i] = a2.shape[i]

        idx_a2 = [np.arange(o_, f_) for o_, f_ in zip(origin_a2, finish_a2)]
        a2_patch = a2[np.ix_(idx_a2[0], idx_a2[1], idx_a2[2])] # (z, x, y, 1)

        a2_patch = a2_patch[..., 0] # (z, x, y, 1)
        pad_width = tuple([(i, j) for i, j in zip(pad_origin, pad_finish)])
        a2_patch = np.pad(a2_patch, pad_width, mode='minimum') # (z, x, y)
        a2_patch = a2_patch[..., np.newaxis] # (z, x, y, 1)
    return a2_patch

def random_patch(scan,gt_scan = None, aux_scan = None, patch_shape=(64,128,128),p_middle=None, a2=None):
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
    if any(range_vals<=0): # patch size is smaller than image shape
        pad_finish = np.zeros_like(range_vals)
        for i in range(len(range_vals)):
            if range_vals[i] <= 0:
                pad_finish[i] = abs(range_vals[i])+1 # here +1 make the scan bigger than patch size

        pad_width = tuple([(0, j) for j in pad_finish])
        scan = np.pad(scan[-1], pad_width, mode='minimum')
        scan = scan[..., np.newaxis]
    # print('scan shape', sh)
    # print('patch shape', p_sh)
    # print('range values', range_vals)
    if p_middle:  # set sampling specific probability on central part
        # print('p_middle, select more big vessels')
        tmp_nb = random.random()
        if tmp_nb < p_middle:
            range_vals_low = list(map(int, (sh[0:3]/3 - p_sh//2) ))
            range_vals_high = list(map(int,(sh[0:3] * 2 /3 - p_sh//2) ))
            # assert range_vals_low > 0 and range_vals_high > 0
            if all(i>0 for i in range_vals_low) and all(j>0 for j in range_vals_high):
                origin = []
                for low, high in zip(range_vals_low, range_vals_high):
                    origin.append(random.randint(low, high))
                # print('p_middle, select more big vessels!')
            else:
                origin = [random.randint(0, x) for x in range_vals]
                # print('p_middle, but range value is negtive')

        else:  #patch from other parts
            origin = [random.randint(0, x) for x in range_vals]
            # print('p_middle, select small vessels!')

    else:
        origin = [random.randint(0, x) for x in range_vals]
        # print('No p_middle, select all vessels')




    finish  = origin + p_sh
    for finish_voxel, scan_size in zip(finish, sh):
        if finish_voxel > scan_size:
            print('warning!:patch size is bigger than scan size', file=sys.stderr)

    idx = [np.arange(o_,f_) for o_,f_ in zip(origin,finish)]
    patch = scan[np.ix_(idx[0], idx[1], idx[2])]


    if a2 is not None:
        a2_patch = get_a2_patch( scan, origin, p_sh, a2)
        if scan.shape[0]>a2.shape[0]:  # scan is original resolution, a2 is downsampled, we put patch from original resolutiion at first
            patch = np.concatenate((patch, a2_patch), axis=-1) # concatenate along the channel axil
        else: # a2 is original resolution, scan is downsampled, we put patch from original resolutiion at first still
            patch = np.concatenate((a2_patch, patch), axis=-1)  # concatenate along the channel axil

    if(gt_scan is not None):

        gt_patch = gt_scan[np.ix_(idx[0],idx[1],idx[2])]
        if (aux_scan is not None):
            aux_patch = aux_scan[np.ix_(idx[0], idx[1], idx[2])]
            return patch, gt_patch, aux_patch
        else:
            return patch,gt_patch
    else:
        return patch

def get_n_patches(scan, patch_shape=(64, 128, 128), stride=0.25):
    sh = np.array(scan.shape, dtype=int)
    p_sh = np.array(patch_shape, dtype=int)

    if isinstance(stride, float) or stride == 1:
        stride = p_sh * stride
    elif isinstance(stride, list):
        stride = np.ones(3) * stride
    else:
        raise Exception('the stride is wrong', stride)

    stride = stride.astype(int)
    n_patches = (sh[0:3] - p_sh + 2 * stride) // stride  # number of patches to cover  the whole image
    n_patches = n_patches[0] * n_patches[1] * n_patches[2]

    return  n_patches
@calculate_time
def deconstruct_patch_gen(scan, patch_shape=(64, 128, 128), stride=0.25, a2=None):
    """
    deconstruct a ct to a batch of patches.

    :param scan: a ct array which need to be deconstructed to (overlapped) patches. shape (z, x, y, 1)
    :param patch_shape: shape (z, x, y)
    :param stride: overlap ratio along each axil. float between 0~1 means the same overlap ratio along all axils;1 means
     no overlap, a list means the corresponding different overlap ratio along different axils.
    :return: a batch of patches, shape (nb_ptches, z, x, y, chn)
    """
    sh = np.array(scan.shape, dtype=int)
    p_sh = np.array(patch_shape, dtype=int)

    if isinstance(stride, float) or stride == 1:
        stride = p_sh * stride
    elif isinstance(stride, list):
        stride = np.ones(3) * stride
    else:
        raise Exception('the stride is wrong', stride)

    stride = stride.astype(int)
    n_patches = (sh[0:3] - p_sh + 2 * stride) // stride  # number of patches to cover  the whole image
    print('number of patches: ', n_patches)
    patches = []

    for z, x, y in np.ndindex(tuple(n_patches)):
        it = np.array([z, x, y], dtype=int)
        origin = it * stride
        finish = it * stride + p_sh
        for i in range(len(finish)):  # when we meet the last patch
            if finish[i] >= sh[i]:
                finish[i] = sh[i] - 1
                origin[i] = finish[i] - p_sh[i]
        idx = [np.arange(o_, f_) for o_, f_ in zip(origin, finish)]
        patch = scan[np.ix_(idx[0], idx[1], idx[2])]  # (96, 144, 144, 1)
        if a2 is not None:  # mtscale is 1
            # print('patching for a2 of mtscale')
            a2_patch = get_a2_patch(scan, origin, p_sh, a2)  # (96, 144, 144, 1)

            if scan.shape[0] > a2.shape[
                0]:
                # scan is original resolution, a2 is downsampled, we put patch from original resolutiion at first
                patch = np.concatenate((patch, a2_patch), axis=-1)  # concatenate along the channel axil
            else:  # a2 is original resolution, scan is downsampled, we put patch from original resolutiion at first still
                patch = np.concatenate((a2_patch, patch), axis=-1)  # concatenate along the channel axil

        patch = np.rollaxis(patch,0,3) # 48, 144, 144, 80, 1
        patch = patch[np.newaxis, ...]

        if a2 is not None:
            x1 = patch[..., 0]
            x1 = x1[..., np.newaxis]
            x2 = patch[..., 1]
            x2 = x2[..., np.newaxis]
            patch = [x1, x2]

        yield patch

def deconstruct_patch(scan,patch_shape=(64,128,128),stride = 0.25, a2=None):
    """
    deconstruct a ct to a batch of patches.

    :param scan: a ct array which need to be deconstructed to (overlapped) patches. shape (z, x, y, 1)
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
    print('number of patches: ', n_patches)
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
        patch = scan[np.ix_(idx[0],idx[1],idx[2])]  #(96, 144, 144, 1)
        if a2 is not None:  # mtscale is 1
            print('patching for a2 of mtscale')
            a2_patch = get_a2_patch(scan, origin, p_sh, a2) #(96, 144, 144, 1)

            if scan.shape[0] > a2.shape[0]:  # scan is original resolution, a2 is downsampled, we put patch from original resolutiion at first
                patch = np.concatenate((patch, a2_patch), axis=-1)  # concatenate along the channel axil
            else:  # a2 is original resolution, scan is downsampled, we put patch from original resolutiion at first still
                patch = np.concatenate((a2_patch, patch), axis=-1)  # concatenate along the channel axil , 96, 144, 144, 2

        patches.append(patch)

    patches = np.array(patches)
    return patches
    
def reconstruct_patch(scan,original_shape=(128,256,256),stride = 0.25):
    """
    reconstruct a ct from a batch of patches.

    :param scan: a batch of patches array, shape(nb, z, x, y, chn), 5 dims
    :param original_shape: shape of original ct scan, shape (z, x, y, 1), 4 dims
    :param stride: overlap ratio on each axil
    :return: a ct with original shape (z, x, y)
    """
    
    p_sh = np.array(scan.shape,dtype=int)[1:4]  # shape (z, x, y)
    sh = np.array(original_shape,dtype=int)[:-1]   # shape (z, x, y)
    
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





@calculate_time
def reconstruct_patch_gen(scan, ptch_shape, original_shape=(128, 256, 256), stride=0.25, chn=6):
    """
    reconstruct a ct from a batch of patches.

    :param scan: a batch of patches array, shape(nb, z, x, y, chn), 5 dims
    :param original_shape: shape of original ct scan, shape (z, x, y, 1), 4 dims
    :param stride: overlap ratio on each axil
    :return: a ct with original shape (z, x, y)
    """




    p_sh = np.array(ptch_shape)  # shape (z, x, y)
    sh = np.array(original_shape, dtype=int)[:-1]  # shape (z, x, y)

    if isinstance(stride, float) or stride == 1:
        stride = p_sh * stride
    elif isinstance(stride, list):
        stride = np.ones(3) * stride
    else:
        raise Exception('the stride is wrong', stride)

    stride = stride.astype(int)

    n_patches = (sh - p_sh + 2 * stride) // stride
    print('number of patches to reconstruct: ', n_patches)

    # init result
    result = np.zeros(tuple(sh) + (chn,), dtype=float)
    n = 0
    for z, x, y in np.ndindex(tuple(n_patches)):
        n += 1
        # print('reconstruct patch number ', n)
        it = np.array([z, x, y], dtype=int)
        origin = it * stride
        finish = it * stride + p_sh

        for i in range(len(finish)):  # when we meet the last patch
            if finish[i] >= sh[i]:
                finish[i] = sh[i] - 1
                origin[i] = finish[i] - p_sh[i]

        idx = [np.arange(o_, f_) for o_, f_ in zip(origin, finish)]
        patch = next(scan) # (125, 128, 128, 64, 6)
        patch = np.rollaxis(patch, 3, 1)  # (125, 64, 128, 128, 6)
        result[np.ix_(idx[0], idx[1], idx[2])] += patch[0]

    # normalize (0-1) output
    r_sum = np.sum(result, axis=-1)

    r_sum[r_sum == 0] = 1  # hmm this should not be necessary - but it is :(
    r_sum = np.repeat(r_sum[:, :, :, np.newaxis], chn, -1)
    # repeat the sum along channel axis (to allow division)

    result = np.divide(result, r_sum)

    return result