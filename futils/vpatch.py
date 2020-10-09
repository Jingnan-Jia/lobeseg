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
        m = s / 60
        h = m / 60
        print('the time cost by ', fun.__name__, ' is ', m, ' minuts.')
        return result

    return decorated


def get_a2_patch_origin_finish(a, origin, p_sh, a2):
    """

    :param a:
    :param origin:
    :param p_sh:
    :param a2:
    :return: origin_a2, finish_a2, numpy array, shape (3, )
    """
    p_sh = np.array(p_sh)
    origin = np.array(origin)
    scale_ratio = np.array(a2.shape) / np.array(a.shape)  # shape: (z, x,y,1)
    center_idx_a = origin + p_sh // 2  # (z, y, x)
    center_idx_a2 = center_idx_a * scale_ratio[:-1]  # (z, y, x)
    center_idx_a2 = center_idx_a2.astype(int)  # (z, y, x)
    origin_a2 = center_idx_a2 - p_sh // 2  # the patch size in a2 is still p_sh
    finish_a2 = center_idx_a2 + p_sh // 2  # (z, y, x)
    return origin_a2, finish_a2


def get_a2_patch(a2, patch_shape, ref, ref_origin):
    """

    :param a2: shape (z, y, x, chn)
    :param patch_shape: (96, 144, 144)
    :param ref: lower or higher resolution array with same dimentions with a2
    :param ref_origin: 3 dimensions (z,y,x)
    :return:
    """
    # get a2_patch according a and its ori, patchshape
    # get the origin list and finish list of a2 patch
    origin_a2, finish_a2 = get_a2_patch_origin_finish(ref, ref_origin, patch_shape, a2)  # np array (z,y,x) shape (3, )

    if all(i >= 0 for i in origin_a2) and all(m < n for m, n in zip(finish_a2, a2.shape)):
        # original_a2 is positive, and finish_a2 is smaller than a2.shape
        idx_a2 = [np.arange(o_, f_) for o_, f_ in zip(origin_a2, finish_a2)]
        a2_patch = a2[np.ix_(idx_a2[0], idx_a2[1], idx_a2[2])]

    else:  # origin_a2 is negative or finish_a2 is greater than a2.shape
        print("origin_a2 or finish_a2 is out of the a2 shape, prepare to do padding")
        pad_origin = np.zeros_like(origin_a2)
        pad_finish = np.zeros_like(finish_a2)
        for i in range(len(origin_a2)):
            if origin_a2[i] < 0:  # patch is out of the left or top of a
                pad_origin[i] = abs(origin_a2[i])
                origin_a2[i] = 0

            if finish_a2[i] > a2.shape[i]:  # patch is out of the right or bottom of a
                pad_finish[i] = finish_a2[i] - a2.shape[i]
                finish_a2[i] = a2.shape[i]

        idx_a2 = [np.arange(o_, f_) for o_, f_ in zip(origin_a2, finish_a2)]
        a2_patch = a2[np.ix_(idx_a2[0], idx_a2[1], idx_a2[2])]  # (z, y, x, chn)

        pad_origin, pad_finish = np.append(pad_origin, 0), np.append(pad_finish, 0)
        pad_width = tuple([(i, j) for i, j in zip(pad_origin, pad_finish)])
        a2_patch = np.pad(a2_patch, pad_width, mode='minimum')  # (z, y, x)

    return a2_patch, idx_a2


def random_patch_(scan, patch_shape, p_middle, needorigin=0, ptch_seed=None):
    random.seed(ptch_seed)
    if ptch_seed:
        print("ptch_seed for this patch is " + str(ptch_seed))
    sh = np.array(scan.shape)  # (z, y, x, chn)
    p_sh = np.array(patch_shape)  # (z, y, x)

    range_vals = sh[0:3] - p_sh  # (z, y, x)
    if any(range_vals <= 0):  # patch size is smaller than image shape
        raise Exception("patch size is bigger than image size. patch size is ", p_sh, " image size is", sh)

    origin = []
    if p_middle:  # set sampling specific probability on central part
        tmp_nb = random.random()
        if ptch_seed:
            print("p_middle random float number for this patch is " + str(tmp_nb))
        if tmp_nb < p_middle:
            range_vals_low = list(map(int, (sh[0:3] / 3 - p_sh // 2)))
            range_vals_high = list(map(int, (sh[0:3] * 2 / 3 - p_sh // 2)))
            # assert range_vals_low > 0 and range_vals_high > 0, means if one patch cannot include center area
            if all(i > 0 for i in range_vals_low) and all(j > 0 for j in range_vals_high):
                for low, high in zip(range_vals_low, range_vals_high):
                    origin.append(random.randint(low, high))
                # print('p_middle, select more big vessels!')
    if len(origin) == 0:
        origin = [random.randint(0, x) for x in range_vals]
        if ptch_seed:
            print("origin for this patch is " + str(origin))

    finish = origin + p_sh
    for finish_voxel, a_size in zip(finish, sh):
        if finish_voxel > a_size:
            print('warning!:patch size is bigger than a size', file=sys.stderr)

    idx = [np.arange(o_, f_) for o_, f_ in zip(origin, finish)]
    patch = scan[np.ix_(idx[0], idx[1], idx[2])]
    if needorigin:
        return patch, idx, origin
    else:
        return patch, idx


def random_patch(a_low, a_hgh, b_low, b_hgh, c_low, c_hgh, patch_shape=(64, 128, 128),
                 p_middle=None, task=None, io=None, ptch_seed=None):
    """
    get ramdom patches from the given ct.

    :param a: one ct array, dimensions: 4, shape order: (z, y, x, chn) todo: check the shape order
    :param b: ground truth, binary masks of object, shape order: (z, y, x, 2)
    :param c: auxiliary a, binary boundary of object, shape order: (z, y, x, 2)
    :param patch_shape: patch shape, shape order: (z, y, x) todo: check the shape order
    :param p_middle: float number between 0~1, probability of patches in the middle parts of ct a

    :return: one patch of ct a (with one patch of gdth and aux if gdth and aux are not None), dimensions: 4,shape oreders are the same as their correspoonding input as

    """
    if task != "vessel":
        p_middle = None
    if io == "1_in_low_1_out_low":
        a_low_patch, idx_low = random_patch_(a_low, patch_shape, p_middle, ptch_seed=ptch_seed)
        b_low_patch = b_low[np.ix_(idx_low[0], idx_low[1], idx_low[2])]
        if c_low is not None:
            c_low_patch = c_low[np.ix_(idx_low[0], idx_low[1], idx_low[2])]
            return [a_low_patch, [b_low_patch, c_low_patch]]
        else:
            return [a_low_patch, b_low_patch]
    elif io == "1_in_hgh_1_out_hgh":
        a_hgh_patch, idx_hgh = random_patch_(a_hgh, patch_shape, p_middle, ptch_seed=ptch_seed)
        b_hgh_patch = b_hgh[np.ix_(idx_hgh[0], idx_hgh[1], idx_hgh[2])]
        if c_hgh is not None:
            c_hgh_patch = c_hgh[np.ix_(idx_hgh[0], idx_hgh[1], idx_hgh[2])]
            return [a_hgh_patch, [b_hgh_patch, c_hgh_patch]]
        else:
            return [a_hgh_patch, b_hgh_patch]
    else:
        if task == "lobe":  # get idx_low at first
            a_low_patch, idx_low, origin = random_patch_(a_low, patch_shape, p_middle, needorigin=True,
                                                         ptch_seed=ptch_seed)
            a_hgh_patch, idx_hgh = get_a2_patch(a_hgh, patch_shape, ref=a_low, ref_origin=origin)
        else:
            a_hgh_patch, idx_hgh, origin = random_patch_(a_hgh, patch_shape, p_middle, needorigin=True,
                                                         ptch_seed=ptch_seed)
            a_low_patch, idx_low = get_a2_patch(a_low, patch_shape, ref=a_hgh, ref_origin=origin)
        if io == "2_in_1_out_low":  # get idx_hgh at first
            b_low_patch = b_low[np.ix_(idx_low[0], idx_low[1], idx_low[2])]
            return [[a_low_patch, a_hgh_patch], b_low_patch]
        elif io == "2_in_1_out_hgh":
            b_hgh_patch = b_hgh[np.ix_(idx_hgh[0], idx_hgh[1], idx_hgh[2])]
            return [[a_low_patch, a_hgh_patch], b_hgh_patch]
        else:
            b_low_patch = b_low[np.ix_(idx_low[0], idx_low[1], idx_low[2])]
            b_hgh_patch = b_hgh[np.ix_(idx_hgh[0], idx_hgh[1], idx_hgh[2])]
            return [[a_low_patch, a_hgh_patch], [b_low_patch, b_hgh_patch]]


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

    return n_patches


def get_stride_list(stride, p_sh):
    if isinstance(stride, float) or stride == 1:
        stride = np.array(p_sh) * stride
    elif isinstance(stride, list):
        stride = np.ones(3) * stride
    else:
        raise Exception('the stride is wrong', stride)
    stride = stride.astype(int)

    return stride


@calculate_time
def deconstruct_patch_gen(scan, patch_shape=(64, 128, 128), stride=0.25, a2=None):
    """
    deconstruct a ct to a batch of patches.

    :param scan: a ct array which need to be deconstructed to (overlapped) patches. shape (z, y, x, 1)
    :param patch_shape: shape (z, y, x)
    :param stride: overlap ratio along each axil. float between 0~1 means the same overlap ratio along all axils;1 means
     no overlap, a list means the corresponding different overlap ratio along different axils.
    :return: a batch of patches, shape (nb_ptches, z, y, x, chn)
    """
    sh = np.array(scan.shape, dtype=int)
    p_sh = np.array(patch_shape, dtype=int)
    if len(scan.shape) == 3:
        scan = scan[..., np.newaxis]
    stride = get_stride_list(stride, p_sh)

    n_patches = (sh[0:3] - p_sh + 2 * stride) // stride  # number of patches to cover  the whole image
    print('number of patches: ', n_patches)
    patches = []

    for z, y, x in np.ndindex(tuple(n_patches)):
        it = np.array([z, y, x], dtype=int)
        origin = it * stride
        finish = it * stride + p_sh
        for i in range(len(finish)):  # when we meet the last patch
            if finish[i] >= sh[i]:
                finish[i] = sh[i] - 1
                origin[i] = finish[i] - p_sh[i]
        idx = [np.arange(o_, f_) for o_, f_ in zip(origin, finish)]
        patch = scan[np.ix_(idx[0], idx[1], idx[2])]  # (96, 144, 144, 1)

        patch = np.rollaxis(patch, 0, 3)  # 48, 144, 144, 80, 1
        patch = patch[np.newaxis, ...]

        if a2 is not None:  # mtscale is 1, a2 is ori resolution, scan is low resolution
            a2_patch, _ = get_a2_patch(a2, p_sh, ref=scan, ref_origin=origin)  # (96, 144, 144, 1)
            a2_patch = np.rollaxis(a2_patch, 0, 3)  # 48, 144, 144, 80, 1
            a2_patch = a2_patch[np.newaxis, ...]
            if a2.shape[0] > scan.shape[0]:  # scan is low resolutin, a2 is high resolution
                patch = [patch, a2_patch]
            else:
                patch = [a2_patch, patch]

        yield patch


def reconstruct_one_from_patch_gen(scan, ptch_shape, original_shape=(128, 256, 256), stride=0.25, chn=6, mot=False):
    p_sh = np.array(ptch_shape)  # shape (z, y, x)
    if len(original_shape) == 4:
        sh = np.array(original_shape, dtype=int)[:-1]  # shape (z, y, x)
    else:
        sh = np.array(original_shape, dtype=int)  # shape (z, y, x)
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
    for z, y, x in np.ndindex(tuple(n_patches)):
        n += 1
        # print('reconstruct patch number ', n)
        it = np.array([z, y, x], dtype=int)
        origin = it * stride
        finish = it * stride + p_sh

        for i in range(len(finish)):  # when we meet the last patch
            if finish[i] >= sh[i]:
                finish[i] = sh[i] - 1
                origin[i] = finish[i] - p_sh[i]

        idx = [np.arange(o_, f_) for o_, f_ in zip(origin, finish)]
        patch = next(scan)  # (125, 128, 128, 64, 6)
        if type(patch) is list:
            if mot:
                patch = patch[0]
            else:
                patch = patch[0]
        patch = np.rollaxis(patch, 3, 1)  # (125, 64, 128, 128, 6)
        result[np.ix_(idx[0], idx[1], idx[2])] += patch[0]
    # normalize (0-1) output
    r_sum = np.sum(result, axis=-1)
    r_sum[r_sum == 0] = 1  # hmm this should not be necessary - but it is :(
    r_sum = np.repeat(r_sum[:, :, :, np.newaxis], chn, -1)
    # repeat the sum along channel axis (to allow division)
    result = np.divide(result, r_sum)

    return result


@calculate_time
def reconstruct_patch_gen(scan, ptch_shape, original_shape=(128, 256, 256), stride=0.25, chn=6, mot=None,
                          original_shape2=None):
    """
    reconstruct a ct from a batch of patches.

    :param scan: a batch of patches array, shape(nb, z, y, x, chn), 5 dims
    :param original_shape: shape of original ct scan, shape (z, y, x, 1), 4 dims
    :param stride: overlap ratio on each axil
    :return: a ct with original shape (z, y, x)
    """
    if mot:
        if original_shape2 is not None:
            if chn == 2:  # not lobe, reconstruct the 2 outputs
                # result = reconstruct_one_from_patch_gen(scan, ptch_shape, original_shape, stride,chn=6)  # this is downsampled output
                result2 = reconstruct_one_from_patch_gen(scan, ptch_shape, original_shape, stride, chn=chn, mot=True)
                result = result2

            elif chn == 6:  # lobe, just use the second output (low resolution), because first output is bad
                result = reconstruct_one_from_patch_gen(scan, ptch_shape, original_shape, stride, chn=chn, mot=True)
        else:
            raise Exception("mot is given, but original_shape2 is None, please assign original_shape2")
    else:  # not mot, use the only one output
        result = reconstruct_one_from_patch_gen(scan, ptch_shape, original_shape, stride, chn=6)

    return result
