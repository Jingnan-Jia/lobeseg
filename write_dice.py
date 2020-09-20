# -*- coding: utf-8 -*-
"""
this .py file is for dice
Created on Wed Apr 12 10:20:10 2020
@author: jjia
"""
import futils.util as futil
import copy
import csv
import glob
import os
import numpy as np
from futils.util import get_gdth_pred_names


def calculate_dices(labels, a, b):
    """
    This function is to calculate dice between the two files.  Both the files dimensions should be 4,
    shape is like: (512, 512, 400, 1) or (400, 512, 512, 1)

    Note: labels include background, but the output dices do not include the dice of background.

    :param labels: a list of labels
    :param a: image array
    :param b: image array
    :return: a list of dices
    """
    aa, bb = copy.deepcopy(a), copy.deepcopy(b)
    
    dices = []
    for l in labels: # only keep the valid labels
        a_ = np.where(aa != l, 0, 1)
        b_ = np.where(bb != l, 0, 1)

        product = a_ * b_
        product_sum = np.sum(product)
        
        a_sum = np.sum(a_)
        b_sum = np.sum(b_)
        
        smooth = 0.0001
        dice = (2. * product_sum + smooth) / (a_sum + b_sum + smooth)
        
        dices.append(dice)
        
    # print('dice_shape', dices.shape)

    return dices
    
    
def write_dices_to_csv(step_nb, labels, gdth_path, pred_path, csv_file, gdth_extension='.nrrd', pred_extension='.nrrd'):
    '''
    this function is to calculate dice between the files in gdth_path and pred_mask_path. all the files must be
    '.nrrd' or '.mhd'. All the files dimensions should be 4, shape is like: (512, 512, 400, 1) or (400, 512, 512, 1)
    the default extension of masks are '.nrrd'

    '''
    print('start calculate dice and write dices to csv')
    gdth_names, pred_names = get_gdth_pred_names(gdth_path, pred_path)

    total_dices_names = ['step_nb']  # dices_names corresponding to total_dices
    total_dices = [step_nb]
    dices_values_matrix = [] # for average computation
    for gdth_name, pred_name in zip(gdth_names, pred_names):
        gdth_name = gdth_name
        pred_name = pred_name
        gdth_file, _, _ = futil.load_itk(gdth_name)
        pred_file, _, _ = futil.load_itk(pred_name)
        dices_values = calculate_dices(labels, gdth_file, pred_file)  # calculated dices exclude background
        dices_values_matrix.append(dices_values)

        dices_names = [gdth_name]
        for l in labels:  # calculated dices exclude background
            dices_names.append('dice_'+str(l)) # dice_names is a list corresponding to the specific dices_values
        total_dices_names.extend(dices_names) # extend a list by another small list

        total_dices.append(True) # place a fixed number under the file name
        total_dices.extend(dices_values)
        print('dice_value')
        print(dices_values)

    dices_values_matrix = np.array(dices_values_matrix)

    # average dice of each class and their names
    ave_dice_of_class = np.average(dices_values_matrix, axis=0)
    total_dices.extend(ave_dice_of_class)

    names_ave_of_dice = ['ave_dice_class_'+ str(l) for l in labels]  # calculated ave dices exclude background
    total_dices_names.extend(names_ave_of_dice)

    # average dice of each image and their names
    ave_dice_of_imgs = np.average (dices_values_matrix, axis=1)
    total_dices.extend (ave_dice_of_imgs)

    names_ave_of_imgs = ['ave_dice_img_' + str (i) for i in range(len (pred_names))]
    total_dices_names.extend (names_ave_of_imgs)

    # average dices of total class and images
    ave_dice_total = np.average(dices_values_matrix)
    total_dices.append(ave_dice_total)

    name_ave_total = 'ave_total'
    total_dices_names.append (name_ave_total)

    if not os.path.exists(csv_file):
        with open(csv_file, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(total_dices_names)

    with open(csv_file, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(total_dices)
    print('finish writing dices to csv file at ' + csv_file)
    return None

def main():
    pass


if __name__=="__main__":
    main()
        