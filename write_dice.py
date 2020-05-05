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

def load_scan(file_name):
    
    """Load mhd or nrrd 3d scan"""

    extension = file_name.split('.')[-1]
    if extension == 'mhd':
        scan, origin, spacing = futil.load_itk(file_name)

    elif extension == 'nrrd':
        scan, origin, spacing = futil.load_nrrd(file_name)

    return np.expand_dims(scan, axis=-1)

def calculate_dices(labels, a, b):
    '''
    this function is to calculate dice between the two files.  Both the files dimensions should be 4,
    shape is like: (512, 512, 400, 1) or (400, 512, 512, 1) 
    '''
    aa, bb = copy.deepcopy(a), copy.deepcopy(b)
    
    dices = []
    for l in labels[1:]:
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
    
    
def write_dices_to_csv(labels, gdth_path, pred_path, csv_file, gdth_extension='.nrrd', pred_extension='.nrrd'):
    '''
    this function is to calculate dice between the files in gdth_path and pred_mask_path. all the files must be 
    '.nrrd' or '.mhd'. All the files dimensions should be 4, shape is like: (512, 512, 400, 1) or (400, 512, 512, 1) 
    the default extension of masks are '.nrrd'
    '''
    print('start calculate dice and write dices to csv')
    gdth_files = sorted (glob.glob (gdth_path + '/*' + '.nrrd'))
    if len(gdth_files)==0:
        gdth_files = sorted (glob.glob (gdth_path + '/*' + '.mhd'))

    pred_files = sorted (glob.glob (pred_path + '/*' + '.nrrd'))
    if len(pred_files)==0:
        pred_files = sorted (glob.glob (pred_path + '/*' + '.mhd'))

    if len(gdth_files)==0:
        raise Exception('ground truth files  are None, Please check the directories', gdth_path)
    if  len(pred_files)==0:
        raise Exception (' predicted files are None, Please check the directories', pred_path)

    if len(pred_files) < len(gdth_files): # only predict several ct
        gdth_files = gdth_files[:len(pred_files)]
    total_dices = []
    total_dices_names = []
    dices_values_matrix = [] # for average computation
    for gdth_name, pred_name in zip(gdth_files, pred_files):
        
        gdth_name = gdth_name
        pred_name = pred_name
        
        gdth_file = load_scan(gdth_name)  # (219, 253, 253, 1)
        pred_file = load_scan(pred_name)
        
        dices_values = calculate_dices(labels, gdth_file, pred_file)

        dices_values_matrix.append(dices_values)
        
        dices_names = [gdth_name]
        for l in labels[1:]:
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

    names_ave_of_dice = ['ave_dice_class_'+ str(i) for i in range(len(labels))]
    total_dices_names.extend(names_ave_of_dice)

    # average dice of each image and their names
    ave_dice_of_imgs = np.average (dices_values_matrix, axis=1)
    total_dices.extend (ave_dice_of_imgs)

    names_ave_of_imgs = ['ave_dice_img_' + str (i) for i in range(len (pred_files))]
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

        
        