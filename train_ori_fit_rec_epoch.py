# -*- coding: utf-8 -*-
"""
Main file to train the model.
=============================================================
Created on Tue Apr  4 09:35:14 2020
@author: Jingnan
"""

import numpy as np
# import matplotlib.pyplot as plt
from futils.dataloader import ScanIterator
import os
import csv
import gc
import sys
import tensorflow as tf
from tensorflow.keras.utils import GeneratorEnqueuer
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras.utils import plot_model
from futils import compiled_models as cpmodels

from set_args import args
from write_dice import write_dices_to_csv
from write_batch_preds import write_preds_to_disk
import segmentor as v_seg
from mypath import Mypath

# os.environ['CUDA_VISIBLE_DEVICES'] = "0" # use the first GPU
tf.keras.mixed_precision.experimental.set_policy('infer') # mix precision training

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
K.set_session(sess)  # set this TensorFlow session as the default session for Keras

K.set_learning_phase(1)  # try with 1


def get_task_list(model_names):
    """
    get task list according to a list of model names. one task may corresponds to multiple models.
    :param model_names: a list of model names
    :return: a list of tasks
    """

    net_task_dict = {
        "net_itgt_lobe_recon": "lobe",
        "net_itgt_vessel_recon": "vessel",
        "net_itgt_lung_recon": "lung",
        "net_itgt_airway_recon": "airway",

        "net_no_label": "no_label",

        "net_only_lobe": "lobe",
        "net_only_vessel": "vessel",
        "net_only_lung": "lung",
        "net_only_airway": "airway"
    }
    return list (map (net_task_dict.get, model_names))


def get_label_list(task_list):
    """
    Get the label list according to given task list.

    :param task_list: a list of task names.
    :return: a list of labels' list.
    """

    task_label_dict = {
        "lobe": [0, 4, 5, 6, 7, 8],
        "vessel": [0, 1],
        "airway": [0, 1],
        "lung": [0, 1],
        "no_label": []
    }

    return list (map (task_label_dict.get, task_list))

def save_model_best_valid(dice_file, segment, model_fpath):
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


def set_lr(K, net, model_names, task, idx_):
    if set(model_names) != set(['net_only_lobe', 'net_only_vessel']) and set(model_names) != set(
            ['net_only_lobe', 'net_only_vessel', 'net_no_label']):  # lr1
        if idx_ == 0:
            if task in ['lobe', 'vessel']:
                lr_seg = 0.0001
                K.set_value(net.optimizer.lr, lr_seg)
                print('step number', idx_, 'lr for', task, 'is', K.eval(net.optimizer.lr), file=sys.stderr)
            else:
                lr_seg = 0.00001
                K.set_value(net.optimizer.lr, lr_seg)
                print('step number', idx_, 'lr for', task, 'is', K.eval(net.optimizer.lr),
                      file=sys.stderr)
        if idx_ == 100000:
            if task in ['lobe', 'vessel']:
                lr_seg = 0.00001
                K.set_value(net.optimizer.lr, lr_seg)
                print('step number', idx_, 'lr for', task, 'is', K.eval(net.optimizer.lr), file=sys.stderr)
            else:
                lr_seg = 0.000001
                K.set_value(net.optimizer.lr, lr_seg)
                print('step number', idx_, 'lr for', task, 'is', K.eval(net.optimizer.lr),
                      file=sys.stderr)
    else:  # lr1
        if idx_ == 0:
            if task in ['lobe']:
                lr_seg = 0.0001
                K.set_value(net.optimizer.lr, lr_seg)
                print('step number', idx_, 'lr for', task, 'is', K.eval(net.optimizer.lr), file=sys.stderr)
            else:
                lr_seg = 0.00001
                K.set_value(net.optimizer.lr, lr_seg)
                print('step number', idx_, 'lr for', task, 'is', K.eval(net.optimizer.lr),
                      file=sys.stderr)
        if idx_ == 100000:
            if task in ['lobe']:
                lr_seg = 0.00001
                K.set_value(net.optimizer.lr, lr_seg)
                print('step number', idx_, 'lr for', task, 'is', K.eval(net.optimizer.lr), file=sys.stderr)
            else:
                lr_seg = 0.000001  # vessel task
                K.set_value(net.optimizer.lr, lr_seg)
                print('step number', idx_, 'lr for', task, 'is', K.eval(net.optimizer.lr),
                      file=sys.stderr)
    return net

def train():
    """
    Main function to train the model.

    :return: None
    """
    # Define the Model, use dash to seperate multi model names, do not use ',' to seperate,
    #  because ',' can lead to unknown error during parse arguments
    model_names = args.model_names.split('-')
    model_names = [i.lstrip() for i in model_names] # remove backspace before each model name

    print('model names: ', model_names)
    if args.aux_output and ('net_only_vessel' in model_names):
        print(model_names)
        raise Exception('net_only_vessel should not have aux output')

    task_list = get_task_list(model_names)
    label_list = get_label_list(task_list)
    path_list = [Mypath(x) for x in task_list] # a list of Mypath objectives

    net_list = cpmodels.load_cp_models(model_names,
                                           nch=1,
                                           lr=args.lr,
                                           nf=args.feature_number,
                                           bn=args.batch_norm,
                                           dr=args.dropout,
                                           ds=args.deep_supervision,
                                           aux=args.aux_output,
                                           net_type='v',
                                           mtscale=args.mtscale)

    train_data_gen_list = []
    for mypath, task, labels, net, model_name in zip (path_list, task_list, label_list, net_list, model_names):
        model_figure_fpath = mypath.model_figure_path() + '/' + model_name + '.png'
        plot_model(net, show_shapes=True, to_file=model_figure_fpath)
        print('successfully plot model structure at: ', model_figure_fpath)

        if args.load: # load saved model
            if task=='lobe':
                old_name = '1593000414_895_lr0.0001ld0mtscale1m6l0m7l0pm0.0no_label_dirLUNA16ao0ds0dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan100tr_nb18no_label_nb400ptsz144ptzsz96'
            elif task=='vessel':
                old_name = '1593000414_802_lr0.0001ld0mtscale1m6l0m7l0pm0.0no_label_dirLUNA16ao0ds0dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan100tr_nb18no_label_nb400ptsz144ptzsz96'
            elif task=='no_label':
                old_name = '1593000414_570_lr0.0001ld0mtscale1m6l0m7l0pm0.0no_label_dirLUNA16ao0ds0dr1bn1fn16trszNonetrzszNonetrsp1.4trzsp2.5ptch_per_scan100tr_nb18no_label_nb400ptsz144ptzsz96'
            saved_model = mypath.model_fpath_best_train(str_name=old_name)
            net.load_weights(saved_model)
            print('loaded weights successfully from: ', saved_model)

        # save model architecture and config
        model_json = net.to_json ()
        with open (mypath.json_fpath(), "w") as json_file:
            json_file.write (model_json)
            print ('successfully write new json file of task ', task, mypath.json_fpath())

        if task == 'vessel' or task=='no_label':
            b_extension = '.mhd'
            aux=0
        else:
            b_extension = '.nrrd'
            aux=args.aux_output

        train_it = ScanIterator(mypath.train_dir(), task=task,
                                sub_dir=mypath.sub_dir(),
                                b_extension=b_extension,
                                ptch_sz=args.ptch_sz, ptch_z_sz=args.ptch_z_sz,
                                trgt_sz=args.trgt_sz, trgt_z_sz=args.trgt_z_sz,
                                trgt_space=args.trgt_space, trgt_z_space=args.trgt_z_space,
                                data_argum=True,
                                patches_per_scan=args.patches_per_scan,
                                ds=args.deep_supervision,
                                labels=labels,
                                batch_size=args.batch_size,
                                shuffle=False,
                                nb=args.tr_nb,
                                nb_no_label=args.no_label_nb,
                                no_label_dir=args.no_label_dir,
                                p_middle=args.p_middle,
                                phase='train',
                                aux=aux,
                                mtscale=args.mtscale)

        enqueuer_train = GeneratorEnqueuer(train_it.generator(), use_multiprocessing=True)
        train_datas = enqueuer_train.get ()
        enqueuer_train.start ()
        train_data_gen_list.append(train_datas)

    best_tr_loss_dic = {'lobe': 10000, 'vessel': 10000, 'airway': 10000, 'no_label': 10000}
    training_step = 50000 # 200000steps, 40 validations

    for idx_ in range(training_step):
        print ('step number: ', idx_)
        for task, net, tr_data, label, mypath in zip(task_list, net_list, train_data_gen_list, label_list, path_list):
            x, y = next(tr_data) # tr_data is a generator or enquerer
            # callbacks
            train_csvlogger = callbacks.CSVLogger(mypath.train_log_fpath(), separator=',', append=True)

            BEST_TR_LOSS = best_tr_loss_dic[task] # set this value to record the best tr_loss for each task,
            # to save the best training model in fit() function
            class ModelCheckpointWrapper(callbacks.ModelCheckpoint):
                def __init__(self, best_init=None, *arg, **kwagrs):
                    super().__init__(*arg, **kwagrs)
                    if best_init is not None:
                        self.best = best_init

            saver_train = ModelCheckpointWrapper(best_init=BEST_TR_LOSS,
                                                  filepath=mypath.model_fpath_best_train(),
                                                  verbose=1,
                                                  save_best_only=True,
                                                  monitor='loss',
                                                  save_weights_only=True,
                                                  save_freq=1)
            history = net.fit(x, y,
                              batch_size=args.batch_size,
                              use_multiprocessing=True,
                              callbacks=[saver_train, train_csvlogger])

            current_tr_loss = history.history['loss'][0]
            old_tr_loss = np.float(best_tr_loss_dic[task])
            if current_tr_loss < old_tr_loss:
                best_tr_loss_dic[task] = current_tr_loss

            period_valid = 5000
            if (idx_ % (period_valid) == 0) and (task != 'no_label'): # one epoch for lobe segmentation, 20 epochs for vessel segmentation
                 # save predicted results and compute the dices
                for phase in ['train', 'valid']:

                    segment = v_seg.v_segmentor(batch_size=args.batch_size,
                                                model=mypath.model_fpath_best_train(),
                                                ptch_sz=args.ptch_sz, ptch_z_sz=args.ptch_z_sz,
                                                trgt_sz=args.trgt_sz, trgt_z_sz=args.trgt_z_sz,
                                                trgt_space_list=[args.trgt_z_space, args.trgt_space, args.trgt_space],
                                                task=task)

                    write_preds_to_disk(segment=segment,
                                        data_dir=mypath.ori_ct_path( phase),
                                        preds_dir=mypath.pred_path(phase),
                                        number=1, stride=0.8)  # set stride 0.8 to save time

                    write_dices_to_csv (step_nb=idx_,
                                        labels=label,
                                        gdth_path=mypath.gdth_path(phase),
                                        pred_path=mypath.pred_path(phase),
                                        csv_file= mypath.dices_fpath(phase))

                    save_model_best_valid(dice_file=mypath.dices_fpath(phase),
                                          segment=segment,
                                          model_fpath=mypath.model_fpath_best_valid())

                    # if set(model_names) != set(['net_only_lobe', 'net_only_vessel']) and set(model_names) != set(
                    #         ['net_only_lobe', 'net_only_vessel', 'net_no_label']):  # lr1
                    #     if idx_ == 0:
                    #         if task in ['lobe', 'vessel']:
                    #             lr_seg = 0.0001
                    #             K.set_value(net.optimizer.lr, lr_seg)
                    #             print('step number', idx_, 'lr for', task, 'is', K.eval(net.optimizer.lr),
                    #                   file=sys.stderr)
                    #         else:
                    #             lr_seg = 0.00001
                    #             K.set_value(net.optimizer.lr, lr_seg)
                    #             print('step number', idx_, 'lr for', task, 'is', K.eval(net.optimizer.lr),
                    #                   file=sys.stderr)
                    #     if idx_ == 50000:
                    #         if task in ['lobe', 'vessel']:
                    #             lr_seg = 0.00001
                    #             K.set_value(net.optimizer.lr, lr_seg)
                    #             print('step number', idx_, 'lr for', task, 'is', K.eval(net.optimizer.lr),
                    #                   file=sys.stderr)
                    #         else:
                    #             lr_seg = 0.000001
                    #             K.set_value(net.optimizer.lr, lr_seg)
                    #             print('step number', idx_, 'lr for', task, 'is', K.eval(net.optimizer.lr),
                    #                   file=sys.stderr)
                    # else:  # lr1
                    #     if idx_ == 0:
                    #         if task in ['lobe','vessel']:
                    #             lr_seg = 0.0001
                    #             K.set_value(net.optimizer.lr, lr_seg)
                    #             print('step number', idx_, 'lr for', task, 'is', K.eval(net.optimizer.lr),
                    #                   file=sys.stderr)
                    #         else:
                    #             lr_seg = 0.00001
                    #             K.set_value(net.optimizer.lr, lr_seg)
                    #             print('step number', idx_, 'lr for', task, 'is', K.eval(net.optimizer.lr),
                    #                   file=sys.stderr)
                    #     if idx_ == 50000:
                    #         if task in ['lobe','vessel']:
                    #             lr_seg = 0.00001
                    #             K.set_value(net.optimizer.lr, lr_seg)
                    #             print('step number', idx_, 'lr for', task, 'is', K.eval(net.optimizer.lr),
                    #                   file=sys.stderr)
                    #         else:
                    #             lr_seg = 0.000001  # vessel task
                    #             K.set_value(net.optimizer.lr, lr_seg)
                    #             print('step number', idx_, 'lr for', task, 'is', K.eval(net.optimizer.lr),
                    #                   file=sys.stderr)

            for key, result in history.history.items ():
                print(key, result)

    for enqueuer_train in train_data_gen_list:
        enqueuer_train.close () # in the future, close should be replaced by stop

    print('finish train: ', mypath.str_name)


if __name__ == '__main__':
    train()




