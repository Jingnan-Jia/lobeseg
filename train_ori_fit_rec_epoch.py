from __future__ import print_function

import numpy as np
# import matplotlib.pyplot as plt
import sys
from dataloader_ori_wo_reshape import TwoScanIterator
import time
import os

import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import GeneratorEnqueuer
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import callbacks
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
import compiled_models_complete_tf_keras

from set_args import args

os.environ['CUDA_VISIBLE_DEVICES'] = "0" # use the first GPU
tf.keras.mixed_precision.experimental.set_policy('infer') # mix precision training

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
K.set_session(sess)  # set this TensorFlow session as the default session for Keras

K.set_learning_phase(1)  # try with 1

class Mypath:
    '''
    Here, I use 'location' to indicatate full file path and name, 'path' to respresent the directory,
    'file_name' to respresent the file name in the parent directory.
    '''
    def __init__(self, task):

        self.task = task
        self.dir_path = os.path.dirname (os.path.realpath (__file__)) # abosolute path of the current script
        self.model_path = os.path.join (self.dir_path, 'models')
        self.log_path = os.path.join (self.dir_path, 'logs')
        self.data_path = os.path.join (self.dir_path, 'data')
        self.results_path = os.path.join (self.dir_path, 'results')


        self.current_time = str (time.time ())
        self.setting = '_' + str (args.lr) + str (args.load) + 'a_o_' + str (
            args.aux_output) + 'ds' + str (args.deep_supervision) + 'dr' + str (args.dropout) + 'bn' + str (
            args.batch_norm) + 'fs' + str (args.feature_size) + 'ptsz' + str (args.ptch_sz) + 'ptzsz' + str (args.ptch_z_sz)
        self.str_name = self.current_time + self.setting

        # self.model_png_location = '/exports/lkeb-hpc/jjia/project/e2e_new/newmodel_'+task+'.png'

    def sub_dir(self):

        if self.task=='lobe':
            if args.iso==1.5:
                sub_dir = 'GLUCOLD_isotropic1dot5'
            elif args.iso==0.7:
                sub_dir = 'GLUCOLD_isotropic0dot7'
            elif args.iso==0:
                sub_dir = 'GLUCOLD'
            elif args.iso==-1:
                sub_dir = 'luna16'
            else:
                raise Exception('Please enter the correct args.iso for isotropic parameter')
        elif self.task=='vessel':
            sub_dir = None

        return sub_dir

    def train_dir(self):
        train_dir = self.data_path + '/' + self.task + '/train'
        return train_dir

    def valid_dir(self):
        if self.task=='no_label':
            return self.train_dir()
        else:
            valid_dir = self.data_path + '/' + self.task + '/valid'
            return valid_dir

    def log_fpath(self):
        return self.log_path + '/' + self.task + '/' + self.str_name + '.log'



    def figure_path(self):
        figure_path = self.dir_path + '/figures'
        if not os.path.exists (figure_path):
            os.makedirs (figure_path)
        return figure_path

    def json_fpath(self):
        return self.model_path + '/' + self.task + '/' + self.str_name + 'MODEL.json'

    def best_weights_location(self):
        return self.model_path + '/' + self.task + '/' + self.str_name + '_best.hdf5'

    def weights_location(self):
        return self.model_path + '/' + self.task + '/' + self.str_name + 'MODEL.hdf5'

    def data_path(self, phase='train'):
        if self.task=='lobe':
            return self.data_path + '/' + self.task + '/' + phase + '/ori_ct/' + self.sub_dir()
        else:
            return self.data_path + '/' + self.task + '/' + phase + '/ori_ct'

    def gdth_path(self, phase='train'):
        if self.task == 'lobe':
            return self.data_path + '/' + self.task + '/' + phase + '/gdth_ct/' + self.sub_dir()
        else:
            return self.data_path + '/' + self.task + '/' + phase + '/gdth_ct'

    def pred_path(self, phase='train'):
        if self.task == 'lobe':
            return self.results_path + '/' + self.task + '/' + phase + '/pred/' + self.sub_dir()\
               + '/' + self.current_time[:8]
        else:
            return self.results_path + '/' + self.task + '/' + phase + '/pred/'+ self.current_time[:8]

    def dices_location(self, phase='train'):
        if self.task == 'lobe':
            return self.results_path + '/' + self.task + '/' + phase + '/pred/' + self.sub_dir()\
               + '/' + self.current_time[:8] + '/dices.csv'
        else:
            return self.results_path + '/' + self.task + '/' + phase + '/pred/' + self.current_time[:8]+ '/dices.csv'



def get_task_list(model_names):

    net_task_dict = {
        "net_itgt_lobe_recon": "lobe",
        "net_itgt_vessel_recon": "vessel",

        "net_lobe": "lobe",
        "net_vessel": "vessel",
        "net_bronchi": "bronchi",
        "net_lung": "lung",
        "net_airway": "airway",
        "net_no_label": "no_label",

        "net_only_lobe": "lobe",
        "net_only_vessel": "vessel",
        "net_only_bronchi": "bronchi",
        "net_only_lung": "lung",
        "net_only_airway": "airway"
    }
    return list (map (net_task_dict.get, model_names))

def get_label_list(task_list):

    task_label_dict = {
        "lobe": [0, 4, 5, 6, 7, 8],
        "vessel": [0, 1],
        "bronchi": [0, 1],
        "airway": [0, 1],
        "lung": [0, 1],
        "no_label": []
    }

    return list (map (task_label_dict.get, task_list))


def train():

    # Define the Model
    model_names = ['net_only_vessel']
    if args.aux_output and 'net_only_vessel' in model_names:
        print(model_names)
        raise Exception('net_only_vessel should not have aux output')


    task_list = get_task_list(model_names)
    label_list = get_label_list(task_list)
    path_list = [Mypath(x) for x in task_list] # a list of Mypath objectives

    net_list = compiled_models_complete_tf_keras.load_cp_models (model_names,
                                                                nch=1,
                                                                lr=args.lr,
                                                                nf=args.feature_size,
                                                                bn=args.batch_norm,
                                                                dr=args.dropout,
                                                                ds=args.deep_supervision,
                                                                aux=args.aux_output,
                                                                net_type='v')


    if args.load: # need to assign the old file name if args.load is True
        old_time = '1584923362.8464801_0.00011a_o_0.5ds2dr1bn1fs16ptsz144ptzsz64'
        str_name = old_time
        for net, task, mypath in zip (net_list, task_list, path_list):
            old_model_fpath = mypath.model_path + '/' + task + '/' + str_name + 'MODEL.hdf5'
            net.load_weights (old_model_fpath)
            print ('loaded weights successfully: ', old_model_fpath)

    train_data_gen_list = []
    valid_data_gen_list = []
    va_data_list = []
    for mypath, task, labels, net, model_name in zip (path_list, task_list, label_list, net_list, model_names):

        plot_model(net, show_shapes=True, to_file=mypath.figure_path() + '/' + model_name + '.png')

        if args.load: # load saved model
            old_time = '1584923362.8464801_0.00011a_o_0.5ds2dr1bn1fs16ptsz144ptzsz64'
            old_model_fpath = mypath.model_path + '/' + task + '/' + old_time + 'MODEL.hdf5'
            net.load_weights(old_model_fpath)
            print('loaded weights successfully: ', old_model_fpath)

        # save model architecture and config
        model_json = net.to_json ()
        with open (mypath.json_fpath(), "w") as json_file:
            json_file.write (model_json)
        print ('successfully write json file of task ', task, mypath.json_fpath())

        if args.iso==1.5:
            new_spacing=[1.5, 1.5, 1.5]
        elif args.iso==0.7:
            new_spacing=[0.7, 0.7, 0.7]
        else:
            new_spacing=None


        if task == 'vessel':
            b_extension = '.mhd'
            c_dir_name = None
        else:
            b_extension = '.nrrd'
            c_dir_name = 'aux_gdth'

        train_it = TwoScanIterator(mypath.train_dir(), task=task,
                                   batch_size=args.batch_size,
                                   c_dir_name=c_dir_name,
                                   sub_dir=mypath.sub_dir(),
                                   trgt_sz=args.trgt_sz, trgt_z_sz=args.trgt_z_sz,
                                   trgt_space=args.trgt_space, trgt_z_space=args.trgt_z_space,
                                   ptch_sz=args.ptch_sz, ptch_z_sz=args.ptch_z_sz,
                                   new_spacing=new_spacing,
                                   b_extension=b_extension,
                                   shuffle=True,
                                   patches_per_scan=args.patches_per_scan,
                                   data_argum=False,
                                   ds=args.deep_supervision,
                                   labels=labels,
                                   nb=args.tr_nb)

        valid_it = TwoScanIterator(mypath.valid_dir(), task=task,
                                 batch_size=args.batch_size,
                                 c_dir_name=c_dir_name,
                                 sub_dir=mypath.sub_dir(),
                                 trgt_sz=args.trgt_sz, trgt_z_sz=args.trgt_z_sz,
                                 trgt_space=args.trgt_space, trgt_z_space=args.trgt_z_space,
                                 ptch_sz=args.ptch_sz, ptch_z_sz=args.ptch_z_sz,
                                 new_spacing=new_spacing,
                                 b_extension=b_extension,
                                 shuffle=False,
                                 patches_per_scan=args.patches_per_scan,  # target size
                                 data_argum=False,
                                 ds=args.deep_supervision,
                                 labels=labels)


        enqueuer_train = GeneratorEnqueuer(train_it.generator(), use_multiprocessing=True)
        enqueuer_valid = GeneratorEnqueuer(valid_it.generator(), use_multiprocessing=True)

        train_datas = enqueuer_train.get ()
        valid_datas = enqueuer_valid.get ()

        enqueuer_train.start ()
        enqueuer_valid.start ()

        train_data_gen_list.append(train_datas)
        valid_data_gen_list.append(valid_datas)


        if task=='no_label':
            valid_data_x_numpy = []
            valid_data_y_numpy = []
            for i in range (5):
                one_valid_data = next (valid_datas)  # cost 7 seconds per image patch using val_it.generator()
                one_valid_data_x = one_valid_data[0]  # output shape:(1,144,144,80,1)
                one_valid_data_y = one_valid_data[1]  # output shape:(1,144,144,80,1)

                valid_data_x_numpy.append (one_valid_data_x[0])
                valid_data_y_numpy.append (one_valid_data_y[0])

            valid_data_numpy = (np.array (valid_data_x_numpy), np.array (valid_data_y_numpy))
        else:
            valid_data_x_numpy = []
            if args.aux_output and args.deep_supervision==2:
                out_nb = 4
                valid_data_y_numpy = [[], [], [], []]
            elif args.aux_output and args.deep_supervision==0:
                out_nb = 2
                valid_data_y_numpy = [[], []]
            elif args.aux_output==False and args.deep_supervision==2:
                out_nb = 3
                valid_data_y_numpy = [[], [], []]
            elif args.aux_output==False and args.deep_supervision==0:
                out_nb = 1
                valid_data_y_numpy = [[]]
            else:
                raise Exception('Please set the correct aux and ds!!!')
            for i in range(5):
                one_valid_data = next(valid_datas) # cost 7 seconds per image patch using val_it.generator()
                # x, y = next(train_it)
                one_valid_data_x = one_valid_data[0] # output shape:(1,144,144,80,1)
                one_valid_data_y = one_valid_data[1] # output 4 lists, each list has shape:(1,144,144,80,1)
                # from (1, 80, 144, 144, 1) to (144, 144, 80, 1)
                # one_valid_data_x = np.rollaxis(one_valid_data_x, 1, 4)
                # one_valid_data_y = np.rollaxis(one_valid_data_y, 1, 4)

                valid_data_x_numpy.append(one_valid_data_x[0])
                # valid_data_y_numpy.append (one_valid_data_y)
                for j in range(out_nb):
                    valid_data_y_numpy[j].append(one_valid_data_y[j][0])
            for _ in range(out_nb):
                valid_data_y_numpy[_] = np.asarray(valid_data_y_numpy[_])
            valid_data_numpy = (np.array(valid_data_x_numpy), valid_data_y_numpy)
        va_data_list.append(valid_data_numpy)

    for enqueuer_valid in valid_data_gen_list:
        enqueuer_valid.close ()

    best_loss_dic = {'lobe': 10000,
                     'vessel': 10000,
                     'airway': 10000,
                     'no_label': 10000}

    segment = None # segmentor for prediction every epoch
    training_step = 2500000
    for idx_ in range(training_step):
        print ('step number: ', idx_)
        for task, net, tr_data, va_data, label, mypath in zip(task_list, net_list, train_data_gen_list, va_data_list, label_list, path_list):
            if idx_==0: # assign the initial loss
                val_loss = best_loss_dic[task]

            x, y = next(tr_data) # tr_data is a generator or enquerer
            valid_data = va_data  # va_data is a fixed numpy data

            # callbacks
            csvlogger = callbacks.CSVLogger(mypath.log_fpath(), separator=',', append=True)
            if idx_>0 or os.path.exists(mypath.log_fpath()):
                    df = pd.read_csv(mypath.log_fpath())
                    EPOCH_INIT = df['epoch'].iloc[-1]
                    BEST_LOSS = min(df['val_loss'])
                    best_loss_dic[task] = BEST_LOSS
                    print('Resume training {2} from EPOCH_INIT {0}, BEST_LOSS {1}'.format(EPOCH_INIT, BEST_LOSS, task))
            else:
                BEST_LOSS = best_loss_dic[task]

            class ModelCheckpointWrapper(callbacks.ModelCheckpoint):
                def __init__(self, best_init=None, *arg, **kwagrs):
                    super().__init__(*arg, **kwagrs)
                    if best_init is not None:
                        self.best = best_init

            checkpointer = ModelCheckpointWrapper(best_init=BEST_LOSS,
                                                  filepath=mypath.best_weights_location(),
                                                  verbose=1,
                                                  save_best_only=True,
                                                  monitor='loss',
                                                  save_weights_only=False,
                                                  save_freq=1)
            saver = ModelCheckpointWrapper (best_init=BEST_LOSS,
                                                   filepath=mypath.weights_location(),
                                                   verbose=1,
                                                   save_best_only=True,
                                                   monitor='loss',
                                                   save_weights_only=True,
                                                   save_freq=1)

            if idx_ % (10000) == 0: # one epoch for lobe segmentation, 20 epochs for vessel segmentation
                history = net.fit (x, y,
                                   batch_size=args.batch_size,
                                   validation_data=valid_data,
                                   use_multiprocessing=True,
                                   callbacks=[checkpointer, saver, csvlogger])

                current_val_loss = history.history['val_loss'][0]
                old_val_loss = np.float(best_loss_dic[task])
                if current_val_loss<old_val_loss:
                    best_loss_dic[task] = history.history['val_loss']

            else:
                history = net.fit (x, y,
                                   batch_size=args.batch_size,
                                   use_multiprocessing=True,
                                   callbacks=[csvlogger])

            print (history.history.keys ())
            for key, result in history.history.items ():
                print(key, result)

    for enqueuer_train in train_data_gen_list:
        enqueuer_train.close () # in the future, close should be replaced by stop

    print('finish train: ', mypath.str_name)


if __name__ == '__main__':
    train()




