
from set_args import args
import os
import time
import numpy as np
from functools import wraps


def mkdir_dcrt(fun):
    @wraps(fun)
    def decorated(*args, **kwargs):
        output = fun(*args, **kwargs)
        if os.path.isdir(output):
            if not os.path.exists (output):
                os.makedirs (output)
                print('successfully create directory:', output)
        elif os.path.isfile(output):
            output = os.path.dirname(output)
            if not os.path.exists (output):
                os.makedirs (output)
                print('successfully create directory:', output)

        return output
    return decorated


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


        self.current_time = str (int(time.time ())) + '_' + str(np.random.randint(1000))
        self.setting = '_lr' + str (args.lr) \
                       + 'ld' + str (args.load) \
                       + 'm6l' + str(args.model_6_levels) \
                       + 'm7l' + str(args.model_7_levels) \
                       + 'pm' + str(args.p_middle) \
                       + 'no_label_dir' + str(args.no_label_dir) \
                       + 'ao' + str (args.aux_output) \
                       + 'ds' + str (args.deep_supervision) \
                       + 'dr' + str (args.dropout) \
                       + 'bn' + str (args.batch_norm) \
                       + 'fn' + str (args.feature_number) \
                       + 'trsz' + str(args.trgt_sz) \
                       + 'trzsz' + str(args.trgt_z_sz)\
                       + 'trsp'+ str(args.trgt_space) \
                       + 'trzsp'+ str(args.trgt_z_space) \
                       + 'ptch_per_scan' + str(args.patches_per_scan) \
                       + 'tr_nb' + str(args.tr_nb) \
                       + 'ptsz' + str (args.ptch_sz) \
                       + 'ptzsz' + str (args.ptch_z_sz)

        self.str_name = self.current_time + self.setting

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
            sub_dir = None # todo: set different vessel dataset apart form SSc, to verify the effect of spacing
        else:
            sub_dir = None

        return sub_dir

    @mkdir_dcrt
    def train_dir(self):
        train_dir = self.data_path + '/' + self.task + '/train'
        return train_dir

    @mkdir_dcrt
    def valid_dir(self):
        if self.task=='no_label':
            valid_dir = self.train_dir()
        else:
            valid_dir = self.data_path + '/' + self.task + '/valid'
        return valid_dir

    @mkdir_dcrt
    def task_log_dir(self):
        task_log_dir = self.log_path + '/' + self.task
        return task_log_dir

    def task_model_dir(self):
        task_model_dir = self.model_path + '/' + self.task
        return task_model_dir



    def log_fpath(self):
        task_log_dir = self.task_log_dir()
        return task_log_dir + '/' + self.str_name + '.log'
    def tr_va_log_fpath(self):
        task_log_dir = self.task_log_dir()
        return task_log_dir + '/' + self.str_name + 'tr_va.log'

    def train_log_fpath(self):
        task_log_dir = self.task_log_dir()
        return task_log_dir + '/' + self.str_name + 'train.log'

    @mkdir_dcrt
    def model_figure_path(self):
        model_figure_path = self.dir_path + '/figures'
        return model_figure_path

    def json_fpath(self):
        task_model_path = self.task_model_dir()
        return task_model_path + '/' + self.str_name + 'MODEL.json'

    def best_va_loss_location(self):
        task_model_path = self.task_model_dir()
        return task_model_path + '/' + self.str_name + 'MODEL.hdf5'

    def best_tr_loss_location(self):
        task_model_path = self.task_model_dir()
        return task_model_path + '/' + self.str_name + '_tr_best.hdf5'

    @mkdir_dcrt
    def ori_ct_path(self, phase='train'):
        if self.task=='lobe':
            data_path = self.data_path + '/' + self.task + '/' + phase + '/ori_ct/' + self.sub_dir()
        else:
            data_path = self.data_path + '/' + self.task + '/' + phase + '/ori_ct'
        return data_path

    @mkdir_dcrt
    def gdth_path(self, phase='train'):
        if self.task == 'lobe':
            gdth_path = self.data_path + '/' + self.task + '/' + phase + '/gdth_ct/' + self.sub_dir()
        else:
            gdth_path = self.data_path + '/' + self.task + '/' + phase + '/gdth_ct'


        return gdth_path

    @mkdir_dcrt
    def pred_path(self, phase='train'):
        if self.task == 'lobe':
            pred_path = self.results_path + '/' + self.task + '/' + phase + '/pred/' + self.sub_dir()\
               + '/' + self.current_time
        else:
            pred_path = self.results_path + '/' + self.task + '/' + phase + '/pred/'+ self.current_time
        return pred_path

    def dices_fpath(self, phase='train'):
        if self.task == 'lobe':
            return self.results_path + '/' + self.task + '/' + phase + '/pred/' + self.sub_dir()\
               + '/' + self.current_time + '/dices.csv'
        else:
            return self.results_path + '/' + self.task + '/' + phase + '/pred/' + self.current_time + '/dices.csv'
