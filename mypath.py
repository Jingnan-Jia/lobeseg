from set_args import args
import os
import time
import numpy as np
from functools import wraps


def mkdir_dcrt(fun): # decorator to create directory if not exist
    """
    A decorator to make directory output from function if not exist.

    :param fun: a function which outputs a directory
    :return: decorated function
    """
    @wraps(fun)
    def decorated(*args, **kwargs):
        output = fun(*args, **kwargs)
        if '.' in output.split('/')[-1]:
            output = os.path.dirname(output)
            if not os.path.exists(output):
                os.makedirs(output)
                print('successfully create directory:', output)
        else:
            if not os.path.exists (output):
                os.makedirs (output)
                print('successfully create directory:', output)

        return fun(*args, **kwargs)

    return decorated


class Mypath(object):
    """
    Here, I use 'fpath' to indicatate full file path and name, 'path' to respresent the directory,
    'file_name' to respresent the file name in the parent directory.
    """
    def __init__(self, task, current_time=None):

        """
        initial valuables.

        :param task: task name, e.g. 'lobe'
        :param current_time: a string represent the current time. It is used to name models and log files. If None, the time will be generated automatically.
        """

        self.task = task
        self.dir_path = os.path.dirname (os.path.realpath (__file__)) # abosolute path of the current script
        self.model_path = os.path.join (self.dir_path, 'models')
        self.log_path = os.path.join (self.dir_path, 'logs')
        self.data_path = os.path.join (self.dir_path, 'data')
        self.results_path = os.path.join (self.dir_path, 'results')

        if current_time:
            self.current_time = current_time
        else:
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
        """
        Sub directory of tasks. It is used to choose different datasets (like 'GLUCOLD', 'SSc').

        :return: sub directory name
        """

        if self.task=='lobe':
            sub_dir = 'GLUCOLD'
        elif self.task=='vessel':
            sub_dir = 'SSc' # todo: set different vessel dataset apart form SSc, to verify the effect of spacing
        elif self.task == 'no_label':
            sub_dir = args.no_label_dir

        return sub_dir

    @mkdir_dcrt
    def train_dir(self):
        """
        training directory.

        :return: training dataset directory for a specific task
        """
        train_dir = self.data_path + '/' + self.task + '/train'
        return train_dir

    @mkdir_dcrt
    def valid_dir(self):
        """
        validation directory.

        :return: validation dataset directory for a specific task
        """
        valid_dir = self.data_path + '/' + self.task + '/valid'
        return valid_dir

    @mkdir_dcrt
    def task_log_dir(self):
        """
        log directory.
        :return: directory to save logs
        """
        task_log_dir = self.log_path + '/' + self.task
        return task_log_dir

    @mkdir_dcrt
    def task_model_dir(self):
        """
        model directory.
        :return: directory to save models
        """
        task_model_dir = self.model_path + '/' + self.task
        return task_model_dir

    @mkdir_dcrt
    def log_fpath(self):
        """
        log full path.

        :return: log full path with suffix .log
        """
        task_log_dir = self.task_log_dir()
        return task_log_dir + '/' + self.str_name + '.log'

    @mkdir_dcrt
    def tr_va_log_fpath(self):
        """
        log full path to save training and validation measuremets during training.

        :return: log full path with suffix .log
        """
        task_log_dir = self.task_log_dir()
        return task_log_dir + '/' + self.str_name + 'tr_va.log'

    @mkdir_dcrt
    def train_log_fpath(self):
        """
        log full path to save training  measuremets during training.

        :return: log full path with suffix .log
        """
        task_log_dir = self.task_log_dir()
        return task_log_dir + '/' + self.str_name + 'train.log'

    @mkdir_dcrt
    def model_figure_path(self):
        """
        Directory where to save figures of model architecture.

        :return: model figure directory
        """
        model_figure_path = self.dir_path + '/figures'
        return model_figure_path

    @mkdir_dcrt
    def json_fpath(self):
        """
        full path of model json.

        :return: model json full path
        """
        task_model_path = self.task_model_dir()
        return task_model_path + '/' + self.str_name + 'MODEL.json'

    @mkdir_dcrt
    def best_va_loss_location(self):
        """
        full path to save best model according to validation loss.
        :return: full path
        """
        task_model_path = self.task_model_dir()
        return task_model_path + '/' + self.str_name + '_tr_best.hdf5'



    @mkdir_dcrt
    def model_fpath_best_train(self):
        """
        full path to save best model according to training loss.
        :return: full path
        """
        task_model_path = self.task_model_dir()
        return task_model_path + '/' + self.str_name + 'MODEL.hdf5'

    @mkdir_dcrt
    def model_fpath_best_valid(self):
        """
        full path to save best model according to training loss.
        :return: full path
        """
        task_model_path = self.task_model_dir()
        return task_model_path + '/' + self.str_name + 'MODEL_valid.hdf5'


    @mkdir_dcrt
    def ori_ct_path(self, phase='train'):
        """
        absolute directory of the original ct for training dataset
        :param phase: 'train' or 'valid'
        :return: directory name
        """
        data_path = self.data_path + '/' + self.task + '/' + phase + '/ori_ct/' + self.sub_dir()
        return data_path

    @mkdir_dcrt
    def gdth_path(self, phase='train'):
        """
        absolute directory of the ground truth of ct for training dataset
        :param phase: 'train' or 'valid'
        :return: directory name
        """
        gdth_path = self.data_path + '/' + self.task + '/' + phase + '/gdth_ct/' + self.sub_dir()
        return gdth_path

    @mkdir_dcrt
    def pred_path(self, phase='train'):
        """
        absolute directory of the prediction results of ct for training dataset
        :param phase: 'train' or 'valid'
        :return: directory name
        """
        pred_path = self.results_path + '/' + self.task + '/' + phase + '/pred/' + self.sub_dir() + '/' + self.current_time
        return pred_path

    @mkdir_dcrt
    def dices_fpath(self, phase='train'):
        """
        full path of the saved dice
        :param phase: 'train' or 'valid'
        :return: file name to save dice
        """
        pred_path = self.pred_path(phase)
        return self.results_path + '/' + self.task + '/' + phase + '/pred/' + self.sub_dir() + '/' + self.current_time + '/dices.csv'
