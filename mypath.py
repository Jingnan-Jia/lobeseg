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

def get_short_names(long_names):
    """
    get task list according to a list of model names. one task may corresponds to multiple models.
    :param model_names: a list of model names
    :return: a list of tasks
    """

    net_task_dict = {
        "net_itgt_lobe_recon": "nilr",
        "net_itgt_vessel_recon": "nivr",
        "net_itgt_lung_recon": "nilr",
        "net_itgt_airway_recon": "niar",

        "net_no_label": "nnl",

        "net_only_lobe": "nol",
        "net_only_vessel": "nov",
        "net_only_lung": "nol",
        "net_only_airway": "noa"
    }
    return list (map (net_task_dict.get, long_names))


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
        long_names = args.model_names.split('-')
        long_names = [i.lstrip() for i in long_names] # remove backspace before each model name
        short_names = get_short_names(long_names)
        short_names = '-'.join(short_names)

        if args.trgt_sz:
            tr_sz_name = 'tsz' + str(args.trgt_sz) + 'z' + str(args.trgt_z_sz)
        else:
            tr_sz_name = ''

        self.setting = '_lr' + str (args.lr) \
                       + 'lrvs' + str (args.lr_vs) \
                       + 'ld' + str (args.load) \
                       + 'mtscale' + str (args.mtscale) \
                       + 'net' + str(short_names) \
                       + 'pm' + str(args.p_middle) \
                       + 'nld' + str(args.no_label_dir) \
                       + 'ao' + str (args.aux_output) \
                       + 'ds' + str (args.deep_supervision) \
                       + 'bn' + str (args.batch_norm) \
                       + 'fn' + str (args.feature_number) \
                       + tr_sz_name \
                       + 'tsp'+ str(args.trgt_space) \
                       + 'z'+ str(args.trgt_z_space) \
                       + 'pps' + str(args.patches_per_scan) \
                       + 'trnb' + str(args.tr_nb) \
                       + 'nlnb' + str(args.no_label_nb) \
                       + 'ptsz' + str (args.ptch_sz) \
                       + 'ptzsz' + str (args.ptch_z_sz)

        self.str_name = self.current_time + self.setting
        a = len(self.str_name)

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
    def model_fpath_best_train(self, str_name=None):
        """
        full path to save best model according to training loss.
        :return: full path
        """
        task_model_path = self.task_model_dir()
        if str_name is None:
            return task_model_path + '/' + self.str_name + 'MODEL.hdf5'
        else:
            return task_model_path + '/' + str_name + 'MODEL.hdf5'

    @mkdir_dcrt
    def model_fpath_best_valid(self, str_name=None):
        """
        full path to save best model according to training loss.
        :return: full path
        """
        task_model_path = self.task_model_dir()
        if str_name is None:
            return task_model_path + '/' + self.str_name + 'MODEL_valid.hdf5'
        else:
            return task_model_path + '/' + str_name + 'MODEL_valid.hdf5'


    @mkdir_dcrt
    def ori_ct_path(self, phase='train', sub_dir=None):
        """
        absolute directory of the original ct for training dataset
        :param phase: 'train' or 'valid'
        :return: directory name
        """
        if sub_dir is None:
            data_path = self.data_path + '/' + self.task + '/' + phase + '/ori_ct/' + self.sub_dir()
        else:
            data_path = self.data_path + '/' + self.task + '/' + phase + '/ori_ct/' + sub_dir
        return data_path

    @mkdir_dcrt
    def gdth_path(self, phase='train', sub_dir=None):
        """
        absolute directory of the ground truth of ct for training dataset
        :param phase: 'train' or 'valid'
        :return: directory name
        """
        if sub_dir is None:
            gdth_path = self.data_path + '/' + self.task + '/' + phase + '/gdth_ct/' + self.sub_dir()
        else:
            gdth_path = self.data_path + '/' + self.task + '/' + phase + '/gdth_ct/' + sub_dir
        return gdth_path

    @mkdir_dcrt
    def pred_path(self, phase='train', sub_dir=None):
        """
        absolute directory of the prediction results of ct for training dataset
        :param phase: 'train' or 'valid'
        :return: directory name
        """
        if sub_dir is None:
            pred_path = self.results_path + '/' + self.task + '/' + phase + '/pred/' + self.sub_dir() + '/' + self.current_time
        else:
            pred_path = self.results_path + '/' + self.task + '/' + phase + '/pred/' + sub_dir + '/' + self.current_time

        return pred_path

    @mkdir_dcrt
    def dices_fpath(self, phase='train'):
        """
        full path of the saved dice
        :param phase: 'train' or 'valid'
        :return: file name to save dice
        """
        pred_path = self.pred_path(phase)
        return pred_path + '/dices.csv'

    @mkdir_dcrt
    def all_metrics_fpath(self, phase='train', sub_dir=None):
        """
        full path of the saved dice
        :param phase: 'train' or 'valid'
        :return: file name to save dice
        """
        pred_path = self.pred_path(phase, sub_dir=sub_dir)
        return pred_path + '/all_metrics.csv'



