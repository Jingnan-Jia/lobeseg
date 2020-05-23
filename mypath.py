
from set_args import args
import os
import time

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


        self.current_time = str (int(time.time ()))
        self.setting = '_lr' + str (args.lr) +'ld'+str (args.load) + 'm6l' + str(args.model_6_levels) + 'm7l' + str(args.model_7_levels) +'pm' + str(args.p_middle) +\
                       'no_label_dir' + str(args.no_label_dir) + 'ao' + str (
            args.aux_output) + 'ds' + str (args.deep_supervision) + 'dr' + str (args.dropout) + 'bn' + str (
            args.batch_norm) + 'fn' + str (args.feature_number) + 'trsz' + str(args.trgt_sz) + 'trzsz' + str(args.trgt_z_sz)\
                       + 'trsp'+ str(args.trgt_space) + 'trzsp'+ str(args.trgt_z_space) + 'ptch_per_scan' + str(args.patches_per_scan)+ \
                       'tr_nb' + str(args.tr_nb) + 'ptsz' + str (args.ptch_sz) + 'ptzsz' + str (args.ptch_z_sz)
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
            sub_dir = None
        else:
            sub_dir = None

        return sub_dir

    def train_dir(self):
        train_dir = self.data_path + '/' + self.task + '/train'
        if not os.path.exists (train_dir):
            os.makedirs (train_dir)
        return train_dir

    def valid_dir(self):
        if self.task=='no_label':
            valid_dir = self.train_dir()
        else:
            valid_dir = self.data_path + '/' + self.task + '/valid'
        if not os.path.exists(valid_dir):
            os.makedirs(valid_dir)
        return valid_dir

    def log_fpath(self):
        task_log_dir = self.log_path + '/' + self.task
        if not os.path.exists (task_log_dir):
            os.makedirs (task_log_dir)
        return task_log_dir + '/' + self.str_name + '.log'
    def tr_va_log_fpath(self):
        task_log_dir = self.log_path + '/' + self.task
        if not os.path.exists(task_log_dir):
            os.makedirs(task_log_dir)
        return task_log_dir + '/' + self.str_name + 'tr_va.log'

    def train_log_fpath(self):
        task_log_dir = self.log_path + '/' + self.task
        if not os.path.exists(task_log_dir):
            os.makedirs(task_log_dir)
        return task_log_dir + '/' + self.str_name + 'train.log'

    def model_figure_path(self):
        model_figure_path = self.dir_path + '/figures'
        if not os.path.exists (model_figure_path):
            os.makedirs (model_figure_path)
        return model_figure_path

    def json_fpath(self):
        task_model_path = self.model_path + '/' + self.task
        if not os.path.exists (task_model_path):
            os.makedirs (task_model_path)
        return task_model_path + '/' + self.str_name + 'MODEL.json'

    def best_va_loss_location(self):
        task_model_path = self.model_path + '/' + self.task
        if not os.path.exists(task_model_path):
            os.makedirs(task_model_path)
        return task_model_path + '/' + self.str_name + 'MODEL.hdf5'

    def best_tr_loss_location(self):
        task_model_path = self.model_path + '/' + self.task
        if not os.path.exists(task_model_path):
            os.makedirs(task_model_path)
        return task_model_path + '/' + self.str_name + '_tr_best.hdf5'

    def ori_ct_path(self, phase='train'):
        if self.task=='lobe':
            data_path = self.data_path + '/' + self.task + '/' + phase + '/ori_ct/' + self.sub_dir()
        else:
            data_path = self.data_path + '/' + self.task + '/' + phase + '/ori_ct'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        return data_path

    def gdth_path(self, phase='train'):
        if self.task == 'lobe':
            gdth_path = self.data_path + '/' + self.task + '/' + phase + '/gdth_ct/' + self.sub_dir()
        else:
            gdth_path = self.data_path + '/' + self.task + '/' + phase + '/gdth_ct'
        if not os.path.exists(gdth_path):
            os.makedirs(gdth_path)
        return gdth_path

    def pred_path(self, phase='train'):
        if self.task == 'lobe':
            pred_path = self.results_path + '/' + self.task + '/' + phase + '/pred/' + self.sub_dir()\
               + '/' + self.current_time
        else:
            pred_path = self.results_path + '/' + self.task + '/' + phase + '/pred/'+ self.current_time
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
        return pred_path

    def dices_fpath(self, phase='train'):
        if self.task == 'lobe':
            return self.results_path + '/' + self.task + '/' + phase + '/pred/' + self.sub_dir()\
               + '/' + self.current_time + '/dices.csv'
        else:
            return self.results_path + '/' + self.task + '/' + phase + '/pred/' + self.current_time + '/dices.csv'
