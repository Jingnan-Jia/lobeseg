import matplotlib.pyplot as plt
import os
import csv
from scipy.ndimage.filters import uniform_filter1d
# import pixiedust

class Hist:
    def __init__(self, file_name, log_dir, task_name, skip_number, new_arch=True, net_name=None, average_N=100):
        ...
        ...
        self.file_name = os.path.join(log_dir, task_name, file_name)
        self.skip_number = skip_number
        self.task_name = task_name
        if task_name == 'vessel':
            self.out_chn = 2
        if task_name == 'airway':
            self.out_chn = 2
        if task_name == 'lobe':
            self.out_chn = 6
        if task_name == 'no_label':
            self.out_chn = 1
        #         print(self.out_chn)
        self.train_color = 'blue'
        self.average_N = average_N
        self.new_arch = new_arch
        if not self.new_arch:
            self.val_out = 'val_out_' + self.task_name + '_segmentation_dice_coef_mean'
            self.train_out = 'out_' + self.task_name + '_segmentation_dice_coef_mean'
        else:
            self.val_out = 'val_' + self.task_name + '_out_segmentation_dice_coef_mean'
            self.train_out = self.task_name + '_out_segmentation_dice_coef_mean'

        self.valid_color = 'red'
        self.super_title = net_name

    def _plot(self, x, y_valid, y_train=None):

        fig = plt.figure(facecolor='w', figsize=(6, 6), dpi=300)
        # fig.suptitle(self.super_title, fontsize=16)
        ax = fig.add_subplot(111)
        ax.scatter(x, y_valid, s=5, c=self.valid_color, marker='o',
                   label='valid')
        y_valid = uniform_filter1d(y_valid, size=self.average_N)
        ax.plot(x, y_valid, c=self.valid_color, label='valid_average')
        if y_train is not None:
            # #             ax2 = fig.add_axes([0,0,1,1])
            ax.scatter(x, y_train, s=5, c=self.train_color, marker='o',
                       label='train')
            y_train = uniform_filter1d(y_train, size=self.average_N)
            ax.plot(x, y_train, c=self.train_color, label='train_average')
        #             ax2.legend()
        ax.legend()
        ax.set_ylim((0, 1))

        ax.set_xlim(left=0)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1])
        ax.set_ylabel('dice')
        ax.set_xlabel('steps')
        # plt.xlim(left=0)
        # plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.plot([0, 90000], [0.95, 0.95], 'k-', lw=1, dashes=[2, 2])
        #         ax.plot([0,90000], [0.97, 0.97], 'k-', lw=1, dashes=[2,2])

        plt.rc('font', size=18)

        # plt.show()
        frame = plt.gca()
        frame.axes.get_yaxis().set_visible(True)
        frame.axes.get_xaxis().set_visible(True)
        plt.savefig('all_train_valid_dice.png')
        plt.close()

    def _get_data(self, y_name):
        with open(self.file_name, encoding='utf-8') as f:
            reader = csv.DictReader((l.replace('\0', '') for l in f))  # avoid error: 'line contains null'
            index = 0
            x = []
            val_ave_dice = []
            for row in reader:
                #                 print(row/)
                if row[self.val_out] is not None and row[self.train_out] is not None:
                    x.append(index)
                    val_ave_dice.append(float(row[y_name]))
                index += 1

            y = val_ave_dice

            return (x, y)

    def plot_val_dice(self):
        # #         self.super_title = self.super_title + 'valid_dice_mean'
        x, y = self._get_data(self.val_out)
        self._plot(x, y)


    def plot_train_dice(self):
        #         self.super_title = self.task_name + 'train_dice_mean'
        x, y = self._get_data(self.train_out)
        self._plot(x, y)

    def plot_train_val_dice(self):
        x, y_valid = self._get_data(self.val_out)
        _, y_train = self._get_data(self.train_out)

        self._plot(x, y_valid, y_train=y_train)



    def plot_all_val_dice(self):
        fig = plt.figure(facecolor='w', figsize=(6, 6), dpi=300)
        ax = fig.add_subplot(111)

        for i in range(1, self.out_chn):
            if self.new_arch:
                y_name = 'val_' + self.task_name + '_out_segmentation_dice_' + str(i)
            else:
                y_name = 'val_out_' + self.task_name + '_segmentation_dice_' + str(i)
            x, y = self._get_data(y_name)
            y = uniform_filter1d(y, size=self.average_N)
            ax.plot(x, y, label='val_dice_' + str(i))
        ax.legend()
        ax.set_ylim((0, 1))

        ax.set_xlim(left=0)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1])
        ax.set_ylabel('dice')
        ax.set_xlabel('steps')
        # plt.xlim(left=0)
        # plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.plot([0, 90000], [0.95, 0.95], 'k-', lw=1, dashes=[2, 2])
        #         ax.plot([0,90000], [0.97, 0.97], 'k-', lw=1, dashes=[2,2])

        plt.rc('font', size=18)
        frame = plt.gca()
        frame.axes.get_yaxis().set_visible(True)
        frame.axes.get_xaxis().set_visible(True)
        # plt.show()
        plt.savefig('all_valid_dice.png')
        plt.close()


    def plot_all_train_dice(self):
        fig = plt.figure(facecolor='w', figsize=(6, 6), dpi=300)
        ax = fig.add_subplot(111)
        for i in range(1, self.out_chn):
            if self.new_arch:
                y_name = self.task_name + '_out_segmentation_dice_' + str(i)
            else:
                y_name = 'out_' + self.task_name + '_segmentation_dice_' + str(i)

            x, y = self._get_data(y_name)
            x, y = x[::skip_number], y[::skip_number]
            y = uniform_filter1d(y, size=self.average_N)
            ax.plot(x, y, label='train_dice_' + str(i))
        ax.legend()
        ax.set_ylim((0, 1))

        ax.set_xlim(left=0)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1])
        ax.set_ylabel('dice')
        ax.set_xlabel('steps')
        # plt.xlim(left=0)
        # plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.plot([0, 90000], [0.95, 0.95], 'k-', lw=1, dashes=[2, 2])
        #         ax.plot([0,90000], [0.97, 0.97], 'k-', lw=1, dashes=[2,2])

        plt.rc('font', size=18)
        # plt.show()
        frame = plt.gca()
        frame.axes.get_yaxis().set_visible(True)
        frame.axes.get_xaxis().set_visible(True)
        plt.savefig('all_train_dice.png')
        plt.close()



class Hist_:
    def __init__(self, file_name, task_name, skip_number, new_arch=True, net_name=None, average_N=100):
        ...
        ...

        self.skip_number = skip_number
        self.task_name = task_name
        if task_name == 'vessel':
            self.out_chn = 2
        if task_name == 'airway':
            self.out_chn = 2
        if task_name == 'lobe':
            self.out_chn = 6
        if task_name == 'no_label':
            self.out_chn = 1
        #         print(self.out_chn)
        self.train_color = 'blue'
        self.average_N = average_N
        self.new_arch = new_arch


        self.val_out =  'ave_total'
        self.train_out =  'ave_total'

        self.valid_color = 'red'
        self.super_title = net_name

    def _plot(self, x, y_valid, y_train=None):

        fig = plt.figure(facecolor='w', figsize=(6, 6), dpi=300)
        # fig.suptitle(self.super_title, fontsize=16)
        ax = fig.add_subplot(111)
        ax.scatter(x, y_valid, s=5, c=self.valid_color, marker='o',
                   label='valid')
        y_valid = uniform_filter1d(y_valid, size=self.average_N)
        ax.plot(x, y_valid, c=self.valid_color, label='valid_average')
        if y_train is not None:
            # #             ax2 = fig.add_axes([0,0,1,1])
            ax.scatter(x, y_train, s=5, c=self.train_color, marker='o',
                       label='train')
            y_train = uniform_filter1d(y_train, size=self.average_N)
            ax.plot(x, y_train, c=self.train_color, label='train_average')
        #             ax2.legend()
        ax.legend()
        ax.set_ylim((0, 1))

        ax.set_xlim(left=0)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1])
        ax.set_ylabel('dice')
        ax.set_xlabel('epochs')
        # plt.xlim(left=0)
        # plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.plot([0, 90000], [0.95, 0.95], 'k-', lw=1, dashes=[2, 2])
        #         ax.plot([0,90000], [0.97, 0.97], 'k-', lw=1, dashes=[2,2])

        plt.rc('font', size=18)

        # plt.show()
        frame = plt.gca()
        frame.axes.get_yaxis().set_visible(True)
        frame.axes.get_xaxis().set_visible(True)
        plt.savefig('all_train_valid_dice.png')
        plt.close()

    def _get_data(self, y_name):
        with open(self.file_name, encoding='utf-8') as f:
            reader = csv.DictReader((l.replace('\0', '') for l in f))  # avoid error: 'line contains null'
            index = 0
            x = []
            val_ave_dice = []
            for row in reader:
                #                 print(row/)
                x.append(index)
                val_ave_dice.append(float(row[y_name]))
                index += 1

            y = val_ave_dice

            return (x, y)

    def plot_val_dice(self):
        # #         self.super_title = self.super_title + 'valid_dice_mean'
        x, y = self._get_data(self.val_out)
        self._plot(x, y)


    def plot_train_dice(self):
        #         self.super_title = self.task_name + 'train_dice_mean'
        x, y = self._get_data(self.train_out)
        self._plot(x, y)

    def plot_train_val_dice(self):
        self.log_dir = 'results/' + task_name + '/valid/pred/GLUCOLD/' + number + '/'
        self.file_name = os.path.join(self.log_dir, file_name)
        x, y_valid = self._get_data(self.val_out)

        self.log_dir = 'results/' + task_name + '/train/pred/GLUCOLD/' + number + '/'
        self.file_name = os.path.join(self.log_dir, file_name)

        _, y_train = self._get_data(self.train_out)

        self._plot(x, y_valid, y_train=y_train)



    def plot_all_val_dice(self):

        self.log_dir = 'results/' + task_name + '/valid/pred/GLUCOLD/' + number + '/'
        self.file_name = os.path.join(self.log_dir, file_name)

        fig = plt.figure(facecolor='w', figsize=(6, 6), dpi=300)
        ax = fig.add_subplot(111)

        for i in range(1, self.out_chn):
            y_name = 'ave_dice_class_'+str(i)

            x, y = self._get_data(y_name)
            y = uniform_filter1d(y, size=self.average_N)
            ax.plot(x, y, label='val_dice_' + str(i))
        ax.legend()
        ax.set_ylim((0, 1))

        ax.set_xlim(left=0)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1])
        ax.set_ylabel('dice')
        ax.set_xlabel('epochs')
        # plt.xlim(left=0)
        # plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.plot([0, 90000], [0.95, 0.95], 'k-', lw=1, dashes=[2, 2])
        #         ax.plot([0,90000], [0.97, 0.97], 'k-', lw=1, dashes=[2,2])

        plt.rc('font', size=18)
        frame = plt.gca()
        frame.axes.get_yaxis().set_visible(True)
        frame.axes.get_xaxis().set_visible(True)
        # plt.show()
        plt.savefig('all_valid_dice.png')
        plt.close()

    def plot_all_train_dice(self):
        self.log_dir = 'results/' + task_name + '/train/pred/GLUCOLD/' + number + '/'
        self.file_name = os.path.join(self.log_dir, file_name)

        fig = plt.figure(facecolor='w', figsize=(6, 6), dpi=300)
        ax = fig.add_subplot(111)

        for i in range(1, self.out_chn):
            y_name = 'ave_dice_class_'+str(i)

            x, y = self._get_data(y_name)
            y = uniform_filter1d(y, size=self.average_N)
            ax.plot(x, y, label='train_dice_' + str(i))
        ax.legend()
        ax.set_ylim((0, 1))

        ax.set_xlim(left=0)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1])
        ax.set_ylabel('dice')
        ax.set_xlabel('epochs')
        # plt.xlim(left=0)
        # plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.plot([0, 90000], [0.95, 0.95], 'k-', lw=1, dashes=[2, 2])
        #         ax.plot([0,90000], [0.97, 0.97], 'k-', lw=1, dashes=[2,2])

        plt.rc('font', size=18)
        frame = plt.gca()
        frame.axes.get_yaxis().set_visible(True)
        frame.axes.get_xaxis().set_visible(True)
        # plt.show()
        plt.savefig('all_train_dice.png')
        plt.close()




task_name = 'vessel'
number = '15850005'
file_name = '1588276777.08482_0.00010a_o_0ds2dr1bn1fs8ptsz144ptzsz96.log'
log_dir = 'logs/'
skip_number = 1 # patches per epoch

hist = Hist(file_name, log_dir, task_name, skip_number, new_arch=True, average_N=10)


hist.plot_all_val_dice()

hist.plot_all_train_dice()

hist.plot_train_val_dice()

# 'net_only_lobe'
# task_name = 'lobe'
# number = '15847902'
# file_name = 'dices.csv'
# skip_number = 10 # patches per epoch
#
# hist = Hist_(file_name, task_name, skip_number, new_arch=True, average_N=10)
#
#
# hist.plot_all_val_dice()
#
# hist.plot_all_train_dice()
#
# hist.plot_train_val_dice()