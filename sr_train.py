"""
Compiled models for different tasks.
=============================================================
Created on Tue Aug  4 09:35:14 2017
@author: Jingnan
"""

import tensorflow as tf
from futils.compiled_models import sr_model
from futils.dataloader import Sr_data_itr
from futils.util import save_model_best

from futils.mypath import Mypath
import os
from tensorflow.keras import callbacks
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from futils import segmentor as v_seg
from write_dice import write_dices_to_csv
from futils.write_batch_preds import write_preds_to_disk
import sys

tf.keras.mixed_precision.experimental.set_policy('infer') # mix precision training

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
K.set_session(sess)  # set this TensorFlow session as the default session for Keras





def main():
    mypath = Mypath("lobe")
    label = [0, 4, 5, 6, 7, 8]

    net = sr_model(lr=0.0001)

    model_figure_fpath = mypath.model_figure_path() + '/sr.png'
    plot_model(net, show_shapes=True, to_file=model_figure_fpath)
    print('successfully plot model structure at: ', model_figure_fpath)

    # save model architecture and config
    model_json = net.to_json()
    with open(mypath.json_fpath(), "w") as json_file:
        json_file.write(model_json)
        print('successfully write new json file of task lobe', mypath.json_fpath())


    x_dir = os.path.join(mypath.train_dir(), "gdth_ct", mypath.sub_dir())
    x_ts_dir = os.path.join(mypath.valid_dir(), "gdth_ct", mypath.sub_dir())

    tr_it = Sr_data_itr(x_dir=x_dir,
                 x_ext=".nrrd",
                 nb=18,
                 data_argum=1,
                        pps=1000)


    # enqueuer_train = GeneratorEnqueuer(tr_it.generator(), use_multiprocessing=True)
    # train_datas = enqueuer_train.get()
    # enqueuer_train.start()

    train_datas = tr_it.generator()

    train_csvlogger = callbacks.CSVLogger(mypath.train_log_fpath(), separator=',', append=True)
    saver_train = callbacks.ModelCheckpoint(filepath=mypath.model_fpath_best_train(),
                                         save_best_only=True,
                                         monitor='loss',
                                         save_weights_only=True,
                                         save_freq=1)
    steps = 100
    for i in range(steps):
        # x, y = next(train_datas)
        # x, y = x[np.newaxis, ...], y[np.newaxis, ...]
        net.fit(train_datas,
                steps_per_epoch=18000,
                use_multiprocessing=True,
                callbacks=[saver_train, train_csvlogger])

        # period_valid = 50000
        # if i % (period_valid) == (period_valid-1):  # one epoch for lobe segmentation, 20 epochs for vessel segmentation
        #     # save predicted results and compute the dices
        for phase in ['train', 'valid']:
            segment = v_seg.v_segmentor(batch_size=1,
                                        model=mypath.model_fpath_best_train(),
                                        ptch_sz=144, ptch_z_sz=96,
                                        trgt_space_list=[None, None, None],
                                        task='lobe', sr=True
                                        )

            write_preds_to_disk(segment=segment,
                                data_dir=mypath.ori_ct_path(phase),
                                preds_dir=mypath.pred_path(phase),
                                number=1, stride=0.25)  # set stride 0.8 to save time

            write_dices_to_csv(step_nb=i,
                               labels=label,
                               gdth_path=mypath.gdth_path(phase),
                               pred_path=mypath.pred_path(phase),
                               csv_file=mypath.dices_fpath(phase))

            save_model_best(dice_file=mypath.dices_fpath(phase),
                            phase=phase,
                                  segment=segment,
                                  model_fpath=mypath.model_fpath_best_valid())

            print('step number', i, 'lr for super resolutin is', K.eval(net.optimizer.lr), file=sys.stderr)


    # enqueuer_train.close()


if __name__=="__main__":
    main()