from set_args import args
from write_dice import write_dices_to_csv
import glob
from write_batch_preds import write_preds_to_disk
from train_ori_fit_rec_epoch import Mypath, get_label_list
import segmentor as v_seg

task_list = ['vessel', 'lobe']
path_list = [Mypath(x) for x in task_list] # a list of Mypath objectives
label_list = get_label_list(task_list)

for task, mypath, label in zip(task_list, path_list, label_list):
    if task != 'no_label':  # save predicted results and compute the dices
        for phase in ['valid']:
            segment = v_seg.v_segmentor(batch_size=args.batch_size,
                                        model=mypath.weights_location(),
                                        ptch_sz=args.ptch_sz, ptch_z_sz=args.ptch_z_sz,
                                        trgt_sz=args.trgt_sz, trgt_z_sz=args.trgt_z_sz,
                                        trgt_space_list=[args.trgt_z_space, args.trgt_space, args.trgt_space],
                                        task=task)

            write_preds_to_disk(segment=segment,
                                data_dir=mypath.data_path(phase),
                                preds_dir=mypath.pred_path(phase),
                                number=1, stride=0.5)

            write_dices_to_csv(labels=label,
                               gdth_path=mypath.gdth_path(phase),
                               pred_path=mypath.pred_path(phase),
                               csv_file=mypath.dices_location(phase))

