# import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
# # 就是将你上层的文件夹添加到工作路径，当然包括文件夹下。
# # 另外，这三行要放到你要导入的模块的前面（貌似不用我提醒。。）


from write_batch_preds import write_preds_to_disk
import segmentor as v_seg
from mypath import Mypath
import sys

from set_args import args

print('args.batch_size', args.batch_size)
print('args.model_fpath', args.model_fpath)
print('args.ptch_sz')
print('args.ptch_z_sz')
print('args.trgt_z_space')
print('args.trgt_space')
print('args.task', args.task)
print('args.data_dir', args.data_dir)
print('args.preds_dir',args.preds_dir)

print(sys.argv)

segment = v_seg.v_segmentor(batch_size=args.batch_size,
                            model=args.model_fpath,
                            ptch_sz = args.ptch_sz, ptch_z_sz = args.ptch_z_sz,
                            trgt_sz = args.trgt_sz, trgt_z_sz = args.trgt_z_sz,
                            trgt_space_list=[args.trgt_z_space, args.trgt_space, args.trgt_space],
                            task=args.task)

write_preds_to_disk(segment=segment,
                    data_dir = args.data_dir,
                    preds_dir= args.preds_dir,
                    number=1, stride = 0.8) # set stride 0.8 to save time

