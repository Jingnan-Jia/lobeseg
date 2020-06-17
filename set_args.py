import argparse

parser = argparse.ArgumentParser(description='End2End Semi-Supervised Lobe Segmentation')


parser.add_argument(
    '-lr',
    '--lr',
    help='learning rate',
    type=float,
    default=0.0001)

parser.add_argument(
    '-load',
    '--load',
    help='load last model',
    type=int,
    default=0)

parser.add_argument(
    '-aux',
    '--aux_output',
    help='Value of Auxiliary Output',
    type=float,
    default=0)

parser.add_argument(
    '-ds',
    '--deep_supervision',
    help='Number of Deep Supervisers',
    type=int,
    default=0)

parser.add_argument(
    '-fn',
    '--feature_number',
    help='Number of initial of conv channels',
    type=int,
    default=16)

parser.add_argument(
    '-bn',
    '--batch_norm',
    help='Set Batch Normalization',
    type=int,
    default=1)

parser.add_argument(
    '-dr',
    '--dropout',
    help='Set Dropout',
    type=int,
    default=1)

parser.add_argument(
    '-trgt_sz',
    '--trgt_sz',
    help='target size',
    type=int,
    default=None)

parser.add_argument(
    '-trgt_z_sz',
    '--trgt_z_sz',
    help='target z size',
    type=int,
    default=None)

parser.add_argument(
    '-trgt_space',
    '--trgt_space',
    help='spacing along x, y ',
    type=float,
    default=1.4)

parser.add_argument(
    '-trgt_z_space',
    '--trgt_z_space',
    help='spacing along z',
    type=float,
    default=2.5)

parser.add_argument(
    '-ptch_sz',
    '--ptch_sz',
    help='patch size',
    type=int,
    default=144)

parser.add_argument(
    '-ptch_z_sz',
    '--ptch_z_sz',
    help='patch z size',
    type=int,
    default=96)

parser.add_argument(
    '-batch_size',
    '--batch_size',
    help='batch_size',
    type=int,
    default=1)

parser.add_argument(
    '-patches_per_scan',
    '--patches_per_scan',
    help='patches_per_scan',
    type=int,
    default=100)

parser.add_argument(
    '-tr_nb',
    '--tr_nb',
    help='nunber of training samples',
    type=int,
    default=18)

parser.add_argument(
    '-no_label_dir',
    '--no_label_dir',
    help='sub directory of no_label data',
    type=str,
    default='GLUCOLD')


parser.add_argument(
    '-p_middle',
    '--p_middle',
    help='p_middle = 0.5 means sample in the middle parts',
    type=str,
    default=0.5)

parser.add_argument(
    '-model_names',
    '--model_names',
    help='model names',
    type=str,
    default='net_only_vessel')

parser.add_argument(
    '-model_6_levels',
    '--model_6_levels',
    help='get a model_6_levels  net with 6 levels (normally there are 5 levels)',
    type=int,
    default=0)

parser.add_argument(
    '-model_7_levels',
    '--model_7_levels',
    help='get a model_7_levels  net with 6 levels (normally there are 5 levels)',
    type=int,
    default=0)

parser.add_argument(
    '-model_mt_scales',
    '--model_mt_scales',
    help='get a model of multi scales  net with 5 levels (normally there are 5 levels)',
    type=int,
    default=0)

args = parser.parse_args()