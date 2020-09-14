import argparse

parser = argparse.ArgumentParser(description='End2End Semi-Supervised Lobe Segmentation')

parser.add_argument('-model_names', '--model_names', help='model names', type=str,
                    default='net_only_vessel-net_no_label-net_only_lobe')

parser.add_argument('-lr_lb', '--lr_lb', help='learning rate for lobe segmentation', type=float, default=0.0001)
parser.add_argument('-lr_vs', '--lr_vs', help='learning rate for vessel segmentation', type=float, default=0.00001)
parser.add_argument('-lr_lu', '--lr_lu', help='learning rate for lung segmentation', type=float, default=0.00001)
parser.add_argument('-lr_aw', '--lr_aw', help='learning rate for airway segmentation', type=float, default=0.00001)
parser.add_argument('-lr_rc', '--lr_rc', help='learning rate for reconstruction', type=float, default=0.00001)

parser.add_argument('-adaptive_lr', '--adaptive_lr', help='adaptive learning rate', type=int, default=1)
parser.add_argument('-attention', '--attention', help='attention loss', type=int, default=1)

parser.add_argument('-ds_lb', '--ds_lb', help='Number of Deep Supervisers', type=int, default=0)
parser.add_argument('-ds_vs', '--ds_vs', help='Number of Deep Supervisers', type=int, default=0)
parser.add_argument('-ds_aw', '--ds_aw', help='Number of Deep Supervisers', type=int, default=0)
parser.add_argument('-ds_lu', '--ds_lu', help='Number of Deep Supervisers', type=int, default=0)
parser.add_argument('-ds_rc', '--ds_rc', help='Number of Deep Supervisers', type=int, default=0)

parser.add_argument('-ao_lb', '--ao_lb', help='Value of Auxiliary Output', type=int, default=0)
parser.add_argument('-ao_vs', '--ao_vs', help='Value of Auxiliary Output', type=int, default=0)
parser.add_argument('-ao_aw', '--ao_aw', help='Value of Auxiliary Output', type=int, default=0)
parser.add_argument('-ao_lu', '--ao_lu', help='Value of Auxiliary Output', type=int, default=0)
parser.add_argument('-ao_rc', '--ao_rc', help='Value of Auxiliary Output', type=int, default=0)

parser.add_argument('-fn', '--feature_number', help='Number of initial of conv channels', type=int, default=16)
parser.add_argument('-bn', '--batch_norm', help='Set Batch Normalization', type=int, default=1)
parser.add_argument('-dr', '--dropout', help='Set Dropout', type=int, default=1)

parser.add_argument('-trgt_sz', '--trgt_sz', help='target size', type=int, default=None)
parser.add_argument('-trgt_z_sz', '--trgt_z_sz', help='target z size', type=int, default=None)

parser.add_argument('-ptch_sz', '--ptch_sz', help='patch size', type=int, default=144)
parser.add_argument('-ptch_z_sz', '--ptch_z_sz', help='patch size', type=int, default=96)

parser.add_argument('-batch_size', '--batch_size', help='batch_size', type=int, default=1)
parser.add_argument('-patches_per_scan', '--patches_per_scan', help='patches_per_scan', type=int, default=100)
parser.add_argument('-no_label_dir', '--no_label_dir', help='dir no_label data', type=str, default='LUNA16')
parser.add_argument('-p_middle', '--p_middle', help='sample in the middle parts', type=float, default=0.5)
parser.add_argument('-mtscale', '--mtscale', help='get a model of multi scales  net', type=int, default=0)
parser.add_argument('-step_nb', '--step_nb', help='training step', type=int, default=100001)
parser.add_argument('-u_v', '--u_v', help='u_v', type=str, default='v')

parser.add_argument('-tsp_lb', '--tsp_lb', help='spacing along x, y and z ', type=str, default='1.4_2.5')
parser.add_argument('-tsp_vs', '--tsp_vs', help='spacing along x, y and z ', type=str, default='1.4_2.5')
parser.add_argument('-tsp_aw', '--tsp_aw', help='spacing along x, y and z ', type=str, default='1.4_2.5')
parser.add_argument('-tsp_lu', '--tsp_lu', help='spacing along x, y and z ', type=str, default='1.4_2.5')
parser.add_argument('-tsp_rc', '--tsp_rc', help='spacing along x, y and z ', type=str, default='1.4_2.5')

parser.add_argument('-low_msk_lb', '--low_msk_lb', help='spacing along x, y and z ', type=int, default=1)
parser.add_argument('-low_msk_vs', '--low_msk_vs', help='spacing along x, y and z ', type=int, default=1)
parser.add_argument('-low_msk_aw', '--low_msk_aw', help='spacing along x, y and z ', type=int, default=1)
parser.add_argument('-low_msk_lu', '--low_msk_lu', help='spacing along x, y and z ', type=int, default=1)
parser.add_argument('-low_msk_rc', '--low_msk_rc', help='spacing along x, y and z ', type=int, default=1)

parser.add_argument('-mot_lb', '--mot_lb', help='multi outpt', type=int, default=0)
parser.add_argument('-mot_vs', '--mot_vs', help='multi outpt', type=int, default=0)
parser.add_argument('-mot_lu', '--mot_lu', help='multi outpt', type=int, default=0)
parser.add_argument('-mot_aw', '--mot_aw', help='multi outpt', type=int, default=0)
parser.add_argument('-mot_rc', '--mot_rc', help='multi outpt', type=int, default=0)

parser.add_argument('-ld_itgt_lb_rc', '--ld_itgt_lb_rc', help='ld_itgt_lb_rc', type=int, default=0)
parser.add_argument('-ld_itgt_vs_rc', '--ld_itgt_vs_rc', help='ld_itgt_vs_rc', type=int, default=0)
parser.add_argument('-ld_itgt_lu_rc', '--ld_itgt_lu_rc', help='ld_itgt_lu_rc', type=int, default=0)
parser.add_argument('-ld_itgt_aw_rc', '--ld_itgt_aw_rc', help='ld_itgt_aw_rc', type=int, default=0)

parser.add_argument('-ld_itgt_lb_rc_name', '--ld_itgt_lb_rc_name', help='ld_itgt_lb_rc_name', type=str, default='None')
parser.add_argument('-ld_itgt_vs_rc_name', '--ld_itgt_vs_rc_name', help='ld_itgt_vs_rc_name', type=str, default='None')
parser.add_argument('-ld_itgt_lu_rc_name', '--ld_itgt_lu_rc_name', help='ld_itgt_lu_rc_name', type=str, default='None')
parser.add_argument('-ld_itgt_aw_rc_name', '--ld_itgt_aw_rc_name', help='ld_itgt_aw_rc_name', type=str, default='None')

parser.add_argument('-ld_rc_name', '--ld_rc_name', help='ld_rc_name', type=str, default='None')
parser.add_argument('-ld_lb_name', '--ld_lb_name', help='ld_lb_name', type=str, default='None')
parser.add_argument('-ld_vs_name', '--ld_vs_name', help='ld_vs_name', type=str, default='None')
parser.add_argument('-ld_lu_name', '--ld_lu_name', help='ld_lu_name', type=str, default='None')
parser.add_argument('-ld_aw_name', '--ld_aw_name', help='ld_aw_name', type=str, default='None')

parser.add_argument('-rc_tr_nb', '--rc_tr_nb', help='rc_tr_nb', type=int, default=400)
parser.add_argument('-lb_tr_nb', '--lb_tr_nb', help='lb_tr_nb', type=int, default=17)
parser.add_argument('-vs_tr_nb', '--vs_tr_nb', help='vs_tr_nb', type=int, default=50)
parser.add_argument('-lu_tr_nb', '--lu_tr_nb', help='lu_tr_nb', type=int, default=0)
parser.add_argument('-aw_tr_nb', '--aw_tr_nb', help='aw_tr_nb', type=int, default=0)

args = parser.parse_args()
