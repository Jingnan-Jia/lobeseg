import argparse

parser = argparse.ArgumentParser(prog="MS-Net",
                                 description='Multi-scale Multi-task Semi-supervised network for Segmentation',
                                 epilog="If you need any help, contact jiajingnan2222@gmail.com")

parser.add_argument('-model_names', '--model_names', help='model names', type=str, default='net_only_lobe-net_only_vessel-net_no_label')

parser.add_argument('-adaptive_lr', '--adaptive_lr', help='adaptive learning rate', type=int, default=1)
parser.add_argument('-attention', '--attention', help='attention loss', type=int, default=0)
parser.add_argument('-fn', '--feature_number', help='Number of initial of conv channels', type=int, default=16)
parser.add_argument('-bn', '--batch_norm', help='Set Batch Normalization', type=int, default=1)
parser.add_argument('-dr', '--dropout', help='Set Dropout', type=int, default=1)
parser.add_argument('-ptch_sz', '--ptch_sz', help='patch size', type=int, default=144)
parser.add_argument('-ptch_z_sz', '--ptch_z_sz', help='patch size', type=int, default=96)
parser.add_argument('-batch_size', '--batch_size', help='batch_size', type=int, default=1)
parser.add_argument('-patches_per_scan', '--patches_per_scan', help='patches_per_scan', type=int, default=100)
parser.add_argument('-no_label_dir', '--no_label_dir', help='dir no_label data', type=str, default='LUNA16')
parser.add_argument('-p_middle', '--p_middle', help='sample in the middle parts', type=float, default=0.5)
parser.add_argument('-step_nb', '--step_nb', help='training step', type=int, default=100001)
parser.add_argument('-u_v', '--u_v', help='u_v', type=str, default='v')
parser.add_argument('-fat', '--fat', help='focus_alt_train', type=int, default=0)


parser.add_argument('-lr_lb', '--lr_lb', help='learning rate for lobe segmentation', type=float, default=0.0001)
parser.add_argument('-lr_vs', '--lr_vs', help='learning rate for vessel segmentation', type=float, default=0.00001)
parser.add_argument('-lr_lu', '--lr_lu', help='learning rate for lung segmentation', type=float, default=0.00001)
parser.add_argument('-lr_aw', '--lr_aw', help='learning rate for airway segmentation', type=float, default=0.00001)
parser.add_argument('-lr_rc', '--lr_rc', help='learning rate for reconstruction', type=float, default=0.00001)

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

parser.add_argument('-tsp_lb', '--tsp_lb', help='spacing along (x, y) and z ', type=str, default='1.4_2.5')
parser.add_argument('-tsp_vs', '--tsp_vs', help='spacing along (x, y) and z ', type=str, default='1.4_2.5')
parser.add_argument('-tsp_aw', '--tsp_aw', help='spacing along (x, y) and z ', type=str, default='1.4_2.5')
parser.add_argument('-tsp_lu', '--tsp_lu', help='spacing along (x, y) and z ', type=str, default='1.4_2.5')
parser.add_argument('-tsp_rc', '--tsp_rc', help='spacing along (x, y) and z ', type=str, default='1.4_2.5')

parser.add_argument('-tsz_lb', '--tsz_lb', help='target size along (x, y) and z ', type=str, default='0_0')
parser.add_argument('-tsz_vs', '--tsz_vs', help='target size along (x, y) and z ', type=str, default='0_0')
parser.add_argument('-tsz_aw', '--tsz_aw', help='target size along (x, y) and z ', type=str, default='0_0')
parser.add_argument('-tsz_lu', '--tsz_lu', help='target size along (x, y) and z ', type=str, default='0_0')
parser.add_argument('-tsz_rc', '--tsz_rc', help='target size along (x, y) and z ', type=str, default='0_0')

io_choices = ["2_in_2_out", "1_in_low_1_out_low", "1_in_hgh_1_out_hgh", "2_in_1_out_low", "2_in_1_out_hgh"]
parser.add_argument('-lb_io', '--lb_io', help='input outpt setting', type=str, choices=io_choices, default="2_in_1_out_low")
parser.add_argument('-vs_io', '--vs_io', help='input outpt setting', type=str, choices=io_choices, default="2_in_1_out_low")
parser.add_argument('-lu_io', '--lu_io', help='input outpt setting', type=str, choices=io_choices, default="2_in_1_out_low")
parser.add_argument('-aw_io', '--aw_io', help='input outpt setting', type=str, choices=io_choices, default="2_in_1_out_low")
parser.add_argument('-rc_io', '--rc_io', help='input outpt setting', type=str, choices=io_choices, default="2_in_1_out_low")

parser.add_argument('-rc_tr_nb', '--rc_tr_nb', help='rc_tr_nb', type=int, default=400)
parser.add_argument('-lb_tr_nb', '--lb_tr_nb', help='lb_tr_nb', type=int, default=17)
parser.add_argument('-vs_tr_nb', '--vs_tr_nb', help='vs_tr_nb', type=int, default=50)
parser.add_argument('-lu_tr_nb', '--lu_tr_nb', help='lu_tr_nb', type=int, default=0)
parser.add_argument('-aw_tr_nb', '--aw_tr_nb', help='aw_tr_nb', type=int, default=0)

parser.add_argument('-ld_itgt_lb_rc_name', '--ld_itgt_lb_rc_name', help='ld_itgt_lb_rc_name', type=str, default='None')
parser.add_argument('-ld_itgt_vs_rc_name', '--ld_itgt_vs_rc_name', help='ld_itgt_vs_rc_name', type=str, default='None')
parser.add_argument('-ld_itgt_lu_rc_name', '--ld_itgt_lu_rc_name', help='ld_itgt_lu_rc_name', type=str, default='None')
parser.add_argument('-ld_itgt_aw_rc_name', '--ld_itgt_aw_rc_name', help='ld_itgt_aw_rc_name', type=str, default='None')

parser.add_argument('-ld_rc_name', '--ld_rc_name', help='ld_rc_name', type=str, default='None')
parser.add_argument('-ld_lb_name', '--ld_lb_name', help='ld_lb_name', type=str, default='None')
parser.add_argument('-ld_vs_name', '--ld_vs_name', help='ld_vs_name', type=str, default='None')
parser.add_argument('-ld_lu_name', '--ld_lu_name', help='ld_lu_name', type=str, default='None')
parser.add_argument('-ld_aw_name', '--ld_aw_name', help='ld_aw_name', type=str, default='None')

args = parser.parse_args()
