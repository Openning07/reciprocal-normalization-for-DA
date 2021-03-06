import argparse


parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
parser.add_argument('--CDAN', type=str, default='CDAN+E', choices=['CDAN', 'CDAN+E'])
parser.add_argument('--method', type=str, default='BNM', choices=['ENT','NO', 'RN'])
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN"])
parser.add_argument('--dset', type=str, default='office-home', choices=['office31', 'image-clef', 'visda', 'office-home'], help="The dataset or source dataset used")
parser.add_argument('--s_dset_path', type=str, default='./data/office-home/Art.txt', help="The source dataset path list")
parser.add_argument('--t_dset_path', type=str, default='./data/office-home/Clipart.txt', help="The target dataset path list")
parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
parser.add_argument('--print_num', type=int, default=100, help="print num ")
parser.add_argument('--batch_size', type=int, default=36, help="number of batch size ")
parser.add_argument('--num_iterations', type=int, default=20004, help="total iterations")
parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
parser.add_argument('--bottle_dim', type=int, default=256, help="the dim of the bottleneck in the FC")
parser.add_argument('--output_dir', type=str, default='RN', help="output directory of our model (in ../snapshot directory)")
parser.add_argument('--run_num', type=str, default='', help=" the name of output files")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--trade_off', type=float, default=1.0, help="parameter for CDAN")
parser.add_argument('--lambda_method', type=float, default=0.1, help="parameter for method")
parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
parser.add_argument('--show', type=bool, default=True, help="whether show the loss functions")
parser.add_argument('--norm_type', type=str, default='rn', help="the type of normalization")
parser.add_argument('--source', type=str, default='P', help="the name of source domain")
parser.add_argument('--target', type=str, default='RW', help="the name of target domain")
parser.add_argument('--dist', type=str, default='l2', help="the measures of edge strength")
parser.add_argument('--root', type=str, default='', help="the root path of data")
parser.add_argument('--lr_mult', type=float, default=1, help="parameter for rn")
parser.add_argument('--seed', type=int, default=None, help="seed")
parser.add_argument('--ent', action='store_true', default=False, help="whether use the entropy")


args = parser.parse_args()

