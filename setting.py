'''
Configs for training & testing
Written by Whalechen
'''

import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        default='/data_path/**/**/',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--img_list',
        default='/**/**/**/train_list.txt',
        type=str,
        help='Path for train data list file')
    parser.add_argument(
        '--img_list_test',
        default='/**/**/**/test_list.txt',
        type=str,
        help='Path for test data list file')
    parser.add_argument(
        '--n_seg_classes',
        default=2,
        type=int,
        help="Number of classification and segmentation classes"
    )
    parser.add_argument(
        '--learning_rate', 
        default=0.0001,
        type=float)
    parser.add_argument(
        '--num_workers',
        default=4,
        type=int,
        help='Number of jobs')
    parser.add_argument(
        '--batch_size', default=16, type=int, help='Batch Size')
    parser.add_argument(
        '--phase', default='train', type=str, help='Phase of train or test')
    parser.add_argument(
        '--save_intervals',
        default=10,
        type=int,
        help='Interation for saving model')
    parser.add_argument(
        '--n_epochs',
        default=200,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--input_D',
        default=128,
        type=int,
        help='Input size of depth')
    parser.add_argument(
        '--input_H',
        default=128,
        type=int,
        help='Input size of height')
    parser.add_argument(
        '--input_W',
        default=32,
        type=int,
        help='Input size of width')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help=
        'Path for resume model.')
    parser.add_argument(
        '--new_layer_names',
        default=['avgpool',"fc"],
        type=list,
        help='New layer except for backbone')
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--gpu_id',
        nargs='+',
        type=int,              
        help='Gpu id lists')
    parser.add_argument(
        '--model',
        # default='Unet_only',
        default='Unet',
        # default='resnet2D',
        # default='twopath',
        # default='resnet',
        type=str)
    parser.add_argument(
        '--model_depth',
        default=10,
        type=int,
        help='Depth of resnet3D')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')
    args = parser.parse_args()
    args.save_folder = "./trails/models/**store_path**/{}_{}_{}".format(args.model, args.model_depth, args.learning_rate)
    
    return args
