import argparse
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
from torch.backends import cudnn

from data_loader import get_loader
from solver import Solver


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training.
    cudnn.benchmark = True
    cudnn.deterministic = True
    warnings.filterwarnings('ignore')

    # Init random seed.
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)

    # Init training parameter.
    config.num_iters_decay = config.num_iters // 2
    config.test_iters = config.num_iters
    config.model_save_step = config.num_iters // 6

    # Logs dir.
    suffix = str(config.expanding_cam).replace(' ', '').replace('[', '').replace(']', '')
    config.log_dir = config.log_dir + '_' + suffix
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    # Models dir.
    config.model_save_dir = os.path.join(config.log_dir, 'models')
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)

    # Samples dir.
    config.sample_dir = os.path.join(config.log_dir, 'samples')
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    # Result dir.
    config.result_dir = os.path.join(config.log_dir, 'results')
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Logger.
    suffic = time.localtime(time.time())
    suffic = str(suffic.tm_year) + '_' + str(suffic.tm_mon).zfill(2) + str(suffic.tm_mday).zfill(2) + '_' + str(
        suffic.tm_hour).zfill(2) + str(suffic.tm_min).zfill(2) + '_' + str(suffic.tm_sec).zfill(2)
    sys.stdout = Logger(os.path.join(config.log_dir, 'log_' + suffic + '.txt'))
    print('Software Version:{}.'.format(config.tag))
    print(config)

    # Data loader.
    data_loader = get_loader(config)

    # Solver for training and testing StarGAN.
    solver = Solver(config, data_loader)

    if config.train:
        solver.train()
        solver.test()
    else:
        solver.test()

    if config.need_sample:
        solver.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tag', type=str, default='v1.0')

    # Model configuration.
    parser.add_argument('-e', '--expanding_cam', type=int, nargs='+', default=None, help='cameras to be expand')
    parser.add_argument('-r', '--reuse_encoder_dir', type=int, default=None,
                        help='the dir of encoder for resume, if exits')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=1, help='weight for gradient penalty')

    # Training configuration.
    parser.add_argument('--source_dataset', type=str, default='market', choices=['market', 'duke'])
    parser.add_argument('--target_dataset', type=str, default='duke', choices=['market', 'duke'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=120000, help='number of total iterations for training D')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=1, help='number of D updates per each G update')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--market_image_dir', type=str,
                        default='/home2/haokun/Datasets/market1501/Market-1501-v15.09.15/bounding_box_train')
    parser.add_argument('--duke_image_dir', type=str,
                        default='/home2/haokun/Datasets/dukemtmc-reid/DukeMTMC-reID/bounding_box_train')
    parser.add_argument('--log_dir', type=str, default='./logs/tmp')
    parser.add_argument('--need_sample', type=str2bool, default=False)

    # Step size.
    parser.add_argument('--log_step', type=int, default=1000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    main(config)
