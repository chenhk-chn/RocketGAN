import argparse
import os
import pickle
import random
import sys
import time
import warnings

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.backends import cudnn
from torch.utils import data
from torchvision import transforms as T

from data_loader import ReidDataset


class Logger(object):
    """Logger the current print"""

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RAPDataset(data.Dataset):
    def __init__(self, image_path, annotation_path, transform, is_train=True):
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.transform = transform
        self.is_train = is_train
        self.fnames = []
        self.viewpoints = []  # 0 for front,1 for back,2 for side
        self.preprocess()
        self.num_data = int(len(self.fnames))

    def preprocess(self):
        file = sio.loadmat(self.annotation_path)
        front = file['RAP_annotation'][0][0][1][:, 51].tolist()
        back = file['RAP_annotation'][0][0][1][:, 52].tolist()
        # sideleft = file['RAP_annotation'][0][0][1][:, 53].tolist()
        # sideright = file['RAP_annotation'][0][0][1][:, 53].tolist()
        fnames = file['RAP_annotation'][0][0][5].tolist()

        idx = 0  # use split 0
        trainval_idx = (file['RAP_annotation'][0][0][0][idx][0][0][0][0][0, :] - 1).tolist()
        test_idx = (file['RAP_annotation'][0][0][0][idx][0][0][0][1][0, :] - 1).tolist()

        idx = trainval_idx if self.is_train else test_idx

        for i in idx:
            self.fnames.append(os.path.join(self.image_path, fnames[i][0][0]))
            if front[i]:
                viewpoint = 0
            elif back[i]:
                viewpoint = 1
            else:
                viewpoint = 2
            self.viewpoints.append(viewpoint)

    def __getitem__(self, index):
        image = Image.open(self.fnames[index])
        viewpoint = self.viewpoints[index]
        name = os.path.basename(self.fnames[index])
        return self.transform(image), viewpoint, name

    def __len__(self):
        return self.num_data


class ViewpointClassifier(nn.Module):
    """Classifier model for viewpoint."""

    def __init__(self, out_dim=3):
        super(ViewpointClassifier, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        base = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2,
                resnet.layer3, resnet.layer4, resnet.avgpool]
        self.main = nn.Sequential(*base)
        self.cls = nn.Linear(2048, out_dim)

    def forward(self, x):
        h = self.main(x).squeeze(2).squeeze(2)
        h = self.cls(h)
        return h


class Solver(object):
    def __init__(self, config, data_loader):
        # training details
        self.lr = config.base_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.epochs = config.num_epochs

        # dataloader.
        self.data_loader = data_loader['train_loader']
        self.raptest_loader = data_loader['raptest_loader']
        self.market_loader = data_loader['market_loader']
        self.duke_loader = data_loader['duke_loader']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.net = ViewpointClassifier()
        self.net_optimizer = torch.optim.Adam(self.net.parameters(), self.lr, [self.beta1, self.beta2])
        self.net.to(self.device)

        # step.
        self.log_dir = config.log_dir
        self.model_save_step = config.model_save_step
        self.model_save_dir = os.path.join(self.log_dir, 'models')

    def warmup_backbone_lr(self, epoch):
        """Warm up resnet learning rate."""
        train_epoches = self.epochs
        if (epoch + 1) <= train_epoches / 12:
            resnet_lr = 12 * (epoch + 1) * self.lr / train_epoches
        elif train_epoches / 12 < (epoch + 1) <= (train_epoches / 3):
            resnet_lr = self.lr
        elif (train_epoches / 3) < (epoch + 1) <= (7 * train_epoches / 12):
            resnet_lr = self.lr / 10
        elif (7 * train_epoches / 12) < (epoch + 1) <= train_epoches:
            resnet_lr = self.lr / 100
        else:
            resnet_lr = self.lr
        for param_group in self.net_optimizer.param_groups:
            param_group['lr'] = resnet_lr
        print('Current Backbone learning rates:{}.'.format(resnet_lr))

    def train(self):
        print('Start training...')
        loss_avg = AverageMeter()
        for epoch in range(self.epochs):
            self.warmup_backbone_lr(epoch)
            for i, (imgs, viewpoints, names) in enumerate(self.data_loader):
                imgs = imgs.to(self.device)
                viewpoints = viewpoints.to(self.device)
                out_cls = self.net(imgs)
                loss = F.cross_entropy(out_cls, viewpoints)
                loss_avg.update(loss.item())
                self.net_optimizer.zero_grad()
                loss.backward()
                self.net_optimizer.step()
            print('[{}/{}]:Current loss:{}'.format(epoch + 1, self.epochs, loss_avg.avg))
            loss_avg.reset()

            self.test()

            if (epoch + 1) % self.model_save_step == 0:
                cls_path = os.path.join(self.model_save_dir, '{}-ViewCls.ckpt'.format(epoch + 1))
                torch.save(self.net.state_dict(), cls_path)
                print('Saved model checkpoints into {}...\n'.format(self.model_save_dir))

        self.annotation()

    def test(self):
        cls = nn.Softmax(dim=1)
        correct = 0
        total = 0
        for i, (imgs, viewpoints, names) in enumerate(self.raptest_loader):
            imgs = imgs.to(self.device)
            viewpoints = viewpoints.to(self.device)
            out_cls = self.net(imgs)
            correct += (viewpoints == cls(out_cls).argmax(dim=1)).sum()
            total += imgs.size(0)
        print('Current correct rate is {}.\n'.format(int(correct) / total))

    def annotation(self):
        cls = nn.Softmax(dim=1)
        for dataset in ['market', 'duke']:
            anno_dict = {}
            dataloader = self.market_loader if dataset == 'market' else self.duke_loader
            for i, (imgs, _, names, _) in enumerate(dataloader):
                imgs = imgs.to(self.device)
                viewpoints = cls(self.net(imgs)).argmax(dim=1)
                for i, name in enumerate(names):
                    anno_dict[name] = viewpoints[i]
            if not os.path.exists(os.path.join(self.log_dir, 'anno')):
                os.mkdir(os.path.join(self.log_dir, 'anno'))
            path = os.path.join(self.log_dir, 'anno', dataset + '.pkl')

            with open(path, 'wb') as fp:
                pickle.dump(anno_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


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

    # Logger.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(os.path.join(config.log_dir, 'models')):
        os.mkdir(os.path.join(config.log_dir, 'models'))
    suffic = time.localtime(time.time())
    suffic = str(suffic.tm_year) + '_' + str(suffic.tm_mon).zfill(2) + str(suffic.tm_mday).zfill(2) + '_' + str(
        suffic.tm_hour).zfill(2) + str(suffic.tm_min).zfill(2) + '_' + str(suffic.tm_sec).zfill(2)
    sys.stdout = Logger(os.path.join(config.log_dir, 'log_' + suffic + '.txt'))
    print('Software Version:{}.'.format(config.tag))
    print(config)

    # Dataloader.
    train_transform = [T.Resize((288, 144)), T.RandomCrop(256, 128), T.RandomHorizontalFlip(), T.ToTensor(),
                       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    train_transform = T.Compose(train_transform)
    dataset = RAPDataset(config.rap_image_dir, config.rap_anno_dir, train_transform, is_train=True)
    train_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                                   shuffle=True, pin_memory=True, drop_last=True)

    test_transform = [T.Resize((256, 128)), T.ToTensor(),
                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    test_transform = T.Compose(test_transform)
    raptest_dataset = RAPDataset(config.rap_image_dir, config.rap_anno_dir, test_transform, is_train=False)
    raptest_loader = data.DataLoader(dataset=raptest_dataset, batch_size=config.batch_size,
                                     num_workers=config.num_workers, shuffle=False, pin_memory=True, drop_last=False)
    market_dataset = ReidDataset(config.market_image_dir, test_transform)
    duke_dataset = ReidDataset(config.duke_image_dir, test_transform)
    market_loader = data.DataLoader(dataset=market_dataset, batch_size=config.batch_size,
                                    num_workers=config.num_workers,
                                    shuffle=False, pin_memory=True, drop_last=False)
    duke_loader = data.DataLoader(dataset=duke_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                                  shuffle=False, pin_memory=True, drop_last=False)

    dataloadr = {'train_loader': train_loader, 'raptest_loader': raptest_loader, 'market_loader': market_loader,
                 'duke_loader': duke_loader}

    # Solver.
    solver = Solver(config, dataloadr)
    solver.train()
    solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tag', type=str, default='v1.0')

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
    parser.add_argument('--num_epochs', type=int, default=60, help='number of total epochsD')
    parser.add_argument('--base_lr', type=float, default=0.00035, help='learning rate for backbone')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='learning rate for D')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=120000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)

    # Directories.
    parser.add_argument('--market_image_dir', type=str,
                        default='/home2/haokun/Datasets/market1501/Market-1501-v15.09.15/bounding_box_train')
    parser.add_argument('--duke_image_dir', type=str,
                        default='/home2/haokun/Datasets/dukemtmc-reid/DukeMTMC-reID/bounding_box_train')
    parser.add_argument('--rap_image_dir', type=str, default='/home2/haokun/Datasets/RAP/dataset')
    parser.add_argument('--rap_anno_dir', type=str, default='/home2/haokun/Datasets/RAP/annotation/RAP_annotation.mat')

    parser.add_argument('--log_dir', type=str, default='./logs')

    # Step size.
    parser.add_argument('--log_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=20)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    main(config)
