import os
import re
from glob import glob

from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class ReidDataset(data.Dataset):
    def __init__(self, image_path, transform, target_c=None):
        self.image_path = image_path
        self.transform = transform
        self.target_c = target_c
        self.fnames = []
        self.pids = []
        self.cams = []
        self.dataset = []
        self.preprocess()
        self.num_data = int(len(self.fnames))

    def preprocess(self):
        pattern = re.compile(r'([-\d]+)_c(\d)')
        img_paths = sorted(glob(os.path.join(self.image_path, '*.jpg')))

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            camid -= 1  # index starts from 0
            if self.target_c:
                if camid in self.target_c:
                    camid = self.target_c.index(camid)
                    self.fnames.append(img_path)
                    self.cams.append(camid)
                    pid = pid2label[pid]  # relabel
                    self.pids.append(pid)
                    self.dataset.append((os.path.basename(img_path), pid, camid))
            else:
                self.fnames.append(img_path)
                self.cams.append(camid)
                pid = pid2label[pid]  # relabel
                self.pids.append(pid)
                self.dataset.append((os.path.basename(img_path), pid, camid))

    def __getitem__(self, index):
        image = Image.open(self.fnames[index])
        cam = self.cams[index]
        name = os.path.basename(self.fnames[index])
        pid = self.pids[index]
        return self.transform(image), cam, name, pid

    def __len__(self):
        return self.num_data


def get_loader(config):
    """Build and return a data loader."""
    train_transform = [T.Resize((256, 128)), T.RandomHorizontalFlip(), T.ToTensor(),
                       T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    train_transform = T.Compose(train_transform)

    test_transform = [T.Resize((256, 128)), T.ToTensor(),
                      T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    test_transform = T.Compose(test_transform)

    # Datasets.
    if config.source_dataset in ['duke'] and config.target_dataset in ['market']:
        source_image_dir = config.duke_image_dir
        target_image_dir = config.market_image_dir
    elif config.source_dataset in ['market'] and config.target_dataset in ['duke']:
        source_image_dir = config.market_image_dir
        target_image_dir = config.duke_image_dir
    else:
        assert 'Dataset not support!'
    source_set = ReidDataset(source_image_dir, train_transform)
    target_set = ReidDataset(target_image_dir, train_transform, config.expanding_cam)
    test_set = ReidDataset(source_image_dir, test_transform)

    # Dataloader.
    source_loader = data.DataLoader(dataset=source_set, batch_size=config.batch_size,
                                    num_workers=config.num_workers, shuffle=True, pin_memory=True, drop_last=True)

    target_loader = data.DataLoader(dataset=target_set, batch_size=config.batch_size,
                                    num_workers=config.num_workers, shuffle=True, pin_memory=True, drop_last=True)

    test_loader = data.DataLoader(dataset=test_set, batch_size=config.batch_size, num_workers=config.num_workers,
                                  shuffle=False, pin_memory=True, drop_last=False)

    return {'source_loader': source_loader, 'target_loader': target_loader, 'test_loader': test_loader}
