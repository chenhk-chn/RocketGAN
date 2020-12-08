import os
import random
import shutil


class sampler(object):
    def __init__(self, config):
        self.source_dataset = config.source_dataset
        self.source_dir = config.result_dir
        if self.source_dataset == 'duke':
            self.target_dir = config.duke_image_dir
            self.c_dim = 6
        elif self.source_dataset == 'market':
            self.target_dir = config.market_image_dir
            self.c_dim = 8

    def sample(self):
        """ sample for cross dataset transfer"""
        random.seed(0)

        if not os.path.exists(os.path.join(self.source_dir, 'bounding_box_train_s2t_sig')):
            os.mkdir(os.path.join(self.source_dir, 'bounding_box_train_s2t_sig'))

        dirlist = os.listdir(self.target_dir)
        for dir in dirlist:
            dir = os.path.basename(dir)
            if dir[0] == 'T':
                continue
            camid = str(random.randint(0, self.c_dim - 1))
            oldname = os.path.join(self.source_dir, 'bounding_box_train_s2t', dir[:-4] + '_fake_s2t_' + camid + '.jpg')
            if not os.path.exists(oldname):
                continue
            newname = os.path.join(self.source_dir, 'bounding_box_train_s2t_sig',
                                   dir[:-4] + '_fake_s2t_' + camid + '.jpg')
            shutil.copyfile(oldname, newname)
