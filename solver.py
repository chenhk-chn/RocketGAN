import datetime
import os
import time

import numpy as np
import torch.nn.functional as F
import tqdm
from torch.autograd import grad
from torchvision.utils import save_image

from model import *
from sampler import sampler


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, config, dataloader):
        """Initialize configurations."""
        # Data loader.
        self.source_loader = dataloader['source_loader']
        self.target_loader = dataloader['target_loader']
        self.test_loader = dataloader['test_loader']
        self.source_dataset = config.source_dataset
        self.target_dataset = config.target_dataset
        self.sampler = sampler(config)

        # Reuse model configurations.
        self.reuse_encoder = config.reuse_encoder_dir is not None
        self.reuse_encoder_dir = config.reuse_encoder_dir
        self.expanding_cam = config.expanding_cam

        # Training configurations.
        self.c_dim = len(self.expanding_cam)  # how many camera does target set have.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model.
        self.Encoder = Encoder(config.g_conv_dim, config.g_repeat_num)  # 2 for mask vector.
        if self.reuse_encoder:
            self.Decoder = Decoder(c_dim=self.c_dim)
            self.D = Discriminator(self.c_dim)
        else:
            self.Decoder = Decoder(c_dim=self.c_dim + 1)
            self.D = Discriminator(self.c_dim + 1)

        self.print_network(self.Encoder, 'Encoder')
        self.print_network(self.Decoder, 'Decoder')
        self.print_network(self.D, 'Discriminator')

        self.Encoder = self.Encoder.to(self.device)
        self.Decoder = self.Decoder.to(self.device)
        self.D = self.D.to(self.device)

        # Build optimizer.
        self.encoder_optimizer = torch.optim.Adam(self.Encoder.parameters(), self.g_lr, (0.5, 0.999))
        self.decoder_optimizer = torch.optim.Adam(self.Decoder.parameters(), self.g_lr, (0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, (0.5, 0.999))

        # Resume.
        if self.reuse_encoder:
            ckpt = torch.load(self.reuse_encoder_dir)
            self.Encoder.load_state_dict(ckpt['Encoder'])
            for k, v in self.Encoder.named_parameters():
                v.requires_grad = False
            print('Successfully load Encoder model form {}.'.format(self.reuse_encoder_dir))

    def generator(self, x, target_c, origin_c=None):
        feature = self.Encoder(x)
        if self.reuse_encoder:
            feature = feature.detach()
        img_trg = self.Decoder(feature, target_c)
        if origin_c is not None:
            img_org = self.Decoder(feature, origin_c)
            return img_trg, img_org
        return img_trg

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, checkpoint):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from {}...'.format(checkpoint))
        ckpt = torch.load(checkpoint)
        self.Encoder.load_state_dict(ckpt['Encoder'])
        self.Decoder.load_state_dict(ckpt['Decoder'])
        self.D.load_state_dict(ckpt['Discriminator'])

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.encoder_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.decoder_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def grad_penalty(self, net, x, coeff=10):
        """Calculate R1 regularization gradient penalty"""
        x.requires_grad = True
        real_predict, _ = net(x)
        gradients = grad(outputs=real_predict.mean(), inputs=x, create_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = (coeff / 2) * ((gradients.norm(2, dim=1) ** 2).mean())
        return gradient_penalty

    def adv_loss(self, x, real=True):
        target = torch.ones(x.size()).type_as(x) if real else torch.zeros(x.size()).type_as(x)
        return nn.MSELoss(reduction='none')(x, target)

    def label2onehot(self, labels):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        if self.reuse_encoder:
            out = torch.zeros(batch_size, self.c_dim)
        else:
            out = torch.zeros(batch_size, self.c_dim + 1)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.cross_entropy(logit, target)

    def opt_d_loss(self, d_loss):
        """Optimization for Discriminator"""
        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

    def opt_g_loss(self, g_loss):
        """Optimization for Generator"""
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        g_loss.backward()

        if not self.reuse_encoder:
            self.encoder_optimizer.step()
        self.decoder_optimizer.step()

    def denorm(self, img):
        """convert img from range [-1,1] to [0,1]"""
        img = (img + 1) / 2
        return img.clamp_(0, 1)

    def train(self):
        """Train StarGAN with multiple datasets."""
        # Data iterators.
        source_iter = iter(self.source_loader)
        target_iter = iter(self.target_loader)

        # Fetch fixed inputs for debugging.
        x_fixed, c_org, _, _ = next(source_iter)  # c for camera num; label for one-hot vector
        x_fixed = x_fixed.to(self.device)
        label_fixed_list = []
        for i in range(self.c_dim):
            c_trg = torch.ones(x_fixed.size(0)).type_as(c_org) * i
            label_trg = self.label2onehot(c_trg)
            label_trg = label_trg.to(self.device)
            label_fixed_list.append(label_trg)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(self.num_iters):
            cyclemod = ['target'] if self.reuse_encoder else ['source', 'target']
            for dataset in cyclemod:
                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                # Fetch real images and labels.
                data_iter = source_iter if dataset == 'source' else target_iter

                try:
                    x_real, c_org, _, _ = next(data_iter)
                except:
                    if dataset == 'source':
                        source_iter = iter(self.source_loader)
                        x_real, c_org, _, _ = next(source_iter)
                    elif dataset == 'target':
                        target_iter = iter(self.target_loader)
                        x_real, c_org, _, _ = next(target_iter)
                if dataset == 'source':
                    c_org = torch.ones(c_org.size()).type_as(c_org) * self.c_dim

                # Generate target domain labels randomly.
                rand_idx = torch.randperm(c_org.size(0))
                c_trg = c_org[rand_idx]

                x_real = x_real.to(self.device)  # Input images.
                c_org = c_org.to(self.device)  # Original domain labels.
                c_trg = c_trg.to(self.device)  # Target domain labels.
                label_org = self.label2onehot(c_org)
                label_trg = self.label2onehot(c_trg)
                label_org = label_org.to(self.device)  # Labels for computing classification loss.
                label_trg = label_trg.to(self.device)  # Labels for computing classification loss.

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                out_real, out_cls = self.D(x_real)
                d_loss_cls = self.classification_loss(out_cls, c_org)

                # Compute loss with fake images.
                x_fake = self.generator(x_real, label_trg)
                out_fake, _ = self.D(x_fake.detach())
                d_loss_adv = self.adv_loss(out_real, True).mean() + self.adv_loss(out_fake, False).mean()

                # Compute loss for gradient penalty.
                d_loss_gp = self.grad_penalty(self.D, x_real)

                # Backward and optimize.
                d_loss = d_loss_adv + self.lambda_gp * d_loss_gp + self.lambda_cls * d_loss_cls
                self.opt_d_loss(d_loss)

                # Logging.
                d_loss_dic = {'D/loss_adv': d_loss_adv.item(), 'D/loss_gp': d_loss_gp.item(),
                              'D/loss_cls': d_loss_cls.item()}

                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                if (i + 1) % self.n_critic == 0:
                    # Original-to-target domain.
                    x_fake, x_reconst = self.generator(x_real, label_trg, label_org)
                    out_fake, out_cls = self.D(x_fake)
                    g_loss_adv = self.adv_loss(out_fake, True).mean()
                    g_loss_cls = self.classification_loss(out_cls, c_trg)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_adv + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    self.opt_g_loss(g_loss)

                    # Logging.
                    g_loss_dic = {'G/loss_adv': g_loss_adv.item(), 'G/loss_rec': g_loss_rec.item(),
                                  'G/loss_cls': g_loss_cls.item()}

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training info.
            if (i + 1) % self.log_step == 0:
                print('\n')
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]\n".format(et, i + 1, self.num_iters)
                for tag, value in d_loss_dic.items():
                    log += "{}:{:.4f}   ".format(tag, value)
                log += '\n'
                for tag, value in g_loss_dic.items():
                    log += "{}: {:.4f}   ".format(tag, value)
                print(log)

                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for label_fixed in label_fixed_list:
                        x_fake_list.append(self.generator(x_fixed, label_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i + 1))
                    save_image(x_concat.data.cpu(), sample_path, nrow=1, padding=0, range=(-1, 1), normalize=True)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                suffix = str(self.expanding_cam).replace(' ', '').replace('[', '').replace(']', '')
                torch.save({
                    'Encoder': self.Encoder.state_dict(),
                    'Decoder': self.Decoder.state_dict(),
                    'Discriminator': self.D.state_dict()
                }, os.path.join(self.model_save_dir, 'Checkpoint_{}_{}.model'.format(suffix, i + 1))
                )
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Translate images using StarGAN trained on multiple datasets."""
        # Load the trained generator.
        with tqdm.tqdm(total=len(self.test_loader)) as pbar:
            with torch.no_grad():
                for i, (x_real, c_org, name, _) in enumerate(self.test_loader):
                    x_real = x_real.to(self.device)

                    # Translate images.
                    c_fixed_num = self.c_dim
                    x_fake_list = [x_real]
                    for jj in range(c_fixed_num):
                        label_trg = self.label2onehot(torch.ones(c_org.size()).type_as(c_org) * jj)
                        label_trg = label_trg.to(self.device)
                        x_fake_list.append(self.generator(x_real, label_trg))

                    for jj in range(len(x_fake_list)):
                        image_list = x_fake_list[jj]
                        for kk in range(len(name)):
                            if jj == 0:
                                continue
                            else:
                                save_name = name[kk][:-4] + '_fake_s2t_' + str(self.expanding_cam[jj - 1]) + '.jpg'
                            save_path = os.path.join(self.result_dir, save_name)
                            save_image(image_list[kk].data, save_path, nrow=1, padding=0, range=(-1, 1), normalize=True)
                    pbar.update(1)

    def sample(self):
        self.sampler.sample()
