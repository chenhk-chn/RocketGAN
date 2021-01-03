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
import tqdm


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
        if self.source_dataset == 'duke':
            self.source_c_dim = 8
        else:
            self.source_c_dim = 6
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
        self.Decoder = Decoder(c_dim=self.source_c_dim + self.c_dim + 2)
        self.D = Discriminator(c_dim=self.source_c_dim+self.c_dim)

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
            print('Successfully load Encoder model from {}.'.format(self.reuse_encoder_dir))

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

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=weight, retain_graph=True, create_graph=True,
                                   only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
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
        """Train StarGAN cross multi dataset."""
        # Fetch fixed inputs for debugging.
        source_iter = iter(self.source_loader)
        target_iter = iter(self.target_loader)
        x_fixed, c_org, _, _ = next(source_iter)
        x_fixed = x_fixed.to(self.device)

        c_target_list = []
        for i in range(self.c_dim):
            c_trg = self.label2onehot(torch.ones(x_fixed.size(0)) * i, self.c_dim)
            c_target_list.append(c_trg.to(self.device))
        mask_target = self.label2onehot(torch.ones(x_fixed.size(0)), 2).to(self.device)  # Mask vector: [0, 1].
        zero_source = torch.zeros(x_fixed.size(0), self.source_c_dim).to(self.device)  # Zero vector for Source dataset.

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(self.num_iters):
            for dataset in ['source', 'target']:

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                data_iter = source_iter if dataset == 'source' else target_iter

                # Fetch real images and labels.
                try:
                    x_real, c_org, _, _ = next(data_iter)
                except:
                    if dataset == 'source':
                        source_iter = iter(self.source_loader)
                        x_real, c_org, _, _ = next(source_iter)
                    elif dataset == 'target':
                        target_iter = iter(self.target_loader)
                        x_real, c_org, _, _ = next(target_iter)

                # Generate target domain labels randomly.
                rand_idx = torch.randperm(c_org.size(0))
                c_trg = c_org[rand_idx]

                if dataset == 'source':
                    label_org = self.label2onehot(c_org, self.source_c_dim)
                    label_trg = self.label2onehot(c_trg, self.source_c_dim)
                    zero = torch.zeros(x_real.size(0), self.c_dim)
                    mask = self.label2onehot(torch.zeros(x_real.size(0)), 2)
                    label_org = torch.cat([label_org, zero, mask], dim=1)
                    label_trg = torch.cat([label_trg, zero, mask], dim=1)
                elif dataset == 'target':
                    label_org = self.label2onehot(c_org, self.c_dim)
                    label_trg = self.label2onehot(c_trg, self.c_dim)
                    zero = torch.zeros(x_real.size(0), self.source_c_dim)
                    mask = self.label2onehot(torch.ones(x_real.size(0)), 2)
                    label_org = torch.cat([zero, label_org, mask], dim=1)
                    label_trg = torch.cat([zero, label_trg, mask], dim=1)

                x_real = x_real.to(self.device)  # Input images.
                c_org = c_org.to(self.device)  # Original domain labels.
                c_trg = c_trg.to(self.device)  # Target domain labels.
                label_org = label_org.to(self.device)  # Labels for computing classification loss.
                label_trg = label_trg.to(self.device)  # Labels for computing classification loss.

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                out_src, out_cls = self.D(x_real)
                d_loss_real = - torch.mean(out_src)
                out_cls = out_cls[:, :self.source_c_dim] if dataset == 'source' else out_cls[:, self.source_c_dim:]
                d_loss_cls = self.classification_loss(out_cls, c_org)

                # Compute loss with fake images.
                x_fake = self.generator(x_real, label_trg)
                out_src, _ = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp + self.lambda_cls * d_loss_cls

                # Logging.
                d_loss_log = {'D/loss_real': d_loss_real.item(), 'D/loss_fake': d_loss_fake.item(),
                              'D/loss_adv': d_loss_real.item() + d_loss_fake.item(), 'D/loss_gp': d_loss_gp.item(),
                              'D/loss_cls': d_loss_cls.item()}

                self.opt_d_loss(d_loss)

                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                if (i + 1) % self.n_critic == 0:
                    # Original-to-target domain.
                    x_fake, x_reconst = self.generator(x_real, label_trg, label_org)
                    out_src, out_cls = self.D(x_fake)
                    g_loss_fake = - torch.mean(out_src)
                    out_cls = out_cls[:, :self.source_c_dim] if dataset == 'source' else out_cls[:, self.source_c_dim:]
                    g_loss_cls = self.classification_loss(out_cls, c_trg)

                    # Target-to-original domain.
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Logging.
                    g_loss_log = {'G/loss_fake': g_loss_fake.item(), 'G/loss_rec': g_loss_rec.item(),
                                  'G/loss_cls': g_loss_cls.item()}

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls

                    self.opt_g_loss(g_loss)

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information and translate fixed images for debugging.
            if (i + 1) % self.log_step == 0:
                print('\n')
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]\n".format(et, i + 1, self.num_iters)
                log += 'D loss: '
                for tag, value in d_loss_log.items():
                    log += "{}: {:.4f}  ".format(tag, value)
                log += '\nG Loss: '
                for tag, value in g_loss_log.items():
                    log += "{}: {:.4f}  ".format(tag, value)
                print(log)

                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_target_list:
                        c_trg = torch.cat([zero_source, c_fixed, mask_target], dim=1)
                        x_fake_list.append(self.generator(x_fixed, c_trg))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i + 1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...\n'.format(sample_path))

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
            if (i + 1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                if (i + 1) % self.log_step == 0:
                    print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Set data loader.
        data_loader = self.test_loader
        pbar = tqdm.tqdm(range(len(data_loader)))

        with torch.no_grad():
            for i, (x_real, c_org, name, _) in enumerate(data_loader):
                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)

                c_target_list = []
                for ii in range(self.c_dim):
                    c_trg = self.label2onehot(torch.ones(x_real.size(0)) * ii, self.c_dim)
                    c_target_list.append(c_trg.to(self.device))
                mask_target = self.label2onehot(torch.ones(x_real.size(0)), 2).to(self.device)  # Mask vector: [0, 1].
                zero_source = torch.zeros(x_real.size(0), self.source_c_dim).to(self.device)  # Zero vector for Source.

                # Translate images.
                x_fake_list = [x_real]
                for c_target in c_target_list:
                    c_trg = torch.cat([zero_source, c_target, mask_target], dim=1)
                    x_fake_list.append(self.generator(x_real, c_trg))

                for jj in range(len(x_fake_list)):
                    image_list = x_fake_list[jj]
                    for kk in range(len(name)):
                        if jj == 0:
                            continue
                        else:
                            # save_name = name_real[kk][:-4] + '_fake_' + str(c_org[kk] + 1) + 'to' + str(jj) + '.jpg'
                            save_name = name[kk][:-4] + '_s2t_' + str(self.expanding_cam[jj - 1]) + '.jpg'
                        save_path = os.path.join(self.result_dir, save_name)
                        save_image(self.denorm(image_list[kk].data), save_path, nrow=1, padding=0)
                state_msg = '[{}/{}]:Test images saved into {}.'.format(i + 1, len(data_loader), self.result_dir)
                pbar.set_description(state_msg)
                pbar.update(1)

    def sample(self):
        self.sampler.sample()
