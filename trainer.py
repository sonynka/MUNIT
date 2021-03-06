"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os

from logging import getLogger

logger = getLogger()

class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters, opts):
        super(MUNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']

        # Initiate the networks
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        self.loss = {}

        # fix the noise used in sampling
        self.s_a = torch.randn(8, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(8, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(opts.output_base + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
            logger.info(
                '{} - {} - Number of parameters: {}'.format(name, model,
                                                        num_params))

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        c_a, s_a_fake = self.gen_a.encode(x_a)
        c_b, s_b_fake = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(c_a_recon, s_a_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(c_b_recon, s_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss['G/rec_x_A'] = self.recon_criterion(x_a_recon, x_a)
        self.loss['G/rec_x_B'] = self.recon_criterion(x_b_recon, x_b)
        self.loss['G/rec_s_A'] = self.recon_criterion(s_a_recon, s_a)
        self.loss['G/rec_s_B'] = self.recon_criterion(s_b_recon, s_b)
        self.loss['G/rec_c_A'] = self.recon_criterion(c_a_recon, c_a)
        self.loss['G/rec_c_B'] = self.recon_criterion(c_b_recon, c_b)
        self.loss['G/cycrec_x_A'] = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss['G/cycrec_x_B'] = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0

        # GAN loss
        self.loss['G/adv_A'] = self.dis_a.calc_gen_loss(x_ba)
        self.loss['G/adv_B'] = self.dis_b.calc_gen_loss(x_ab)

        # domain-invariant perceptual loss
        self.loss['G/vgg_A'] = self.compute_vgg_loss(self.vgg, x_ba.cuda(), x_b.cuda()) if hyperparameters['vgg_w'] > 0 else 0
        self.loss['G/vgg_B'] = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0

        # total loss
        self.loss['G/total'] = hyperparameters['gan_w'] * self.loss['G/adv_A'] + \
                              hyperparameters['gan_w'] * self.loss['G/adv_B'] + \
                              hyperparameters['recon_x_w'] * self.loss['G/rec_x_A'] + \
                              hyperparameters['recon_s_w'] * self.loss['G/rec_s_A'] + \
                              hyperparameters['recon_c_w'] * self.loss['G/rec_c_A'] + \
                              hyperparameters['recon_x_w'] * self.loss['G/rec_x_B'] + \
                              hyperparameters['recon_s_w'] * self.loss['G/rec_s_B'] + \
                              hyperparameters['recon_c_w'] * self.loss['G/rec_c_B'] + \
                              hyperparameters['recon_x_cyc_w'] * self.loss['G/cycrec_x_A'] + \
                              hyperparameters['recon_x_cyc_w'] * self.loss['G/cycrec_x_B'] + \
                              hyperparameters['vgg_w'] * self.loss['G/vgg_A'] + \
                              hyperparameters['vgg_w'] * self.loss['G/vgg_B']
        self.loss['G/total'].backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
            x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
            x_ba2.append(self.gen_a.decode(c_b, s_a_fake.unsqueeze(0)))
            x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
            x_ab2.append(self.gen_b.decode(c_a, s_b_fake.unsqueeze(0)))

        outputs = {}
        outputs['A/real'] = x_a
        outputs['B/real'] = x_b

        outputs['A/rec'] = torch.cat(x_a_recon)
        outputs['B/rec'] = torch.cat(x_b_recon)

        outputs['A/B_random_style'] = torch.cat(x_ab1)
        outputs['A/B'] = torch.cat(x_ab2)
        outputs['B/A_random_style'] = torch.cat(x_ba1)
        outputs['B/A'] = torch.cat(x_ba2)

        self.train()

        return outputs

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, _ = self.gen_a.encode(x_a)
        c_b, _ = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # D loss
        self.loss['D/A'] = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss['D/B'] = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss['D/total'] = hyperparameters['gan_w'] * self.loss['D/A'] + hyperparameters['gan_w'] * self.loss['D/B']
        self.loss['D/total'].backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            old_lr = self.dis_opt.param_groups[0]['lr']

            self.dis_scheduler.step()

            new_lr = self.dis_opt.param_groups[0]['lr']
            if old_lr != new_lr:
                logger.info('Updated D learning rate: {}'.format(new_lr))
        if self.gen_scheduler is not None:
            old_lr = self.gen_opt.param_groups[0]['lr']
            self.gen_scheduler.step()
            new_lr = self.gen_opt.param_groups[0]['lr']
            if old_lr != new_lr:
                logger.info('Updated G learning rate: {}'.format(new_lr))

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        logger.info('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

        logger.info('Saving snapshots to: {}'.format(snapshot_dir))
