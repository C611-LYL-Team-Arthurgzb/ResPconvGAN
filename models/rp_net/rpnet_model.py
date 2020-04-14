import torch
from torch.nn import functional as F
import util.util as util
from models import networks
from models.rp_net.base_model import BaseModel
import time
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
from models.modules.net import PConvUNet

class NetModel(BaseModel):
    def name(self):
        return 'NetModel'




    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D', 'style', 'content', 'tv','hole','valid']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.opt.show_flow:
            self.visual_names = ['real_A', 'fake_B', 'real_B', 'flow_srcs']
        else:
            self.visual_names = ['real_input', 'fake_B', 'real_GTimg','mask_global','output_comp']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']



        #
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
        #                               opt.which_model_netG, opt,  opt.norm, opt.use_spectral_norm_G, opt.init_type, self.gpu_ids, opt.init_gain)
        self.netG = PConvUNet().to(self.device)
        print(self.netG)
        if self.isTrain:
            use_sigmoid = False
            if opt.gan_type == 'vanilla':
                use_sigmoid = True  # only vanilla GAN using BCECriterion
            # don't use cGAN
            self.netD = networks.define_D(opt.input_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.use_spectral_norm_D, opt.init_type, self.gpu_ids, opt.init_gain)

        # add style extractor
        self.vgg16_extractor = util.VGG16FeatureExtractor().to(self.gpu_ids[0])
        self.vgg16_extractor = torch.nn.DataParallel(self.vgg16_extractor, self.gpu_ids)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(gan_type=opt.gan_type).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL1_mask = networks.Discounted_L1(opt).to(self.device) # make weights/buffers transfer to the correct device
            # VGG loss
            self.criterionL2_style_loss = torch.nn.MSELoss()
            self.criterionL2_content_loss = torch.nn.MSELoss()
            # TV loss
            self.tv_criterion = networks.TVLoss(self.opt.tv_weight)

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            if self.opt.gan_type == 'wgan_gp':
                opt.beta1 = 0
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                    lr=opt.lr, betas=(opt.beta1, 0.9))
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.9))
            else:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
 
        self.print_networks(opt.verbose)

    def set_input(self, input):
        self.image_paths = input['input_img_paths']
        real_input = input['input_img'].to(self.device)
        real_GTimg = input['GTimg'].to(self.device)
        mask_global = input['mask'].to(self.device)



        self.opt.mask_type = 'random'
        self.opt.mask_sub_type = 'island'

        self.real_input = real_input
        self.real_GTimg = real_GTimg
        self.mask_global = mask_global

    






    def forward(self):

        self.fake_B, self.mask_B = self.netG(self.real_input, self.mask_global)  # G(A)





    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        fake_B = self.fake_B
        # Real
        real_GTimg = self.real_GTimg # GroundTruth


        self.pred_fake = self.netD(fake_B.detach())
        self.pred_real = self.netD(real_GTimg)
        
        # fake_AB = torch.cat((self.real_input, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        # pred_fake = self.netD(fake_AB.detach())
        # self.loss_D_fake = self.criterionGAN(pred_fake, False)
        #
        # fake_AB = torch.cat((self.real_input, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        # pred_fake = self.netD(fake_AB.detach())
        # self.loss_D_fake = self.criterionGAN(pred_fake, False)
        #
        # # Real
        # real_AB = torch.cat((self.real_input, self.real_GTimg), 1)
        # pred_real = self.netD(real_AB)
        # self.loss_D_real = self.criterionGAN(pred_real, True)
        # # combine loss and calculate gradients
        # self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        # self.loss_D.backward()
        
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)
        self.loss_D_real = self.criterionGAN (self.pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

        # if self.opt.gan_type == 'wgan_gp':
        #     gradient_penalty, _ = util.cal_gradient_penalty(self.netD, real_GTimg, fake_B.detach(), self.device, constant=1, lambda_gp=self.opt.gp_lambda)
        #     self.loss_D_fake = torch.mean(self.pred_fake)
        #     self.loss_D_real = -torch.mean(self.pred_real)
        #
        #     self.loss_D = self.loss_D_fake + self.loss_D_real + gradient_penalty
        # else:
        #     if self.opt.gan_type in ['vanilla', 'lsgan']:
        #         self.loss_D_fake = self.criterionGAN(self.pred_fake, False)
        #         self.loss_D_real = self.criterionGAN (self.pred_real, True)
        #
        #         self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        #
        #     elif self.opt.gan_type == 're_s_gan':
        #         self.loss_D = self.criterionGAN(self.pred_real - self.pred_fake, True)
        #
        # self.loss_D.backward()


    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_B = self.fake_B
        real_B = self.real_GTimg

        pred_fake = self.netD(fake_B)
        # fake_AB = torch.cat((self.real_input, self.fake_B), 1)
        # pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.gan_weight
        
        #print(self.mask_global.shape,self.real_GTimg.shape,self.fake_B.shape)

        self.output_comp = self.mask_global * self.real_GTimg + (1 - self.mask_global) * self.fake_B
        # if self.opt.gan_type == 'wgan_gp':
        #     self.loss_G_GAN = -torch.mean(pred_fake)
        # else:
        #     if self.opt.gan_type in ['vanilla', 'lsgan']:
        #         self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.gan_weight
        #
        #     elif self.opt.gan_type == 're_s_gan':
        #         pred_real = self.netD (real_B)
        #         self.loss_G_GAN = self.criterionGAN (pred_fake - pred_real, True) * self.opt.gan_weight
        #
        #     elif self.opt.gan_type == 're_avg_gan':
        #         self.pred_real = self.netD(real_B)
        #         self.loss_G_GAN =  (self.criterionGAN (self.pred_real - torch.mean(self.pred_fake), False) \
        #                        + self.criterionGAN (self.pred_fake - torch.mean(self.pred_real), True)) / 2.
        #         self.loss_G_GAN *=  self.opt.gan_weight


        

        # If we change the mask as 'center with random position', then we can replacing loss_G_L1_m with 'Discounted L1'.
        self.loss_G_L1, self.loss_G_L1_m = 0, 0
        self.loss_G_L1 += self.criterionL1(self.fake_B, self.real_GTimg) * self.opt.lambda_A
        # calcuate mask construction loss
        # When mask_type is 'center' or 'random_with_rect', we can add additonal mask region construction loss (traditional L1).
        # Only when 'discounting_loss' is 1, then the mask region construction loss changes to 'discounting L1' instead of normal L1.
        # if self.opt.mask_type == 'center' or self.opt.mask_sub_type == 'rect':
        #     mask_patch_fake = self.fake_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2-2*self.opt.overlap, \
        #                                         self.rand_l:self.rand_l+self.opt.fineSize//2-2*self.opt.overlap]
        #     mask_patch_real = self.real_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2-2*self.opt.overlap, \
        #                                 self.rand_l:self.rand_l+self.opt.fineSize//2-2*self.opt.overlap]
        #     # Using Discounting L1 loss
        #     self.loss_G_L1_m += self.criterionL1_mask(mask_patch_fake, mask_patch_real)*self.opt.mask_weight_G

        self.loss_hole = self.criterionL1((1 - self.mask_global) * self.fake_B, (1 - self.mask_global) * self.real_GTimg)
        self.loss_valid = self.criterionL1(self.mask_global * self.fake_B, self.mask_global * self.real_GTimg)

        self.loss_G = self.loss_G_L1 + self.loss_G_L1_m + self.loss_G_GAN

        # Then, add TV loss
        self.loss_tv = self.tv_criterion(self.fake_B.float())

        # Finally, add style loss
        vgg_ft_fakeB = self.vgg16_extractor(fake_B)
        vgg_ft_realB = self.vgg16_extractor(real_B)
        self.loss_style = 0
        self.loss_content = 0

        for i in range(3):
            self.loss_style += self.criterionL2_style_loss(util.gram_matrix(vgg_ft_fakeB[i]), util.gram_matrix(vgg_ft_realB[i]))
            self.loss_content += self.criterionL2_content_loss(vgg_ft_fakeB[i], vgg_ft_realB[i])

        self.loss_style *= self.opt.style_weight
        self.loss_content *= self.opt.content_weight
        self.loss_hole *= 10.0
        self.loss_valid *= 1.0

        self.loss_G += (self.loss_valid + self.loss_hole + self.loss_style + self.loss_content + self.loss_tv)

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


