from collections import OrderedDict

import torch
from torch import nn
from .base_model import BaseModel
from . import networks
from itertools import chain

class AutoGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='mh_resnet_6blocks', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.n_input_modal = opt.n_input_modal
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'SR_L1', 'G_SR']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_MHG(opt.n_input_modal, opt.input_nc+opt.n_input_modal+1, opt.output_nc, opt.ngf, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator;
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, len(opt.modal_names))

        if self.isTrain:
            # define loss functions
            self.criterionCls = nn.CrossEntropyLoss()
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.criterionL2 = torch.nn.MSELoss()
            self.criterionKL = torch.nn.KLDivLoss()

        self.all_modal_names = opt.modal_names
        self.sr_weight = 0.1
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_B_no_mask = input['B'][:, :self.opt.input_nc].to(self.device)
        self.modal_names = [i[0] for i in input['modal_names']]

        target_modal_names = input['modal_names'][-1]
        self.real_B_Cls = torch.tensor([self.all_modal_names.index(i) for i in target_modal_names]).to(self.device)

        if hasattr(self, 'sr'):
            self.sr.real_A = self.real_B_no_mask
            self.sr.real_B = self.real_B_no_mask

    def forward(self, train=False):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if train:
            self.fake_B, self.decoder_features = self.netG(self.real_A, True)
        else:
            self.fake_B = self.netG(self.real_A, train)  # G(A)

    def adv(self, i):
        with torch.no_grad():
            pred, _ = self.netD(i)
            # predicted, _ = torch.max(cls_real, 1)
            return pred


    def backward_D(self):
        #------Calculate GAN loss for the discriminator------
        # Fake; stop backprop to the generator by detaching fake_B
        fake_B = self.fake_B.detach() # fake_B from generator
        g_pred_fake, g_cls_fake = self.netD(fake_B.detach())
        self.loss_D_fake = self.criterionGAN(g_pred_fake, False)

        fake_rec_B = self.sr.fake_B.detach() # fake_B from autoencoder
        e_pred_fake, e_cls_fake = self.netD(fake_rec_B)
        self.loss_D_fake += self.criterionGAN(e_pred_fake, False)
        # Real
        pred_real, cls_real = self.netD(self.real_B_no_mask)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # self.loss_D_cls = self.criterionCls(cls_real, self.real_B_Cls)

        # self.loss_D_cls_fake = -self.logsoftmax(g_cls_fake).mean()
        # self.loss_D_cls_fake = self.criterionL2(self.softmax(g_cls_fake), fake_labels.to(self.device))
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake * 0.5 + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        g_pred_fake, g_cls_fake = self.netD(self.fake_B) # b, 1, 30, 30
        self.loss_G_GAN = self.criterionGAN(g_pred_fake, True)
        # e_pred_fake, e_cls_fake = self.netD(self.sr.fake_B)
        # self.loss_G_GAN += self.criterionGAN(e_pred_fake, True)
        # self.loss_G_cls = self.criterionCls(g_cls_fake, self.real_B_Cls)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B_no_mask) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G_SR = 0
        sr_decoder_features = self.sr.get_features()
        for i in range(len(sr_decoder_features)):
            self.loss_G_SR += self.criterionL1(sr_decoder_features[i], self.decoder_features[i]) * self.sr_weight
            # self.loss_G_SR += self.criterionKL(torch.nn.functional.log_softmax(self.decoder_features[i]), torch.nn.functional.softmax(sr_decoder_features[i])) * self.sr_weight

        # Third, SR loss
        self.loss_SR_L1 = self.sr.compute_loss()

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_SR_L1 + self.loss_G_SR
        self.loss_G.backward()

    def optimize_parameters(self):
        self.sr.forward()
        self.forward(True)                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G and SR
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def compute_visuals(self):
        """Calculate additional output images for tensorboard visualization"""
        pass

    def get_current_visuals(self):
        modal_imgs = []
        for i in range(self.n_input_modal):
            modal_imgs.append(self.real_A[:, i*(self.n_input_modal+1+self.opt.input_nc):i*(self.n_input_modal+1+self.opt.input_nc)+self.opt.input_nc, :, :])
        modal_imgs.append(self.real_B_no_mask)
        visual_ret = OrderedDict()
        for name, img in zip(self.modal_names, modal_imgs):
            visual_ret[name] = img
        visual_ret['fake_' + self.modal_names[-1]] = self.fake_B
        if hasattr(self, 'sr'):
            visual_ret['reconstruct'] = self.sr.fake_B
        return visual_ret

    def add_srmodel(self, sr_model):
        self.sr = sr_model
        self.optimizer_G = torch.optim.Adam(chain(self.netG.parameters(), self.sr.netG.parameters()), lr=self.opt.lr,
                                            betas=(self.opt.beta1, 0.999))