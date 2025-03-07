import torch
from .base_model import BaseModel
from . import networks
from .loss import maskedL1
import os, sys
import numpy as np
import torch.nn.functional as F
from util.image_pool import ImagePool
from loguru import logger


class Generator1PoolModel(BaseModel):

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
        # parser.set_defaults(norm='instance', netG='unetlike')
        # if is_train:
        #     parser.set_defaults(pool_size=50, gan_mode='vanilla')
        # else:
        #     parser.set_defaults(dataset_mode='bratstest')
        logger.info("No model-related parser options changed.")
        return parser

    def log_config(self):
        """write default config into log file
        """
        msg = "Print model options\n"
        msg = msg + "loss names: " + str(self.loss_names) + "\n"
        msg = msg + "visual_names: " + str(self.visual_names) + "\n"
        msg = msg + "direction: " + str(self.direction) + "\n"
        # msg = msg + "trainimg_savepath: " + str(self.trainimg_savepath) + "\n"
        if self.isTrain:
            msg = msg + "optimizer: " + type(self.optimizer_G).__name__ + "\n"
        msg = msg + "model_names: " + str(self.model_names) + "\n"
        logger.info(msg)

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        self.loss_names = ['D_tumor', 'D_brain', 'D_pix',                           # losses to track and print
                        'G_GAN_tumor', 'tumor_L1',
                        'G_GAN_brain', 'brain_L1',
                        'G_GAN_pix', 'pix_L1',
                        'fuse_L1',
                    ]
        self.visual_names = ['real_A', 'real_B',
                            'fake_B_tumor', 'fake_B_brain', 'fake_B_pix', 'fuse']   # related image to save and show
        if self.isTrain:
            self.model_names = ['G1', 'D_tumor', 'D_brain','D_pix']
        else:  # during test time, only load G
            self.model_names = ['G1']
        self.direction = opt.direction                                              # cross modality synthesis direction
        # self.trainimg_savepath = opt.trainimg_savepath

        # define a generator
        self.netG1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # define a discriminator
        if self.isTrain:
            getIntermFeat = False
            self.netD_tumor = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, getIntermFeat)
            self.netD_brain = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, getIntermFeat)
            self.netD_pix = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, getIntermFeat)
            # self.netD_tumor = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
            #                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # self.netD_brain = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
            #                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # self.netD_pix = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
            #                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode, self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionmL1 = maskedL1()
            self.L1_weight = opt.lambda_L1
            self.gan_weight = opt.lambda_gan
            
            # DWA param
            self.num_of_tasks = [7]  # tasks for each head
            self.avg_cost = []       # task cost in one epoch
            self.lambda_weight = []
            for noft in self.num_of_tasks:
                self.avg_cost.append(np.zeros([opt.n_epochs + opt.n_epochs_decay + 1, noft], dtype=np.float32))  # avg task cost ndarray [total_epoch, task]
                self.lambda_weight.append(np.ones([noft, opt.n_epochs + opt.n_epochs_decay + 1]))   # each head's ndarray [task, total_epoch]

            self.fake_tumor_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_brain_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_pix_pool = ImagePool(opt.pool_size)    # create image buffer to store previously generated images

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_tumor = torch.optim.Adam(self.netD_tumor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_brain = torch.optim.Adam(self.netD_brain.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_pix = torch.optim.Adam(self.netD_pix.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_tumor)
            self.optimizers.append(self.optimizer_D_brain)
            self.optimizers.append(self.optimizer_D_pix)
        # log model config
        self.log_config()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap images in domain A and domain B.
        """
        modal = self.direction.split('_')
        self.real_A = input[modal[0]].to(self.device)
        self.real_B = input[modal[1]].to(self.device)
        self.real_B_mask = (self.real_B>self.real_B.min()).float()
        self.image_paths = input['img_path']
        if self.opt.isTrain:
            self.attention = input['seg'].to(self.device)
            self.real_B_tumor = torch.where(self.attention == 1, self.real_B, self.real_B*0-1)
            self.real_B_brain = torch.where(self.attention < 1, self.real_B, self.real_B*0-1)
            self.real_A_tumor = torch.where(self.attention == 1, self.real_A, self.real_A*0-1)
            self.real_A_brain = torch.where(self.attention < 1, self.real_A, self.real_A*0-1)
            self.real_brain_mask = torch.where(self.attention < 1, self.real_B_mask, self.real_B_mask*0)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters>."""
        self.fake_B_tumor, self.fake_B_brain, self.fake_B_pix, self.fuse, self.gate, _ = self.netG1(self.real_A)

    def test(self):
        """Run forward pass; called by both functions <test>."""
        self.fake_B_tumor, self.fake_B_brain, self.fake_B_pix, self.fuse, self.gate, self.feat = self.netG1(self.real_A)

    def backward_D_basic(self, model, inp, pred, gt):
        ## fake
        fake_B = torch.cat((inp, pred), 1)
        # fake_B = pred
        g_pred_fake = model(fake_B.detach())
        # g_pred_fake, cls_fake = model(fake_B.detach())
        loss_D_fake_dis = self.criterionGAN(g_pred_fake, False)
        # loss_D_fake_cls = self.criterionCLS(cls_fake, False)
        loss_D_fake = loss_D_fake_dis #+ loss_D_fake_cls
        ## Real
        real_B = torch.cat((inp, gt), 1)
        # real_B = gt
        pred_real = model(real_B)
        # pred_real, cls_real = model(real_B)
        loss_D_real_dis = self.criterionGAN(pred_real, True)
        # loss_D_real_cls = self.criterionCLS(cls_real, True)
        loss_D_real = loss_D_real_dis #+ loss_D_real_cls
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D

    def backward_D(self):
        #------Calculate GAN loss for the discriminator------
        # Fake; stop backprop to the generator by detaching fake_B
        fake_B_t = self.fake_tumor_pool.query(self.fake_B_tumor)
        self.loss_D_tumor = self.backward_D_basic(self.netD_tumor, self.real_A_tumor, fake_B_t, self.real_B_tumor)
        self.loss_D_tumor.backward()

        #------Calculate GAN loss for the discriminator------
        # Fake; stop backprop to the generator by detaching fake_B
        fake_B_b = self.fake_brain_pool.query(self.fake_B_brain)
        self.loss_D_brain = self.backward_D_basic(self.netD_brain, self.real_A_brain, fake_B_b, self.real_B_brain)
        self.loss_D_brain.backward()

        #------Calculate GAN loss for the discriminator------
        # Fake; stop backprop to the generator by detaching fake_B
        fake_B_p = self.fake_pix_pool.query(self.fake_B_pix)
        self.loss_D_pix = self.backward_D_basic(self.netD_pix, self.real_A, fake_B_p, self.real_B)
        self.loss_D_pix.backward()

    def backward_G(self):
        """Calculate GAN, L1 and fuse loss for the generator"""
        fake_B_cat = torch.cat((self.real_A_tumor, self.fake_B_tumor), 1)
        # fake_B_cat = self.fake_B_tumor
        g_pred_fake = self.netD_tumor(fake_B_cat)
        self.loss_G_GAN_tumor = self.criterionGAN(g_pred_fake, True)*self.gan_weight
        loss_tumor_L1_global = self.criterionL1(self.fake_B_tumor, self.real_B_tumor)
        # loss_tumor_L1_global = self.criterionDice((self.fake_B_tumor+1)/2, self.attention)
        loss_tumor_L1_local = self.criterionmL1(self.fake_B_tumor, self.real_B_tumor, self.attention)
        # loss_tumor_L1_local = self.criterionL1(self.fake_B_tumor_seg, self.real_B_tumor)
        self.loss_tumor_L1 = (loss_tumor_L1_global + loss_tumor_L1_local)*self.L1_weight
        # self.loss_tumor_ssim = self.criterionSSIM((self.fake_B_tumor+1)/2, (self.real_B_tumor+1)/2)
        # self.loss_G_tumor = self.loss_G_GAN_tumor + self.loss_tumor_L1 + self.loss_tumor_ssim

        fake_B_cat = torch.cat((self.real_A_brain, self.fake_B_brain), 1)
        # fake_B_cat = self.fake_B_brain
        g_pred_fake = self.netD_brain(fake_B_cat)
        self.loss_G_GAN_brain = self.criterionGAN(g_pred_fake, True)*self.gan_weight
        loss_brain_L1_global = self.criterionL1(self.fake_B_brain, self.real_B_brain)
        # loss_brain_L1_global = self.criterionDice((self.fake_B_brain+1)/2, self.real_brain_mask)
        loss_brain_L1_local = self.criterionmL1(self.fake_B_brain, self.real_B_brain, self.real_brain_mask)
        # loss_brain_L1_local = self.criterionL1(self.fake_B_brain_seg, self.real_B_brain)
        self.loss_brain_L1 = (loss_brain_L1_global + loss_brain_L1_local)*self.L1_weight
        # self.loss_brain_ssim = self.criterionSSIM((self.fake_B_brain+1)/2, (self.real_B_brain+1)/2)
        # self.loss_G_brain = self.loss_G_GAN_brain + self.loss_brain_L1 + self.loss_brain_ssim

        fake_B_cat = torch.cat((self.real_A, self.fake_B_pix), 1)
        # fake_B_cat = self.fake_B_pix
        g_pred_fake = self.netD_pix(fake_B_cat)
        self.loss_G_GAN_pix = self.criterionGAN(g_pred_fake, True)*self.gan_weight
        self.loss_pix_L1 = self.criterionL1(self.fake_B_pix, self.real_B)*self.L1_weight
        # self.loss_pix_idt = self.criterionL1(self.pred, self.real_B)
        # self.loss_pix_ssim = self.criterionSSIM((self.fake_B_pix+1)/2, (self.real_B+1)/2)
        # self.loss_G_pix = self.loss_G_GAN_pix + self.loss_pix_L1 + self.loss_pix_ssim + self.loss_pix_idt

        self.loss_fuse_L1 = self.criterionL1(self.fuse, self.real_B)*self.L1_weight

        losses = [
            self.loss_G_GAN_tumor, self.loss_tumor_L1, #self.loss_tumor_ssim, #
            self.loss_G_GAN_brain, self.loss_brain_L1, #self.loss_brain_ssim, #
            self.loss_G_GAN_pix, self.loss_pix_L1, #self.loss_pix_ssim, #self.loss_pix_idt, #
            self.loss_fuse_L1,
        ]
        self.avg_cost[0][self.epoch, :] += [l.item() / self.len_data for l in losses]
        self.loss_G = sum([self.lambda_weight[0][i, self.epoch] * losses[i] for i in range(self.num_of_tasks[0])])
        self.loss_G.backward()

    def softmax(self, f):
        # instead: first shift the values of f so that the highest number is 0:
        f1 = f - np.max(f) # f becomes [-666, -333, 0]
        return np.exp(f1) / np.sum(np.exp(f1))  # safe to do, gives the correct answer

    def update_weight_cost(self, index, len_data):
        """update loss weight
        weightes in epoch i = task_num * softmax( weightes in epoch i-1 / weightes in epoch i-2 )
        """
        self.epoch = index
        self.len_data = len_data
        for i in range(len(self.num_of_tasks)):
            if index == 1 or index == 2:
                self.lambda_weight[i][:, index] = 1.0
            else:
                w = self.avg_cost[i][index - 1, :] / self.avg_cost[i][index - 2, :]
                self.lambda_weight[i][:, index] = self.num_of_tasks[i] * self.softmax(w)

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD_tumor, True)  # enable backprop for D
        self.set_requires_grad(self.netD_brain, True)  # enable backprop for D
        self.set_requires_grad(self.netD_pix, True)    # enable backprop for D

        self.optimizer_D_tumor.zero_grad()     # set D's gradients to zero
        self.optimizer_D_brain.zero_grad()     # set D's gradients to zero
        self.optimizer_D_pix.zero_grad()       # set D's gradients to zero

        self.backward_D()                # calculate gradients for D

        self.optimizer_D_tumor.step()          # update D's weights
        self.optimizer_D_brain.step()          # update D's weights
        self.optimizer_D_pix.step()            # update D's weights

        # update G
        self.set_requires_grad(self.netD_tumor, False)
        self.set_requires_grad(self.netD_brain, False)
        self.set_requires_grad(self.netD_pix, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        # torch.nn.utils.clip_grad_norm_(self.netG1.parameters(), 1)
        self.optimizer_G.step()             # udpate G's weights

    def get_current_weight(self):
        """Return traning loss weights."""
        ret_ = []
        for item in self.lambda_weight:
            ret_.append(item.tolist())
        return ret_

    def get_current_lr(self):
        """Return traning learning rate."""
        ret_ = self.optimizers[0].param_groups[0]['lr']
        return ret_
    
    def savetest(self, i):
        save_path = self.opt.results_dir + self.opt.name + '/testfeat_' + self.opt.epoch + '/'
        self.check_paths(save_path+'1')
        self.check_paths(save_path+'2')
        self.check_paths(save_path+'3')
        # save_name = os.path.join(save_path, str(i)+'.npy')
        np.save(save_path+'/1/'+str(i)+'.npy', self.feat[0].cpu().detach().numpy())
        np.save(save_path+'/2/'+str(i)+'.npy', self.feat[1].cpu().detach().numpy())
        np.save(save_path+'/3/'+str(i)+'.npy', self.feat[2].cpu().detach().numpy())


    def check_paths(self, path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except OSError as e:
            print(e)
            sys.exit(1)
