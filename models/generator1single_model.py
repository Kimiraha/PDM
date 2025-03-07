import torch
from .base_model import BaseModel
from . import networks
from .loss import SSIM
import imageio, os
from PIL import Image
import numpy as np
import torch.nn.functional as F
from util.image_pool import ImagePool


class Generator1SingleModel(BaseModel):

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
        #     parser.set_defaults(pool_size=0, gan_mode='vanilla')
        # else:
        #     parser.set_defaults(dataset_mode='bratstest')
        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        self.loss_names = ['D_pix', #'G_tumor', 'G_brain', 'G_pix']
                        'G_GAN_pix', 'pix_L1', 'pix_ssim', 'pix_idt', #
                    ]
        self.visual_names = ['real_A', 'real_B',
                            'fake_B_pix', 'pred']
        if self.isTrain:
            self.model_names = ['G1', 'D_pix']
        else:  # during test time, only load G
            self.model_names = ['G1']
        self.direction = opt.direction
        self.trainimg_savepath = opt.trainimg_savepath
        # define a generator
        self.netG1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'generator1single', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, n_blocks=9)
        # define a discriminator
        if self.isTrain:  
            getIntermFeat = False
            self.netD_pix = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, getIntermFeat)
            # self.netD_pix = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
            #                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode, self.device, target_real_label=1.0)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionSSIM = SSIM()
            
            # DWA param
            self.num_of_tasks = [4]  # tasks for each G
            self.avg_cost = []
            self.lambda_weight = []
            for noft in self.num_of_tasks:
                self.avg_cost.append(np.zeros([opt.n_epochs + opt.n_epochs_decay + 1, noft], dtype=np.float32))
                self.lambda_weight.append(np.ones([noft, opt.n_epochs + opt.n_epochs_decay + 1]))

            self.fake_pix_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            D_rate = 1
            self.optimizer_G = torch.optim.Adam(self.netG1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_pix = torch.optim.Adam(self.netD_pix.parameters(), lr=opt.lr/D_rate, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_pix)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap images in domain A and domain B.
        """
        modal = self.direction.split('_')
        self.real_A = input[modal[0]].to(self.device)
        self.real_B = input[modal[1]].to(self.device)
        self.image_paths = input['img_path']
        if self.opt.isTrain:
            self.attention = input['seg'].to(self.device)
            self.real_B_tumor = torch.where(self.attention == 1, self.real_B, self.real_B*0-1)
            self.real_B_brain = torch.where(self.attention < 1, self.real_B, self.real_B*0-1)
            self.real_A_tumor = torch.where(self.attention == 1, self.real_A, self.real_A*0-1)
            self.real_A_brain = torch.where(self.attention < 1, self.real_A, self.real_A*0-1)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B_pix = self.netG1(self.real_A)
        self.pred = self.netG1(self.real_B)

    def inference(self):
        self.fake_B_pix = self.netG1(self.real_A)
        self.pred = self.netG1(self.fake_B_pix)

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
        fake_B_p = self.fake_pix_pool.query(self.fake_B_pix)
        self.loss_D_pix = self.backward_D_basic(self.netD_pix, self.real_A, fake_B_p, self.real_B)
        self.loss_D_pix.backward()

    def backward_G(self):
        """Calculate GAN, L1 and feat loss for the generator"""
        fake_B_cat = torch.cat((self.real_A, self.fake_B_pix), 1)
        # fake_B_cat = self.fake_B_pix
        g_pred_fake = self.netD_pix(fake_B_cat)
        self.loss_G_GAN_pix = self.criterionGAN(g_pred_fake, True)
        self.loss_pix_L1 = self.criterionL1(self.fake_B_pix, self.real_B)
        self.loss_pix_idt = self.criterionL1(self.pred, self.real_B)
        self.loss_pix_ssim = self.criterionSSIM(self.fake_B_pix, self.real_B)
        # self.loss_G_pix = self.loss_G_GAN_pix + self.loss_pix_L1 + self.loss_pix_ssim + self.loss_pix_idt

        losses = [ 
            self.loss_G_GAN_pix, self.loss_pix_L1, self.loss_pix_ssim, self.loss_pix_idt, #
        ]
        self.avg_cost[0][self.epoch, :] += [l.item() / self.len_data for l in losses]
        self.loss_G = sum([self.lambda_weight[0][i, self.epoch] * losses[i] for i in range(self.num_of_tasks[0])])
        self.loss_G.backward()

    def softmax(self, f):
        # instead: first shift the values of f so that the highest number is 0:
        f1 = f - np.max(f) # f becomes [-666, -333, 0]
        return np.exp(f1) / np.sum(np.exp(f1))  # safe to do, gives the correct answer

    def update_weight_cost(self, index, len_data):
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
        self.set_requires_grad(self.netD_pix, True)  # enable backprop for D
        self.optimizer_D_pix.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D_pix.step()          # update D's weights

        # update G
        self.set_requires_grad(self.netD_pix, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        # torch.nn.utils.clip_grad_norm_(self.netG1.parameters(), 1)
        self.optimizer_G.step()             # udpate G's weights

    def saveimg(self, e):
        save_path = self.trainimg_savepath + '/test/epoch_' + str(e)
        self.check_paths(save_path)
        save_name = os.path.join(save_path, 'input_data'+'.png')
        self.save_image(save_name, (self.real_A[0]+1)/2)
        save_name = os.path.join(save_path, 'target'+'.png')
        self.save_image(save_name, (self.real_B[0]+1)/2)
        # save_name = os.path.join(save_path, 'real_tumor'+'.png')
        # self.save_image(save_name, (self.real_B_tumor[0]+1)/2)
        # save_name = os.path.join(save_path, 'real_brain'+'.png')
        # self.save_image(save_name, (self.real_B_brain[0]+1)/2)
        save_name = os.path.join(save_path, 'fake_pix'+'.png')
        self.save_image(save_name, (self.fake_B_pix[0]+1)/2)
        save_name = os.path.join(save_path, 'fake_pred'+'.png')
        self.save_image(save_name, (self.pred[0]+1)/2)


    def save_image(self, filename, data):
        data = data*255
        img = data.clone().clamp(0, 255).cpu().detach().numpy()
        img = img.astype(np.uint8)
        imageio.imwrite(filename, img[0])

    def savetest(self, i):
        save_path = self.opt.results_dir + self.opt.name + '/test1branch_' + self.opt.epoch
        self.check_paths(save_path)
        save_name = os.path.join(save_path, str(i)+'.png')

        out = self.fake_B_pix[0].cpu().detach().numpy()
        out = ((out+1)/2)*255
        out = Image.fromarray(np.uint8(out[0]))
        out.save(save_name)

    def check_paths(self, path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except OSError as e:
            print(e)
            sys.exit(1)
