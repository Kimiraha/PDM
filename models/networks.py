import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import functools
import numpy as np
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], n_blocks=9):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'generator1':
        net = ResnetGenerator1(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'generator1branch':
        net = ResnetGenerator1branch(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'generator1gate':
        net = ResnetGenerator1Gate(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks)
    elif netG == 'generator1single':
        net = ResnetGenerator1Single(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], getIntermFeat=False):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'multiscale':
        net = MultiscaleDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer, num_D=3, getIntermFeat=getIntermFeat)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, device, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.device = device
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor.to(self.device)

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


##############################################################################
# Generator
##############################################################################

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetGenerator1(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator1, self).__init__()
        # input_nc = 1
        # output_nc = 1
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        norm_layer = nn.InstanceNorm2d

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # source encoder
        self.inc1 = Inconv(input_nc, ngf)
        self.down1_1 = Down(ngf, ngf * 2)
        self.down1_2 = Down(ngf * 2, ngf * 4)

        model = []
        for i in range(n_blocks):
            model += [ResBlock(ngf * 4, padding_type=padding_type,
                               norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.resblocks1 = nn.Sequential(*model)
        model = []
        for i in range(n_blocks):
            model += [ResBlock(ngf * 4, padding_type=padding_type,
                               norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.resblocks2 = nn.Sequential(*model)
        model = []
        for i in range(n_blocks):
            model += [ResBlock(ngf * 4, padding_type=padding_type,
                               norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.resblocks3 = nn.Sequential(*model)

        # decoder1
        # self.up1_1 = Up(ngf * 4, ngf * 2)
        # self.up1_2 = Up(ngf * 2, ngf)
        # self.outc1 = Outconv(ngf, output_nc)

        # # decoder2
        # self.up2_1 = Up(ngf * 4, ngf * 2)
        # self.up2_2 = Up(ngf * 2, ngf)
        # self.outc2 = Outconv(ngf, output_nc)

        # # decoder3
        # self.up3_1 = Up(ngf * 4, ngf * 2)
        # self.up3_2 = Up(ngf * 2, ngf)
        # self.outc3 = Outconv(ngf, output_nc)

        # decoder1
        self.cov1x1_1_1 = nn.Conv2d(ngf * 4 * 2, ngf * 4, kernel_size=1)
        self.up1_1 = Up(ngf * 4, ngf * 2)
        self.cov1x1_1_2 = nn.Conv2d(ngf * 2 * 2, ngf * 2, kernel_size=1)
        self.up1_2 = Up(ngf * 2, ngf)
        self.cov1x1_1_3 = nn.Conv2d(ngf * 2, ngf, kernel_size=1)
        self.outc1 = Outconv(ngf, output_nc)

        # decoder2
        self.cov1x1_2_1 = nn.Conv2d(ngf * 4 * 2, ngf * 4, kernel_size=1)
        self.up2_1 = Up(ngf * 4, ngf * 2)
        self.cov1x1_2_2 = nn.Conv2d(ngf * 2 * 2, ngf * 2, kernel_size=1)
        self.up2_2 = Up(ngf * 2, ngf)
        self.cov1x1_2_3 = nn.Conv2d(ngf * 2, ngf, kernel_size=1)
        self.outc2 = Outconv(ngf, output_nc)

        # decoder3
        self.cov1x1_3_1 = nn.Conv2d(ngf * 4 * 2, ngf * 4, kernel_size=1)
        self.up3_1 = Up(ngf * 4, ngf * 2)
        self.cov1x1_3_2 = nn.Conv2d(ngf * 2 * 2, ngf * 2, kernel_size=1)
        self.up3_2 = Up(ngf * 2, ngf)
        self.cov1x1_3_3 = nn.Conv2d(ngf * 2, ngf, kernel_size=1)
        self.outc3 = Outconv(ngf, output_nc)

    def forward(self, x):

        out = {}
        x1 = self.inc1(x)
        x2 = self.down1_1(x1)
        y0 = self.down1_2(x2)

        out['bottle1'] = self.resblocks1(y0)
        out['bottle2'] = self.resblocks2(y0)
        out['bottle3'] = self.resblocks3(y0)

        # out['res1_feat3'] = self.up1_1(out['bottle1'])
        # out['res1_feat2'] = self.up1_2(out['res1_feat3'])
        # out['res1'] = self.outc1(out['res1_feat2'])

        # out['res2_feat3'] = self.up2_1(out['bottle2'])
        # out['res2_feat2'] = self.up2_2(out['res2_feat3'])
        # out['res2'] = self.outc2(out['res2_feat2'])

        # out['res3_feat3'] = self.up3_1(out['bottle3'])
        # out['res3_feat2'] = self.up3_2(out['res3_feat3'])
        # out['res3'] = self.outc3(out['res3_feat2'])
        
        y1_c1 = torch.cat([out['bottle1'], y0], dim=1)
        y1_c1 = self.cov1x1_1_1(y1_c1)
        out['res1_feat3'] = self.up1_1(y1_c1)
        y1_c2 = torch.cat([out['res1_feat3'], x2], dim=1)
        y1_c2 = self.cov1x1_1_2(y1_c2)
        out['res1_feat2'] = self.up1_2(y1_c2)
        y1_c3 = torch.cat([out['res1_feat2'], x1], dim=1)
        y1_c3 = self.cov1x1_1_3(y1_c3)
        out['res1'] = self.outc1(y1_c3)

        y2_c1 = torch.cat([out['bottle2'], y0], dim=1)
        y2_c1 = self.cov1x1_2_1(y2_c1)
        out['res2_feat3'] = self.up2_1(y2_c1)
        y2_c2 = torch.cat([out['res2_feat3'], x2], dim=1)
        y2_c2 = self.cov1x1_2_2(y2_c2)
        out['res2_feat2'] = self.up2_2(y2_c2)
        y2_c3 = torch.cat([out['res2_feat2'], x1], dim=1)
        y2_c3 = self.cov1x1_2_3(y2_c3)
        out['res2'] = self.outc2(y2_c3)

        y3_c1 = torch.cat([out['bottle3'], y0], dim=1)
        y3_c1 = self.cov1x1_3_1(y3_c1)
        out['res3_feat3'] = self.up3_1(y3_c1)
        y3_c2 = torch.cat([out['res3_feat3'], x2], dim=1)
        y3_c2 = self.cov1x1_3_2(y3_c2)
        out['res3_feat2'] = self.up3_2(y3_c2)
        y3_c3 = torch.cat([out['res3_feat2'], x1], dim=1)
        y3_c3 = self.cov1x1_3_3(y3_c3)
        out['res3'] = self.outc3(y3_c3)

        return out['res1'], out['res2'], out['res3'], [x1, x2, y0]


class ResnetGenerator1Gate(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator1Gate, self).__init__()
        # input_nc = 1
        # output_nc = 1
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        norm_layer = nn.InstanceNorm2d

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # source encoder
        self.inc1 = Inconv(input_nc, ngf)
        self.down1_1 = Down(ngf, ngf * 2)
        self.down1_2 = Down(ngf * 2, ngf * 4)

        model = []
        for i in range(n_blocks):
            model += [ResBlock(ngf * 4, padding_type=padding_type,
                               norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.resblocks1 = nn.Sequential(*model)
        model = []
        for i in range(n_blocks):
            model += [ResBlock(ngf * 4, padding_type=padding_type,
                               norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.resblocks2 = nn.Sequential(*model)
        model = []
        for i in range(n_blocks):
            model += [ResBlock(ngf * 4, padding_type=padding_type,
                               norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.resblocks3 = nn.Sequential(*model)

        # decoder1
        self.up1_1 = Up(ngf * 4, ngf * 2)
        self.up1_2 = Up(ngf * 2, ngf)
        self.outc1 = Outconv(ngf, output_nc)

        # decoder2
        self.up2_1 = Up(ngf * 4, ngf * 2)
        self.up2_2 = Up(ngf * 2, ngf)
        self.outc2 = Outconv(ngf, output_nc)

        # decoder3
        self.up3_1 = Up(ngf * 4, ngf * 2)
        self.up3_2 = Up(ngf * 2, ngf)
        self.outc3 = Outconv(ngf, output_nc)

        # Router
        self.router = nn.Sequential(
            Up(ngf * 4, ngf * 2),
            Up(ngf * 2, ngf),
            nn.Conv2d(ngf, 3, kernel_size=1),
        )
        self.softmax = nn.Softmax(dim=1)

        # # Router
        # self.router = nn.Sequential(
        #     Down(ngf * 4, ngf * 2),
        #     Down(ngf * 2, ngf),
        #     nn.Conv2d(ngf, 3, kernel_size=1),
        #     nn.AdaptiveMaxPool2d(1),
        # )
        # self.softmax = nn.Softmax(dim=1)

        # # Router
        # self.router = nn.Sequential(
        #     Inconv(input_nc * 3, ngf),
        #     Inconv(ngf, ngf),
        #     nn.Conv2d(ngf, 3, kernel_size=7, padding=3),
        # )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        out = {}
        x1 = self.inc1(x)
        x2 = self.down1_1(x1)
        y0 = self.down1_2(x2)

        out['bottle1'] = self.resblocks1(y0)
        out['bottle2'] = self.resblocks2(y0)
        out['bottle3'] = self.resblocks3(y0)

        out['res1_feat3'] = self.up1_1(out['bottle1'])
        out['res1_feat2'] = self.up1_2(out['res1_feat3'])
        out['res1'] = self.outc1(out['res1_feat2'])

        out['res2_feat3'] = self.up2_1(out['bottle2'])
        out['res2_feat2'] = self.up2_2(out['res2_feat3'])
        out['res2'] = self.outc2(out['res2_feat2'])

        out['res3_feat3'] = self.up3_1(out['bottle3'])
        out['res3_feat2'] = self.up3_2(out['res3_feat3'])
        out['res3'] = self.outc3(out['res3_feat2'])

        pred = torch.cat([out['res1'], out['res2'], out['res3']], dim=1)
        # gate = self.router(y0)
        # gate = self.softmax(gate)
        gate = 0.5

        fuse = gate * (pred+1)/2
        fuse = fuse.sum(dim=1, keepdim=True)*2-1


        # pred = torch.cat([out['res1'], out['res2'], out['res3']], dim=1)
        # gate = self.router(y0)
        # gate = self.softmax(gate)

        # fuse = gate * (pred+1)/2
        # fuse = fuse.sum(dim=1, keepdim=True)*2-1


        # pred = torch.cat([out['res1'], out['res2'], out['res3']], dim=1)
        # gate = self.router(pred)
        # gate = self.softmax(gate)

        # fuse = gate * (pred+1)/2
        # fuse = fuse.sum(dim=1, keepdim=True)*2-1

        return out['res1'], out['res2'], out['res3'], fuse, gate, [out['bottle1'], out['bottle2'], out['bottle3']]


class ResnetGenerator1Single(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator1Single, self).__init__()
        # input_nc = 1
        # output_nc = 1
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        norm_layer = nn.InstanceNorm2d

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # source encoder
        self.inc1 = Inconv(input_nc, ngf)
        self.down1_1 = Down(ngf, ngf * 2)
        self.down1_2 = Down(ngf * 2, ngf * 4)

        model = []
        for i in range(n_blocks):
            model += [ResBlock(ngf * 4, padding_type=padding_type,
                               norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.resblocks1 = nn.Sequential(*model)

        # decoder1
        self.up1_1 = Up(ngf * 4, ngf * 2)
        self.up1_2 = Up(ngf * 2, ngf)
        self.outc1 = Outconv(ngf, output_nc)


    def forward(self, x):

        out = {}
        x1 = self.inc1(x)
        x2 = self.down1_1(x1)
        y0 = self.down1_2(x2)

        out['bottle1'] = self.resblocks1(y0)

        out['res1_feat3'] = self.up1_1(out['bottle1'])
        out['res1_feat2'] = self.up1_2(out['res1_feat3'])
        out['res1'] = self.outc1(out['res1_feat2'])

        return out['res1']


class ResnetGenerator1branch(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator1branch, self).__init__()
        # input_nc = 1
        # output_nc = 1
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        norm_layer = nn.InstanceNorm2d

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # source encoder
        self.inc1 = Inconv(input_nc, ngf)
        self.down1_1 = Down(ngf, ngf * 2)
        self.down1_2 = Down(ngf * 2, ngf * 4)

        model = []
        for i in range(n_blocks):
            model += [ResBlock(ngf * 4, padding_type=padding_type,
                               norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.resblocks1 = nn.Sequential(*model)


        # decoder1
        self.cov1x1_1_1 = nn.Conv2d(ngf * 4 * 2, ngf * 4, kernel_size=1)
        self.up1_1 = Up(ngf * 4, ngf * 2)
        self.cov1x1_1_2 = nn.Conv2d(ngf * 2 * 2, ngf * 2, kernel_size=1)
        self.up1_2 = Up(ngf * 2, ngf)
        self.cov1x1_1_3 = nn.Conv2d(ngf * 2, ngf, kernel_size=1)
        self.outc1 = Outconv(ngf, output_nc)


    def forward(self, x):

        out = {}
        x1 = self.inc1(x)
        x2 = self.down1_1(x1)
        y0 = self.down1_2(x2)

        out['bottle1'] = self.resblocks1(y0)
        
        y1_c1 = torch.cat([out['bottle1'], y0], dim=1)
        y1_c1 = self.cov1x1_1_1(y1_c1)
        out['res1_feat3'] = self.up1_1(y1_c1)
        y1_c2 = torch.cat([out['res1_feat3'], x2], dim=1)
        y1_c2 = self.cov1x1_1_2(y1_c2)
        out['res1_feat2'] = self.up1_2(y1_c2)
        y1_c3 = torch.cat([out['res1_feat2'], x1], dim=1)
        y1_c3 = self.cov1x1_1_3(y1_c3)
        out['res1'] = self.outc1(y1_c3)

        return out['res1']


class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Inconv, self).__init__()
        self.pad = nn.ReflectionPad2d(3)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0, bias=1)
        self.inn = nn.InstanceNorm2d(out_ch)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.relu(self.inn(self.conv(self.pad(x))))
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=1)
        self.inn = nn.InstanceNorm2d(out_ch)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.relu(self.inn(self.conv(x)))
        return x


# Define a Resnet block
class ResBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return nn.ReLU(True)(out)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.Conv2d(in_ch, out_ch,
            #           kernel_size=3, stride=1,
            #           padding=1, bias=use_bias),
            nn.ConvTranspose2d(in_ch, out_ch,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.outconv(x)
        return x


##############################################################################
# Discriminator
##############################################################################

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw, bias=use_bias), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        """Standard forward."""
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc=input_nc, ndf=ndf, n_layers=n_layers,
                                    norm_layer=norm_layer, getIntermFeat=getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result

    def fuse_forward(self, input):
        num_D = 1
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result