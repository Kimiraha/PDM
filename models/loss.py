import torch
from torch import nn
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable
import time
        

class GradNormLoss(nn.Module):
    def __init__(self, num_of_task=3, alpha=1.5):
        super(GradNormLoss, self).__init__()
        self.num_of_task = num_of_task
        self.alpha = alpha
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.w = nn.Parameter(torch.ones(num_of_task, dtype=torch.float, device=self.device))
        self.l1_loss = nn.L1Loss()
        self.L_0 = None

    # standard forward pass
    def forward(self, L_t: torch.Tensor):
        # initialize the initial loss `Li_0`
        if self.L_0 is None:
            self.L_0 = L_t.detach() # detach
        # compute the weighted loss w_i(t) * L_i(t)
        self.L_t = L_t 
        self.wL_t = L_t * self.w
        # the reduced weighted loss
        self.total_loss = self.wL_t.sum()
        return self.total_loss

    # additional forward & backward pass
    def additional_forward_and_backward(self, grad_norm_weights, optimizer: torch.optim.Optimizer):
        # do `optimizer.zero_grad()` outside
        time0 = time.time()
        self.total_loss.backward(retain_graph=True)
        time1 = time.time()
        # in standard backward pass, `w` does not require grad
        self.w.grad.data = self.w.grad.data * 0.0

        self.GW_t = []
        for i in range(self.num_of_task):
            # get the gradient of this task loss with respect to the shared parameters
            GiW_t = torch.autograd.grad(self.L_t[i], grad_norm_weights, retain_graph=True) #, create_graph=True)
            # compute the norm
            self.GW_t.append(torch.norm(GiW_t[0] * self.w[i]))
        time2 = time.time()
        self.GW_t = torch.stack(self.GW_t) # do not detatch
        self.bar_GW_t = self.GW_t.detach().mean()
        self.tilde_L_t = (self.L_t / self.L_0).detach()
        self.r_t = self.tilde_L_t / self.tilde_L_t.mean()
        grad_loss = self.l1_loss(self.GW_t, self.bar_GW_t * (self.r_t ** self.alpha))
        self.w.grad = torch.autograd.grad(grad_loss, self.w)[0]
        optimizer.step()

        self.GW_ti, self.bar_GW_t, self.tilde_L_t, self.r_t, self.L_t, self.wL_t = None, None, None, None, None, None
        # re-norm
        self.w.data = self.w.data / self.w.data.sum() * self.num_of_task
        time3 = time.time()
        print("total forward: %.3f, get grad: %.3f, reset: %.3f\n" % (time1-time0, time2-time1, time3-time2))


class TverskyLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                square=False):
        """
        paper: https://arxiv.org/pdf/1706.05721.pdf
        """
        super(TverskyLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.alpha = 0.7
        self.beta = 1 - self.alpha

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp = (y*x).sum()
        fp = ((1-y)*x).sum()
        fn = (y*(1-x)).sum()
        # tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)


        tversky = (tp + self.smooth) / (tp + self.alpha*fn + self.beta*fp + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]
        tversky = tversky.mean()

        return 1-tversky


class FocalLoss(nn.Module):
   """
   copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
   This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
   'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
       Focal_Loss= -1*alpha*(1-pt)*log(pt)
   :param num_class:
   :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
   :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                   focus on hard misclassified example
   :param smooth: (float,double) smooth value when cross entropy
   :param balance_index: (int) balance class index, should be specific when alpha is float
   :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
   """

   def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
       super(FocalLoss, self).__init__()
       self.apply_nonlin = apply_nonlin
       self.alpha = alpha
       self.gamma = gamma
       self.balance_index = balance_index
       self.smooth = smooth
       self.size_average = size_average

       if self.smooth is not None:
           if self.smooth < 0 or self.smooth > 1.0:
               raise ValueError('smooth value should be in [0,1]')

   def forward(self, logit, target):
       if self.apply_nonlin is not None:
           logit = self.apply_nonlin(logit)
       num_class = logit.shape[1]

       if logit.dim() > 2:
           # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
           logit = logit.view(logit.size(0), logit.size(1), -1)
           logit = logit.permute(0, 2, 1).contiguous()
           logit = logit.view(-1, logit.size(-1))
       target = torch.squeeze(target, 1)
       target = target.view(-1, 1)
       # print(logit.shape, target.shape)
       # 
       alpha = self.alpha

       if alpha is None:
           alpha = torch.ones(num_class, 1)
       elif isinstance(alpha, (list, np.ndarray)):
           assert len(alpha) == num_class
           alpha = torch.FloatTensor(alpha).view(num_class, 1)
           alpha = alpha / alpha.sum()
       elif isinstance(alpha, float):
           alpha = torch.ones(num_class, 1)
           alpha = alpha * (1 - self.alpha)
           alpha[self.balance_index] = self.alpha

       else:
           raise TypeError('Not support alpha type')
       
       if alpha.device != logit.device:
           alpha = alpha.to(logit.device)

       idx = target.cpu().long()

       one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
       one_hot_key = one_hot_key.scatter_(1, idx, 1)
       if one_hot_key.device != logit.device:
           one_hot_key = one_hot_key.to(logit.device)

       if self.smooth:
           one_hot_key = torch.clamp(
               one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
       pt = (one_hot_key * logit).sum(1) + self.smooth
       logpt = pt.log()

       gamma = self.gamma

       alpha = alpha[idx]
       alpha = torch.squeeze(alpha)
       loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

       if self.size_average:
           loss = loss.mean()
       else:
           loss = loss.sum()
       return loss



def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


class BiDiceLoss(nn.Module):
	def __init__(self):
		super(BiDiceLoss, self).__init__()
	
	def forward(self, input, targets):
		N = targets.size()[0]
		smooth = 1
		input_flat = input.view(N, -1)
		targets_flat = targets.view(N, -1)
	
		intersection = input_flat * targets_flat 
		N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)

		loss = 1 - N_dice_eff.sum() / N
		return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(
        img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(
        img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1-_ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def get_high_frequency_kernel(choose_kernel):
    
    kernelX = [[[-1.0,  0.0,  1.0],
                [-2.0,  0.0,  2.0],
                [-1.0,  0.0,  1.0]]]
    kernelY = [[[-1.0, -2.0,  -1.0],
                [0.0,   0.0,   0.0],
                [1.0,   2.0,   1.0]]]  # out_channel,channels
    kernelX = torch.cuda.FloatTensor(kernelX).expand(1,1,3,3)
    kernelY = torch.cuda.FloatTensor(kernelY).expand(1,1,3,3)

    # print("model high frequency kernel is",choose_kernel)

    return kernelX, kernelY

class SobelLoss(nn.Module):
    def __init__(self, h_kernel=''):
        super(SobelLoss, self).__init__()
        self.kernelX, self.kernelY = get_high_frequency_kernel(h_kernel)
        self.weightX = nn.Parameter(data=self.kernelX, requires_grad=False)
        self.weightY = nn.Parameter(data=self.kernelY, requires_grad=False)

    def forward(self, pred, gt):
        fakeX = F.conv2d(pred, self.weightX, bias=None, stride=1, padding=1)
        fakeY = F.conv2d(pred, self.weightY, bias=None, stride=1, padding=1)
        # fake_sobel = torch.sqrt((fakeX*fakeX) + (fakeY*fakeY))
        fake_sobel = (torch.abs(fakeX) + torch.abs(fakeY))/2
        gtX = F.conv2d(gt, self.weightX, bias=None, stride=1, padding=1)
        gtY = F.conv2d(gt, self.weightY, bias=None, stride=1, padding=1)
        # gt_sobel = torch.sqrt((gtX*gtX) + (gtY*gtY))
        gt_sobel = (torch.abs(gtX) + torch.abs(gtY))/2
        # print(fake_sobel.shape, gt_sobel.shape)

        high_frequency_loss = F.l1_loss(fake_sobel, gt_sobel)

        return high_frequency_loss#,gt_sobel,fake_sobel


class feature_loss(nn.Module):

    def __init__(self,device):
        super().__init__()
        self.device = device

    def g_rof_loss(self, D_f_hard_mid, D_f_latent_mid, reduction='mean'):

        mid_layer_num = len(D_f_hard_mid)
        batch_size = D_f_hard_mid[0].size(0)
        loss=torch.zeros(mid_layer_num, dtype=torch.float32).cuda(self.device)
        for mid_layer_i in range(mid_layer_num):
            if D_f_hard_mid[0].size() != D_f_hard_mid[0].size():
                raise ValueError("feature_loss(mid_layer_{}):Using a D_f_hard size ({}) that is different to the D_f_latent size ({}) is deprecated. "
                                 "Please ensure they have the same size.".format(mid_layer_i, D_f_hard_mid.size(), D_f_hard_mid.size()))
            rof_loss=nn.MSELoss(reduction=reduction)(D_f_hard_mid[mid_layer_i], D_f_latent_mid[mid_layer_i])
            loss[mid_layer_i]=rof_loss

        return torch.mean(loss)

    def singleforward(self, D_f_hard_mid, D_f_latent_mid):
        l_G_rof = self.g_rof_loss(D_f_hard_mid, D_f_latent_mid)
        return l_G_rof

    def forward(self, D_f_hard_mid, D_f_latent_mid):
        d_num = len(D_f_hard_mid)
        l_G_rof = torch.zeros(d_num, dtype=torch.float32).cuda(self.device)
        for i in range(d_num):
            l_G_rof[i] = self.singleforward(D_f_hard_mid[i], D_f_latent_mid[i])
        return torch.mean(l_G_rof)


class maskedL1(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, seg):
        if seg.sum() == 0:
            return 0
        pred_tmp = torch.where(seg==1, pred, pred*0-1)
        error = torch.sum(torch.abs(pred_tmp-gt))
        loss = error / seg.sum()

        return loss
