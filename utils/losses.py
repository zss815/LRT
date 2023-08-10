import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def Gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    if torch.cuda.is_available():
        gauss=gauss.cuda()
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    window_1D = Gaussian(window_size, 1.5).unsqueeze(1)
    window_2D = window_1D.mm(window_1D.t()).float().unsqueeze(0).unsqueeze(0)
    window = window_2D.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, return_full=False, data_range=None):
    if data_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
            
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = data_range

    _, channel, height, width = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=0, groups=channel)
    mu2 = F.conv2d(img2, window, padding=0, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=0, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=0, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=0, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ssim_value = ssim_map.mean()
    else:
        ssim_value = ssim_map.mean(dim=(1,2,3))

    if return_full:
        return ssim_value, cs
    return ssim_value

def ms_ssim(img1, img2, window_size=11, size_average=True, data_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    
    for _ in range(levels):
        ssim_, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, data_range=data_range)
        mssim.append(ssim_)
        mcs.append(cs)
        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    
    # Normalize to avoid NaNs during training unstable models, not compliant with original definition
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True, data_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.data_range = data_range

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class MS_SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MS_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        return ms_ssim(img1, img2, window_size=self.window_size, size_average=self.size_average, normalize=True)

class TVLoss(nn.Module):
    def __init__(self, weight=1):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def Gradient(img):
    grad_x = torch.mean(img[:, :, :, :-1] - img[:, :, :, 1:], 1, keepdim=True)
    grad_y = torch.mean(img[:, :, :-1, :] - img[:, :, 1:, :], 1, keepdim=True)
    return grad_x,grad_y
    
   
def Smooth_loss(I,img_ref,gamma=100):
    grad_x = torch.abs(I[:, :, :, :-1] - I[:, :, :, 1:])
    grad_y = torch.abs(I[:, :, :-1, :] - I[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img_ref[:, :, :, :-1] - img_ref[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img_ref[:, :, :-1, :] - img_ref[:, :, 1:, :]), 1, keepdim=True)

    grad_x *= torch.exp(-gamma*grad_img_x)
    grad_y *= torch.exp(-gamma*grad_img_y)

    return grad_x.mean() + grad_y.mean()



