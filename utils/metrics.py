import numpy as np
from skimage.metrics import structural_similarity


def PSNR(img1,img2,data_range):
    mse = ((img1 - img2) ** 2).mean()
    psnr = 10 * np.log10(np.power(data_range,2) / (mse+1e-6))
    return psnr

def SSIM(img1,img2,data_range,channel_first=True):
    if channel_first:
        c=img1.shape[0]
        total_ssim=0
        for i in range(c):
            img1_c=img1[i]
            img2_c=img2[i]
            ssim=structural_similarity(img1_c,img2_c,data_range=data_range)
            total_ssim+=ssim
        ssim=total_ssim/c
    else:
        c=img1.shape[-1]
        total_ssim=0
        for i in range(c):
            img1_c=img1[...,i]
            img2_c=img2[...,i]
            ssim=structural_similarity(img1_c,img2_c,data_range=data_range)
            total_ssim+=ssim
        ssim=total_ssim/c
    return ssim
