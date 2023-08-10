import torch
from torch import nn

from utils.losses import SSIM, Smooth_loss
        
        
class MSLoss(nn.Module):
    def __init__(self):
        super(MSLoss,self).__init__()
        self.l1_loss=nn.L1Loss()
        self.ssim=SSIM(window_size=11,size_average=True)
        
    def forward(self,out,re,re_d2,re_d4,denoise_d4,I_d4,hf,gt,gt_d2,gt_d4,low_d4,I_gt_d4,hf_gt):
        B,view_num,C,H,W=gt.shape
        gt=gt.reshape(B*view_num,C,H,W)
        gt_d2=gt_d2.reshape(B*view_num,C,H//2,W//2)
        gt_d4=gt_d4.reshape(B*view_num,C,H//4,W//4)
        low_d4=low_d4.reshape(B*view_num,C,H//4,W//4)
        hf_gt=hf_gt.reshape(B*view_num,C,H,W)
        
        #Final reconstruction loss
        recon_loss=self.l1_loss(gt,out)+self.l1_loss(gt,re)
        ssim_loss=1-self.ssim(gt,out)
        
        #Deep supervision loss
        d2_loss=self.l1_loss(gt_d2,re_d2)
        d4_loss=self.l1_loss(gt_d4,re_d4)
        denoise_loss=self.l1_loss(low_d4,denoise_d4)
        
        #Illumination loss
        smooth_loss=Smooth_loss(I_d4,gt_d4)
        I_d4=I_d4.reshape(B,view_num,1,H//4,W//4)
        I_d4_norm=torch.zeros_like(I_d4).cuda()
        for i in range(B):
            I_d4_one=I_d4[i]
            I_d4_one=(I_d4_one-I_d4_one.min())/(I_d4_one.max()-I_d4_one.min())
            I_d4_norm[i]=I_d4_one
        
        ref_loss=self.l1_loss(I_d4_norm,I_gt_d4)
        
        #High-frequency loss
        hf_loss=self.l1_loss(hf,hf_gt)
        
        print('recon_loss: {:.4f}, ssim_loss: {:.4f}'.format(recon_loss,ssim_loss))
        print('d2 loss: {:.4f}, d4 loss: {:.4f}'.format(d2_loss,d4_loss))
        print('Denoise loss: {:.4f}, smooth_loss: {:.4f}'.format(denoise_loss,smooth_loss))
        print('Ref loss: {:.4f}'.format(ref_loss))
        print('HF loss :{:.4f}'.format(hf_loss))
        
        total_loss=5*recon_loss+1*ssim_loss+5*d2_loss+5*d4_loss+10*denoise_loss+0.1*smooth_loss+1*ref_loss+1*hf_loss
        
        return total_loss