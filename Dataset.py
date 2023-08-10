import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import cv2
from torchvision.transforms import GaussianBlur, Resize

from utils.LFProcess import MP2SAIs
from utils.ImgProcess import *
    
    
class TrainData(Dataset):
    def __init__(self,data_root,crop_size):
        super(TrainData,self).__init__()
        self.gt_path=[]
        for file in os.listdir(os.path.join(data_root,'LFReal','Train')):
            if not file.startswith('.'):
                self.gt_path.append(os.path.join(data_root,'LFReal','Train',file))
        self.crop_size=crop_size
    
    def __getitem__(self,index):
        gt_path=self.gt_path[index]
        gt_mp=np.array(Image.open(gt_path))
        
        #select parameters randomly
        v_scale=np.random.uniform(0.05,0.2)
        k=np.random.uniform(1,4)
        log_read_mean=0.63*np.log(k)+0.61
        log_read=np.random.uniform(log_read_mean-0.141,log_read_mean+0.141)
        read_scale=np.exp(log_read)
        
        log_row_mean=0.43*np.log(k)+0.2
        log_row=np.random.uniform(log_row_mean-0.146,log_row_mean+0.146)
        row_scale=np.exp(log_row)
        
        #Synthesize low-light LF
        low_mp=illum_adjust(gt_mp,v_scale)
        low_raw_mp = RGB2Bayer(low_mp).astype(np.float64)
        low_raw_noise_mp = Noise_syn(low_raw_mp,k,read_scale,row_scale,q=2)
        low_noise_mp = cv2.cvtColor(low_raw_noise_mp, cv2.COLOR_BAYER_GB2RGB)
        
        I_mp=cv2.cvtColor(low_mp, cv2.COLOR_RGB2YCrCb)[...,0]
        I_mp=(I_mp-I_mp.min())/(I_mp.max()-I_mp.min())
        I_sais=MP2SAIs(I_mp,ah=7,aw=7)   #[49,H,W]
        
        #normalize
        gt_mp=gt_mp/255
        inp_mp=low_noise_mp/255
        low_mp=low_mp/255

        gt_sais=MP2SAIs(gt_mp,ah=7,aw=7)  #[49,H,W,3]
        inp_sais=MP2SAIs(inp_mp,ah=7,aw=7)  #[49,H,W,3]
        low_sais=MP2SAIs(low_mp,ah=7,aw=7)  #[49,H,W,3]
            
        H,W=inp_sais.shape[1],inp_sais.shape[2]
        
        if self.crop_size:
            h_begin=np.random.randint(H-self.crop_size)
            w_begin=np.random.randint(W-self.crop_size)
            inp_sais=inp_sais[:,h_begin:h_begin+self.crop_size,w_begin:w_begin+self.crop_size]
            gt_sais=gt_sais[:,h_begin:h_begin+self.crop_size,w_begin:w_begin+self.crop_size]
            low_sais=low_sais[:,h_begin:h_begin+self.crop_size,w_begin:w_begin+self.crop_size]
            I_sais=I_sais[:,h_begin:h_begin+self.crop_size,w_begin:w_begin+self.crop_size]
            
        inp_sais=np.transpose(inp_sais,axes=(0,3,1,2))  
        gt_sais=np.transpose(gt_sais, axes=(0,3,1,2))
        low_sais=np.transpose(low_sais, axes=(0,3,1,2))
        inp_sais=torch.from_numpy(inp_sais).float()  #[49,3,h,w]
        gt_sais=torch.from_numpy(gt_sais).float()  #[49,3,h,w]
        low_sais=torch.from_numpy(low_sais).float()  #[49,3,h,w]
        I_sais=np.expand_dims(I_sais,axis=1) #[49,1,h,w]
        I_sais=torch.from_numpy(I_sais).float()
        
        h,w=gt_sais.shape[2],gt_sais.shape[3]
        
        gt_sais_blur=GaussianBlur(kernel_size=3,sigma=0.5)(gt_sais)
        gt_sais_d2=Resize(size=(h//2,w//2))(gt_sais_blur)  #[49,3,h/2,w/2]
        gt_sais_blur2=GaussianBlur(kernel_size=3,sigma=0.5)(gt_sais_d2)
        gt_sais_d4=Resize(size=(h//4,w//4))(gt_sais_blur2)  #[49,3,h/4,w/4]
        
        low_sais_blur=GaussianBlur(kernel_size=3,sigma=0.5)(low_sais)
        low_sais_d2=Resize(size=(h//2,w//2))(low_sais_blur)  #[49,3,h/2,w/2]
        low_sais_blur2=GaussianBlur(kernel_size=3,sigma=0.5)(low_sais_d2)
        low_sais_d4=Resize(size=(h//4,w//4))(low_sais_blur2)  #[49,3,h/4,w/4]
        
        I_sais_blur=GaussianBlur(kernel_size=3,sigma=0.5)(I_sais)
        I_sais_d2=Resize(size=(h//2,w//2))(I_sais_blur)  #[49,1,h/2,w/2]
        I_sais_blur2=GaussianBlur(kernel_size=3,sigma=0.5)(I_sais_d2)
        I_sais_d4=Resize(size=(h//4,w//4))(I_sais_blur2)  #[49,1,h/4,w/4]
        
        hf_sais=gt_sais-gt_sais_blur  #[49,3,h,w]
        
        return inp_sais,gt_sais,gt_sais_d2,gt_sais_d4,low_sais_d4,I_sais_d4,hf_sais
    
    def __len__(self):
        return len(self.gt_path)
    
    
class ValData(Dataset):
    def __init__(self,data_root):
        super(ValData,self).__init__()
        self.gt_path=[]
        for file in os.listdir(os.path.join(data_root,'LFReal','Val')):
            if not file.startswith('.'):
                self.gt_path.append(os.path.join(data_root,'LFReal','Val',file))
    
    def __getitem__(self,index):
        gt_path=self.gt_path[index]
        gt_mp=np.array(Image.open(gt_path))
        
        #select parameters randomly
        v_scale=np.random.uniform(0.05,0.2)
        k=np.random.uniform(1,4)
        log_read_mean=0.63*np.log(k)+0.61
        log_read=np.random.uniform(log_read_mean-0.141,log_read_mean+0.141)
        read_scale=np.exp(log_read)
        
        log_row_mean=0.43*np.log(k)+0.2
        log_row=np.random.uniform(log_row_mean-0.146,log_row_mean+0.146)
        row_scale=np.exp(log_row)
        
        #Synthesize low-light LF
        low_mp=illum_adjust(gt_mp,v_scale)
        low_raw_mp = RGB2Bayer(low_mp).astype(np.float64)
        low_raw_noise_mp = Noise_syn(low_raw_mp,k,read_scale,row_scale,q=2)
        low_noise_mp = cv2.cvtColor(low_raw_noise_mp, cv2.COLOR_BAYER_GB2RGB)
        
        #normalize
        gt_mp=gt_mp/255
        inp_mp=low_noise_mp/255
        
        gt_sais=MP2SAIs(gt_mp,ah=7,aw=7)  #[49,H,W,3]
        inp_sais=MP2SAIs(inp_mp,ah=7,aw=7)  #[49,H,W,3]
        
        inp_sais=np.transpose(inp_sais,axes=(0,3,1,2))  
        gt_sais=np.transpose(gt_sais, axes=(0,3,1,2))  
        inp_sais=torch.from_numpy(inp_sais).float()  #[49,3,H,W]
        gt_sais=torch.from_numpy(gt_sais).float()  #[49,3,H,W]
        
        return inp_sais,gt_sais

    def __len__(self):
        return len(self.gt_path)
        
        
