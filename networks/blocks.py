import torch
import torch.nn as nn

    
#Residual block
class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
       super(ResBlock,self).__init__() 
       self.conv1=nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                                 nn.GroupNorm(num_groups=8, num_channels=out_channels),
                                 nn.LeakyReLU(inplace=True))
       self.conv2=nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                 nn.GroupNorm(num_groups=8, num_channels=out_channels))
       if in_channels != out_channels:
            self.shortcut=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1),
                                            nn.GroupNorm(num_groups=8, num_channels=out_channels))
       else:
            self.shortcut=None
       self.relu=nn.LeakyReLU(inplace=True)
       self.stride=stride
       
    def forward(self,x):
        residual=self.conv1(x)
        residual=self.conv2(residual)
        if self.shortcut:
            x=self.shortcut(x)
        if self.stride!=1:
            x=nn.functional.interpolate(x,scale_factor=0.5,mode='bilinear')
        out=self.relu(x+residual)
        return out


#Upsampling and skip connection
class UpSkip(nn.Module):
    def __init__(self,in_channels,out_channels,mode):
        super(UpSkip, self).__init__()
        self.upsample=nn.Upsample(scale_factor=2,mode='bilinear')
        self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.mode=mode
        
    def forward(self,x1,x2):
        x1=self.upsample(x1)
        x1=self.conv(x1)
        if self.mode=='sum':
            out=x1+x2
        elif self.mode=='cat':
            out=torch.cat([x1,x2],1)
        return out
  
  
    
 
    
        




 
