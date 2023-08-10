import torch
import torch.nn as nn
from torchvision.transforms import GaussianBlur, Resize

from blocks import ResBlock, UpSkip


class AngularFormer(nn.Module):
    def __init__(self,in_channels,an,pool_size,num_heads):
        super(AngularFormer,self).__init__()
        self.an=an
        self.num_heads=num_heads
        self.pool=nn.AdaptiveAvgPool2d(pool_size)
        dim=(pool_size**2)*in_channels
        self.norm=nn.LayerNorm(dim)
        self.qk=nn.Linear(dim,in_channels*2)
        self.scale = (in_channels/num_heads) ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.conv=nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1)
        
    def forward(self,inp):
        #input size: [B*A*A,C,H,W]
        N,C,H,W=inp.shape
        B=N//(self.an*self.an)
        x=self.pool(inp) #[B*A*A,C,S,S]
        x=x.flatten(1)  #[B*A*A,C*S*S]
        x=self.norm(x)  #[B*A*A,C*S*S]
        x=x.reshape(B,self.an*self.an,-1)  #[B,A*A,C*S*S]
        qk=self.qk(x).reshape(B,self.an*self.an,2,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)   #[2,B,head,A*A,C/head]
        q,k = qk.unbind(0)   #[B,head,A*A,C/head]
        att = (q @ k.transpose(-2, -1))*self.scale  #[B,head,A*A,A*A]
        att = self.softmax(att)   #[B,head,A*A,A*A]
        v=inp.reshape(B,self.an*self.an,self.num_heads,-1).permute(0,2,1,3)  #[B,head,A*A,(C/head)*H*W]
        x=(att@v).permute(0,2,1,3).reshape(B*self.an*self.an,C,H,W)  #[B*A*A,C,H,W]
        out=inp+self.conv(x)   #[B*A*A,C,H,W]
        
        return out
 
    
class MSWinAttention(nn.Module):
    def __init__(self,dim,att_drop=0,proj_drop=0):
        super(MSWinAttention,self).__init__()
        self.num_heads = 4
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim,dim)
        
        self.sr1=nn.Conv2d(dim, 2*dim//self.num_heads, kernel_size=8, stride=8)
        self.norm1=nn.LayerNorm(2*dim//self.num_heads)
        self.kv1 = nn.Linear(2*dim//self.num_heads, 2*dim//self.num_heads)
        
        self.sr2=nn.Conv2d(dim, 2*dim//self.num_heads, kernel_size=4, stride=4)
        self.norm2=nn.LayerNorm(2*dim//self.num_heads)
        self.kv2 = nn.Linear(2*dim//self.num_heads, 2*dim//self.num_heads)
        
        self.sr3=nn.Conv2d(dim, 2*dim//self.num_heads, kernel_size=4, stride=4)
        self.norm3=nn.LayerNorm(2*dim//self.num_heads)
        self.kv3 = nn.Linear(2*dim//self.num_heads, 2*dim//self.num_heads)
        
        self.sr4=nn.Conv2d(dim, 2*dim//self.num_heads, kernel_size=2, stride=2)
        self.norm4=nn.LayerNorm(2*dim//self.num_heads)
        self.kv4 = nn.Linear(2*dim//self.num_heads, 2*dim//self.num_heads)
        
        self.act = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)
        self.att_drop = nn.Dropout(att_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def window_partition(self,inp,n):
        #input size: [B*A*A,C,H,W]
        N,C,H,W = inp.shape
        x = inp.reshape(N, C, n, H//n, n, W//n)
        x = x.permute(0,2,4,1,3,5).reshape(-1,C,H//n,W//n)  #[B*A*A*n*n,C,H/n,W/n]
        return x
    
    def window_reverse(self,inp,n,H,W):
        #input size: [B*A*A*n*n,(H/n)*(W/n),C]
        N=int(inp.shape[0]/(n**2))
        x=inp.reshape(N,n,n,H//n,W//n,-1).permute(0,1,3,2,4,5)  #[B*A*A,n,H/n,n,W/n,C]
        x=x.reshape(N,H*W,-1)  #[B*A*A,H*W,C]
        return x
        
    def forward(self,inp):
        #input size: [B*A*A,C,H,W]
        N,C,H,W=inp.shape
        x=inp.flatten(2).transpose(1,2) #[B*A*A,H*W,C]
        x=self.norm(x)
        q=self.q(x).reshape(N,H*W,self.num_heads,C//self.num_heads).permute(2, 0, 1, 3)  #[4,B*A*A,H*W,C/4]
        q1,q2,q3,q4=q[0],q[1],q[2],q[3]  #[B*A*A,H*W,C/4]
        #Head 1
        x1=self.act(self.norm1(self.sr1(inp).reshape(N,2*C//self.num_heads,-1).permute(0,2,1)))  #[B*A*A,(H/8)*(W/8),C/2]
        kv1=self.kv1(x1).reshape(N, -1, 2, C//self.num_heads).permute(2, 0, 1, 3)  #[2,B*A*A,(H/8)*(W/8),C/4]
        k1,v1=kv1[0],kv1[1]  #[B*A*A,(H/8)*(W/8),C/4]
        att1=(q1 @ k1.transpose(-2,-1))*self.scale  #[B*A*A,H*W,(H/8)*(W/8)]
        att1 = self.softmax(att1)  
        att1 = self.att_drop(att1)  #[B*A*A,H*W,(H/8)*(W/8)]
        head1 = att1 @ v1    #[B*A*A,H*W,C/4]
        #Head 2
        num_windows=4
        n=int(num_windows**0.5)
        q2=q2.reshape(N,n,H//n,n,W//n,-1).permute(0,1,3,2,4,5).reshape(-1,(H//n)*(W//n),C//self.num_heads)  #[B*A*A*4,(H/2)*(W/2),C/4]
        x2=self.window_partition(inp,n)  #[B*A*A*4,C,H/2,W/2]
        x2=self.act(self.norm2(self.sr2(x2).reshape(N*num_windows,2*C//self.num_heads,-1).permute(0,2,1)))   #[B*A*A*4,(H/8)*(W/8),C/2]  
        kv2=self.kv2(x2).reshape(N*num_windows, -1, 2, C//self.num_heads).permute(2, 0, 1, 3)  #[2,B*A*A*4,(H/8)*(W/8),C/4]
        k2,v2=kv2[0],kv2[1]  #[B*A*A*4,(H/8)*(W/8),C/4]
        att2=(q2 @ k2.transpose(-2,-1))*self.scale  #[B*A*A*4,(H/2)*(W/2),(H/8)*(W/8)]
        att2=self.softmax(att2) 
        att2=self.att_drop(att2)  #[B*A*A*4,(H/2)*(W/2),(H/8)*(W/8)]
        head2=att2 @ v2  #[B*A*A*4,(H/2)*(W/2),C/4]
        head2=self.window_reverse(head2,n,H,W)  #[B*A*A,H*W,C/4]
        #Head 3
        num_windows=16
        n=int(num_windows**0.5)
        q3=q3.reshape(N,n,H//n,n,W//n,-1).permute(0,1,3,2,4,5).reshape(-1,(H//n)*(W//n),C//self.num_heads)  #[B*A*A*16,(H/4)*(W/4),C/4]
        x3=self.window_partition(inp,n)  #[B*A*A*16,C,H/4,W/4]
        x3=self.act(self.norm3(self.sr3(x3).reshape(N*num_windows,2*C//self.num_heads,-1).permute(0,2,1)))   #[B*A*A*16,(H/16)*(W/16),C/2]
        kv3=self.kv3(x3).reshape(N*num_windows, -1, 2, C//self.num_heads).permute(2, 0, 1, 3)  #[2,B*A*A*16,(H/16)*(W/16),C/4]
        k3,v3=kv3[0],kv3[1]  #[B*A*A*16,(H/16)*(W/16),C/4]
        att3=(q3 @ k3.transpose(-2,-1))*self.scale  #[B*A*A*16,(H/4)*(W/4),(H/16)*(W/16)]
        att3=self.softmax(att3) 
        att3=self.att_drop(att3)  #[B*A*A*16,(H/4)*(W/4),(H/16)*(W/16)]
        head3=att3 @ v3  #[B*A*A*16,(H/4)*(W/4),C/4]
        head3=self.window_reverse(head3,n,H,W)  #[B*A*A,H*W,C/4]
        #Head 4
        num_windows=64
        n=int(num_windows**0.5)
        q4=q4.reshape(N,n,H//n,n,W//n,-1).permute(0,1,3,2,4,5).reshape(-1,(H//n)*(W//n),C//self.num_heads)  #[B*A*A*64,(H/8)*(W/8),C/4]
        x4=self.window_partition(inp,n)  #[B*A*A*64,C,H/8,W/8]
        x4=self.act(self.norm4(self.sr4(x4).reshape(N*num_windows,2*C//self.num_heads,-1).permute(0,2,1)))   #[B*A*A*64,(H/16)*(W/16),C/2]
        kv4=self.kv4(x4).reshape(N*num_windows, -1, 2, C//self.num_heads).permute(2, 0, 1, 3)  #[2,B*A*A*64,(H/16)*(W/16),C/4]
        k4,v4=kv4[0],kv4[1]  #[B*A*A*64,(H/16)*(W/16),C/4]
        att4=(q4 @ k4.transpose(-2,-1))*self.scale  #[B*A*A*64,(H/8)*(W/8),(H/16)*(W/16)]
        att4=self.softmax(att4) 
        att4=self.att_drop(att4)  #[B*A*A*64,(H/8)*(W/8),(H/16)*(W/16)]
        head4=att4 @ v4  #[B*A*A*64,(H/8)*(W/8),C/4]
        head4=self.window_reverse(head4,n,H,W)  #[B*A*A,H*W,C/4]
        
        x=torch.cat([head1,head2,head3,head4],dim=-1)  #[B*A*A,H*W,C]
        x=self.proj(x)  #[B*A*A,H*W,C]
        out=self.proj_drop(x)  #[B*A*A,H*W,C]
        
        return out
    
    
class MLP(nn.Module):
    def __init__(self,in_features,hid_features,drop):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(in_features,hid_features)
        self.dwconv = nn.Conv2d(hid_features,hid_features,kernel_size=3,stride=1,padding=1,groups=hid_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_features,in_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self,inp,H,W):
        #input size: [B*A*A,H*W,C]
        N, _, C = inp.shape
        x=self.fc1(inp)  #[B*A*A,H*W,C]
        x = x.transpose(1, 2).reshape(N, C, H, W)  #[B*A*A,C,H,W]
        x=self.dwconv(x)  #[B*A*A,C,H,W]
        x = x.flatten(2).transpose(1, 2)  #[B*A*A,H*W,C]
        x=self.act(x)
        x = self.drop(x)
        x = self.fc2(x)   #[B*A*A,H*W,C]
        out = self.drop(x)
        
        return out
 
        
class FormerBlock(nn.Module):
    def __init__(self,dim,drop=0,att_drop=0,mlp_ratio=1):
        super(FormerBlock,self).__init__()
        self.att=MSWinAttention(dim,att_drop,drop)
        self.norm=nn.LayerNorm(dim)
        mlp_hid_dim = int(dim * mlp_ratio)
        self.mlp=MLP(dim,mlp_hid_dim,drop)
        
    def forward(self,inp):
        #input size: [B*A*A,C,H,W]
        N,C,H,W=inp.shape
        x=inp.flatten(2).transpose(1,2) #[B*A*A,H*W,C]
        x=x+self.att(inp)  #[B*A*A,H*W,C]
        out=x+self.mlp(self.norm(x),H,W)  #[B*A*A,H*W,C]
        out=out.reshape(N,H,W,-1).permute(0,3,1,2)  #[B*A*A,C,H,W]
        
        return out

    
class LRT(nn.Module):
    def __init__(self,in_channels,base_channels,an,num_formers):
        super(LRT,self).__init__()
        num_channels=[base_channels, base_channels*2, base_channels*4, base_channels*8]
        self.spatial_block1=ResBlock(in_channels,num_channels[0],stride=1)
        
        self.spatial_block2=ResBlock(num_channels[0],num_channels[1],stride=2)
        self.angular_former1=AngularFormer(num_channels[1],an,pool_size=4,num_heads=4)
        
        self.spatial_block3=ResBlock(num_channels[1],num_channels[2],stride=2)
        self.angular_former2=AngularFormer(num_channels[2],an,pool_size=2,num_heads=8)
        
        self.spatial_block4=ResBlock(num_channels[2],num_channels[3],stride=2)
        former_list=[]
        for _ in range(num_formers):
            former_list.append(FormerBlock((num_channels[3])))
        self.former_list=nn.ModuleList(former_list)
        
        self.skip1=UpSkip(num_channels[3],num_channels[2],mode='sum')
        self.spatial_block5=ResBlock(num_channels[2],num_channels[2],stride=1)
        self.angular_former3=AngularFormer(num_channels[2],an,pool_size=2,num_heads=8)
        self.noise=nn.Sequential(nn.Conv2d(num_channels[1],in_channels,kernel_size=3,stride=1,padding=1),
                                  nn.Tanh())
        self.illum=nn.Sequential(nn.Conv2d(num_channels[1],1,kernel_size=3,stride=1,padding=1),
                                  nn.Sigmoid())
        
        self.skip2=UpSkip(num_channels[2],num_channels[1],mode='sum')
        self.spatial_block6=ResBlock(num_channels[1],num_channels[1],stride=1)
        self.angular_former4=AngularFormer(num_channels[1],an,pool_size=4,num_heads=4)
        self.res1=nn.Sequential(nn.Conv2d(num_channels[1],in_channels,kernel_size=3,stride=1,padding=1),
                                nn.Tanh())
        
        self.skip3=UpSkip(num_channels[1],num_channels[0],mode='sum')
        self.spatial_block7=ResBlock(num_channels[0],num_channels[0],stride=1)
        self.res2=nn.Sequential(nn.Conv2d(num_channels[0]//2,in_channels,kernel_size=3,stride=1,padding=1),
                                nn.Tanh())
        self.pool=nn.AdaptiveAvgPool2d(8)
        self.fc1=nn.Linear(8*8,16)
        self.fc2=nn.Sequential(nn.Linear(16,1),nn.Sigmoid())
        self.hf=nn.Sequential(nn.Conv2d(num_channels[0]//2+3,in_channels,kernel_size=3,stride=1,padding=1),
                                nn.Tanh())
        
        
    def forward(self,inp):
        #input size: [B,A*A,3,H,W]
        B,view_num,C,H,W=inp.size()
        x=inp.reshape(B*view_num,C,H,W)  #[B*A*A,3,H,W]
        x_blur=GaussianBlur(kernel_size=3,sigma=0.5)(x)
        x_d2=Resize(size=(H//2,W//2))(x_blur)  #[B*A*A,3,H/2,W/2]
        x_blur2=GaussianBlur(kernel_size=3,sigma=0.5)(x_d2)
        x_d4=Resize(size=(H//4,W//4))(x_blur2)  #[B*A*A,3,H/4,W/4]
        
        x1=self.spatial_block1(x)   #[B*A*A,16,H,W]
        
        x2=self.spatial_block2(x1)  #[B*A*A,32,H/2,W/2]
        x2=self.angular_former1(x2)  #[B*A*A,32,H/2,W/2]
        
        x3=self.spatial_block3(x2)  #[B*A*A,64,H/4,W/4]
        x3=self.angular_former2(x3)  #[B*A*A,64,H/4,W/4]
        
        x4=self.spatial_block4(x3)   #[B*A*A,128,H/8,W/8]
        for former in self.former_list:
            x4=former(x4)  #[B*A*A,128,H/8,W/8]
            
        x3=self.skip1(x4,x3)  #[B*A*A,64,H/4,W/4]
        x3=self.spatial_block5(x3)  #[B*A*A,64,H/4,W/4]
        x3=self.angular_former3(x3)  #[B*A*A,64,H/4,W/4]
        feat_num=x3.shape[1]//2
        noise=self.noise(x3[:,:feat_num,:,:])  #[B*A*A,3,H/4,W/4]
        denoise_d4=x_d4+0.2*noise
        denoise_d4=torch.clamp(denoise_d4,0,1)  #[B*A*A,3,H/4,W/4]
        I_d4=self.illum(x3[:,feat_num:,:,:])   #[B*A*A,1,H/4,W/4]
        re_d4=denoise_d4*(1/I_d4)   
        re_d4=torch.clamp(re_d4,0,1)  #[B*A*A,3,H/4,W/4]
        
        x2=self.skip2(x3,x2)  #[B*A*A,32,H/2,W/2]
        x2=self.spatial_block6(x2)  #[B*A*A,32,H/2,W/2]
        x2=self.angular_former4(x2)  #[B*A*A,32,H/2,W/2]
        res1=self.res1(x2)  #[B*A*A,3,H/2,W/2]
        re_d2=Resize(size=(H//2,W//2))(re_d4)+res1
        re_d2=torch.clamp(re_d2,0,1)  #[B*A*A,3,H/2,W/2]
        
        x1=self.skip3(x2,x1)  #[B*A*A,16,H,W]
        x1=self.spatial_block7(x1)  #[B*A*A,16,H,W]
        feat_num=x1.shape[1]//2
        res2=self.res2(x1[:,:feat_num,:,:])  #[B*A*A,3,H,W]
        re=Resize(size=(H,W))(re_d2)+res2
        re=torch.clamp(re,0,1)  #[B*A*A,3,H,W]
        
        I=I_d4.reshape(B,view_num,1,H//4,W//4)
        I=I[:,(view_num-1)//2,:,:,:]  #[B,1,H/4,W/4]
        r=self.pool(I).flatten(1) #[B,16*16]
        r=self.fc2(self.fc1(r))*3+1  #[B,1]
        r=r.reshape(B,1,1,1,1)
        inp_adjust=(inp*r).reshape(B*view_num,C,H,W)  #[B*A*A,3,H,W]
        temp=torch.cat([x1[:,feat_num:,:,:],inp_adjust],dim=1) #[B*A*A,11,H,W]
        hf=self.hf(temp)  #[B*A*A,3,H,W]
        out=re+hf
        out=torch.clamp(out,0,1)  #[B*A*A,3,H,W]
        
        return out,re,re_d2,re_d4,denoise_d4,I_d4,hf
        
        
        
        
    
