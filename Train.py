import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from networks.LRT import LRT
from Dataset import TrainData, ValData
from Loss import MSLoss

from utils.metrics import PSNR, SSIM


def adjust_learning_rate(optimizer):
    lr=optimizer.param_groups[0]['lr']
    lr=lr*0.9
    optimizer.param_groups[0]['lr']=lr
        
        
def train(args):
    lr_epoch=50
    epoch_dict,psnr_dict,ssim_dict={},{},{}
    for i in range(1,args.save_num+1):
        epoch_dict[str(i)]=0
        psnr_dict[str(i)]=0
        ssim_dict[str(i)]=0
    best_psnr=0
    best_ssim=0
    best_epoch=0
    
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root,exist_ok=True)
            
    train_set=TrainData(args.data_root,crop_size=args.crop_size)
    val_set=ValData(args.data_root)
    
    train_loader = DataLoader(dataset=train_set,batch_size=args.batch_size,shuffle=True)
    val_loader = DataLoader(dataset=val_set,batch_size=args.batch_size,shuffle=False)
    test_num=len(test_set)
    
    model=LRT(in_channels=3,base_channels=16,an=7,num_formers=1)
    print('model parameters: {}'.format(sum(param.numel() for param in model.parameters())))
    
    criterion=MSLoss()
    
    model.cuda() 
    #criterion.cuda()
    
    if args.pre_train:
        model.load_state_dict(torch.load(args.model_path))
        
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr_init)
    
    for epoch in range(args.max_epoch):
        model.train()
        if epoch % lr_epoch==0 and epoch!=0:
            adjust_learning_rate(optimizer)
        
        for idx,(inp,gt,gt_d2,gt_d4,low_d4,I_gt_d4,hf_gt) in enumerate(train_loader):
            inp,gt,gt_d2,gt_d4,low_d4,I_gt_d4,hf_gt=Variable(inp).cuda(),Variable(gt).cuda(),Variable(gt_d2).cuda(),Variable(gt_d4).cuda(),Variable(low_d4).cuda(),Variable(I_gt_d4).cuda(),Variable(hf_gt).cuda()
            optimizer.zero_grad()
            out,re,re_d2,re_d4,denoise_d4,I_d4,hf=model(inp)
            loss=criterion(out,re,re_d2,re_d4,denoise_d4,I_d4,hf,gt,gt_d2,gt_d4,low_d4,I_gt_d4,hf_gt)
            loss.backward()
            optimizer.step()
            print('Epoch: %i, step: %i, train_loss: %f' %(epoch,idx,loss.item()))
            print('')
        
        model.eval()
        with torch.no_grad():
            total_psnr=0
            total_ssim=0
            
            for inp,gt in val_loader:
                inp=inp.cuda()
                pred=model(inp)[0]
                pred=pred.cpu().numpy()
                gt=gt.numpy()
                B=gt.shape[0]
                N,C,H,W=pred.shape
                pred=pred.reshape(B,N//B,C,H,W)
                view_num=N//B
                
                for i in range(B):
                    one_psnr,one_ssim=0,0
                    one_pred=pred[i]
                    one_gt=gt[i]
                    for j in range(view_num):
                        p=one_pred[j]
                        g=one_gt[j]
                        psnr=PSNR(g,p,data_range=1)
                        ssim=SSIM(g,p,data_range=1)
                        one_psnr+=psnr
                        one_ssim+=ssim
                    one_psnr=one_psnr/view_num
                    one_ssim=one_ssim/view_num
                    total_psnr+=one_psnr
                    total_ssim+=one_ssim
                        
            ave_psnr=total_psnr/test_num
            ave_ssim=total_ssim/test_num
            print('Epoch {}, average PSNR {}, SSIM {}'.format(epoch,ave_psnr,ave_ssim))
            print('')
    
        #save models            
        if epoch<args.save_num:
            torch.save(model.state_dict(),os.path.join(args.save_root,'model%s.pth'%str(epoch+1)))
            psnr_dict[str(epoch+1)]=ave_psnr
            ssim_dict[str(epoch+1)]=ave_ssim
            epoch_dict[str(epoch+1)]=epoch
        else:
            if ave_psnr>min(psnr_dict.values()):
                torch.save(model.state_dict(),
                            os.path.join(args.save_root,'model%s.pth'%(min(psnr_dict,key=lambda x: psnr_dict[x]))))
                epoch_dict[min(psnr_dict,key=lambda x: psnr_dict[x])]=epoch
                ssim_dict[min(psnr_dict,key=lambda x: psnr_dict[x])]=ave_ssim
                psnr_dict[min(psnr_dict,key=lambda x: psnr_dict[x])]=ave_psnr                
        if ave_psnr>best_psnr:
            best_psnr=ave_psnr
            best_ssim=ave_ssim 
            best_epoch=epoch
        print('Best PSNR {}, SSIM {}, epoch {}'.format(best_psnr,best_ssim,best_epoch))
        print('Epoch {}'.format(epoch_dict))
        print('PSNR {}'.format(psnr_dict))
        print('SSIM {}'.format(ssim_dict))
        print('')
        

if __name__=='__main__':  
    parser = argparse.ArgumentParser(description='LRT')
    parser.add_argument('--data_root', default='', type=str)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--lr_init', default=1e-3, type=float)
    parser.add_argument('--crop_size', default=256, type=int)
    parser.add_argument('--max_epoch',default=300,type=int)
    parser.add_argument('--save_num', default=10, type=int, help='number of saved models')
    parser.add_argument('--save_root',default='',type=str)
    parser.add_argument('--pre_train',default=False,type=bool)
    parser.add_argument('--model_path',default='',type=str)
    args = parser.parse_known_args()[0]
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    train(args)
