import numpy as np
import cv2


# MacroPixel to stacked SAIs
def MP2SAIs(x,ah,aw,resize=None):
    sai_all=[]
    for i in range(ah):
        for j in range(aw):
            img=x[i::ah, j::aw]
            if resize:
                img=cv2.resize(img,resize,interpolation=cv2.INTER_LINEAR)
            sai_all.append(img)
    sai_all=np.stack(sai_all)
    return sai_all


# MacroPixel to SAI array
def MP2Array(x,ah,aw,resize=None):
    sai_array = []
    for i in range(ah):
        out_h = []
        for j in range(aw):
            img=x[i::ah, j::aw]
            if resize:
                img=cv2.resize(img,resize,interpolation=cv2.INTER_LINEAR)
            out_h.append(img)
        sai_array.append(np.concatenate(out_h, 1))
    sai_array = np.concatenate(sai_array,0)
    return sai_array


# Stacked SAIs to MacroPixel
def SAIs2MP(x,ah,aw,h,w):
    MP=[]
    for i in range(h):
        out_h=[]
        for j in range(w):
            out_h.append(x[:,i,j].reshape(ah,aw,-1))
        MP.append(np.concatenate(out_h,1))
    MP=np.concatenate(MP,0)
    return MP


# Stacked SAIs to SAI array
def SAIs2Array(x,ah,aw):
    array=[]
    for i in range(ah):
        temp=x[i*aw:(i+1)*aw]
        temp=np.transpose(temp,axes=(1,0,2,3))
        temp=temp.reshape(temp.shape[0],temp.shape[1]*temp.shape[2],temp.shape[3])
        array.append(temp)
    array=np.concatenate(array,axis=0)
    return array

# SAI array to stacked SAIs
def Array2SAIs(x,ah,aw,h,w,resize):
    sais=[]
    for i in range(ah):
        for j in range(aw):
            sai=x[i*h:(i+1)*h,j*w:(j+1)*w]
            if resize:
                sai=cv2.resize(sai,resize,interpolation=cv2.INTER_LINEAR)
            sais.append(sai)
    sais=np.stack(sais)
    return sais
     

# SAI array to MacroPixel
def Array2MP(x,h,w):
    MP=[]
    for i in range(h):
        out_h=[]
        for j in range(w):
            out_h.append(x[i::h,j::w])
        MP.append(np.concatenate(out_h,1))
    MP=np.concatenate(MP,0)
    return MP


# Retain central n*n views
def MP2ArrayCrop(x,ah,aw,begin,num,resize=None):
    sai_array = []
    for i in range(begin,begin+num):
        out_h = []
        for j in range(begin,begin+num):
            img=x[i::ah, j::aw]
            if resize:
                img=cv2.resize(img,resize,interpolation=cv2.INTER_LINEAR)
            out_h.append(img)
        sai_array.append(np.concatenate(out_h, 1))
    sai_array = np.concatenate(sai_array,0)
    return sai_array


def MP2SAIsCrop(x,ah,aw,begin,num,resize=None):
    sai_all=[]
    for i in range(begin,begin+num):
        for j in range(begin,begin+num):
            img=x[i::ah, j::aw]
            if resize:
                img=cv2.resize(img,resize,interpolation=cv2.INTER_LINEAR)
            sai_all.append(img)
    sai_all=np.stack(sai_all)
    return sai_all


#EPI
def EPI(sai_array,ah,aw,is_row,idx,position):
    h=sai_array.shape[0]//ah
    w=sai_array.shape[1]//aw
    if is_row:
        p=(idx-1)*h+position
        epi_all=sai_array[p]
        epi=[]
        for i in range(aw):
            temp=epi_all[i*w:(i+1)*w]
            temp=np.expand_dims(temp,axis=0)
            epi.append(temp)
        epi=np.concatenate(epi,axis=0)
    else:
        p=(idx-1)*w+position
        epi_all=sai_array[:,p]
        epi=[]
        for i in range(ah):
            temp=epi_all[i*h:(i+1)*h]
            temp=np.expand_dims(temp,axis=1)
            epi.append(temp)
        epi=np.concatenate(epi,axis=1)
    return epi