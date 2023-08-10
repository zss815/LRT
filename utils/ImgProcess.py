import numpy as np
import cv2



def RGB2Bayer(img):
    ''' Mosacing to GRBG format Bayer pattern
        |G|R|
        |B|G|
    '''
    h, w, c = img.shape
    Bayer = np.zeros([h, w], dtype=np.uint8)
    Bayer[0::2, 0::2] = img[0::2, 0::2, 1]  # G
    Bayer[0::2, 1::2] = img[0::2, 1::2, 0]  # R
    Bayer[1::2, 0::2] = img[1::2, 0::2, 2]  # B
    Bayer[1::2, 1::2] = img[1::2, 1::2, 1]  # G
    return Bayer

    
def illum_adjust(img,v_scale):
    img_hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    #Brightness
    if v_scale:
        v=img_hsv[...,2]
        v=v.astype(np.float64)
        v=v*v_scale
        v[v>255]=255
        v=v.astype(np.uint8)
        img_hsv[...,2]=v
    img=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2RGB)
    return img


def row_noise_gen(clean_raw,row_scale):
    h,w=clean_raw.shape
    noise=np.random.normal(0,row_scale,(h,1))
    noise=np.repeat(noise,w,axis=1)
    return noise
    
    
def Noise_syn(clean_raw,k,read_scale,row_scale,q):
    shot_noise = np.random.poisson(clean_raw/k)*k  #large k, large shot noise
    read_noise=np.random.normal(0,read_scale,np.shape(clean_raw))
    #read_noise=tukeylambda.rvs(lam=0, loc=0, scale=read_scale, size=np.shape(clean_raw), random_state=None)
    row_noise=row_noise_gen(clean_raw,row_scale)
    quanti_noise=np.random.uniform(-q/2,q/2,np.shape(clean_raw))
    noise_raw=shot_noise+read_noise+row_noise+quanti_noise
    noise_raw[noise_raw<0]=0
    noise_raw=np.uint8(noise_raw)
    return noise_raw
  