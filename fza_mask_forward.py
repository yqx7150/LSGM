from fza_mask import fza
import torch
import numpy as np
import cv2
def fza_mask_forward(img):
  masked_data_0=img[0,0,:,:].detach().cpu().numpy()
  masked_data_1=img[0,1,:,:].detach().cpu().numpy()
  masked_data_2=img[0,2,:,:].detach().cpu().numpy()
  di=3
  z1=300
  LX1=200
  dp=0.014
  NX,NY=256,256
  S=2*dp*NX
  r1=0.23
  M=di/z1
  ri=(1+M)*r1
  I_0=cv2.filter2D(masked_data_0.astype('float32'), -1, fza(), borderType=cv2.BORDER_CONSTANT)*2*dp*dp/ri**2
  I_1=cv2.filter2D(masked_data_1.astype('float32'), -1, fza(), borderType=cv2.BORDER_CONSTANT)*2*dp*dp/ri**2
  I_2=cv2.filter2D(masked_data_2.astype('float32'), -1, fza(), borderType=cv2.BORDER_CONSTANT)*2*dp*dp/ri**2
  forward_fza_0=I_0-np.mean(I_0[:])
  forward_fza_1=I_1-np.mean(I_1[:])
  forward_fza_2=I_2-np.mean(I_2[:])
  forward_fza_0=torch.as_tensor(forward_fza_0).cuda()
  forward_fza_1=torch.as_tensor(forward_fza_1).cuda()
  forward_fza_2=torch.as_tensor(forward_fza_2).cuda()
  
  return forward_fza_0,forward_fza_1,forward_fza_2
