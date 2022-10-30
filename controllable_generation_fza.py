from models import utils as mutils
import torch
import numpy as np
from sampling_fza import NoneCorrector, NonePredictor, shared_corrector_update_fn, shared_predictor_update_fn
import functools
from operator_fza import forward,backward
import matplotlib.pyplot as plt
import cv2
#from skimage.measure import compare_psnr,compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import scipy.io as io
from fza_mask import fza

def get_pc_inpainter(sde, predictor, corrector, inverse_scaler, snr,
                     n_steps=1, probability_flow=False, continuous=False,
                     denoise=True, eps=1e-5):
  # Define predictor & corrector
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def get_inpaint_update_fn(update_fn):
    """Modify the update function of predictor & corrector to incorporate data information."""

    def inpaint_update_fn(model, data, mask, x, t,data_ob):
      with torch.no_grad():
        vec_t = torch.ones(data.shape[0], device=data.device) * t
        x, x_mean = update_fn(x, vec_t, model=model) 
        #print(data_ob.transpose(0,2).unsqueeze(0).shape,data_ob.transpose(0,2).unsqueeze(0).dtype)  
        masked_data_mean, std = sde.marginal_prob(data, vec_t) #std = self.sigma_min * (self.sigma_max / self.sigma_min) ** vec_t  mean = data
        masked_data = masked_data_mean + torch.randn_like(x) * std[:, None, None, None]
        '''
        masked_data_0=masked_data[0,0,:,:].detach().cpu().numpy()
        masked_data_1=masked_data[0,1,:,:].detach().cpu().numpy()
        masked_data_2=masked_data[0,2,:,:].detach().cpu().numpy()
        di=3
        z1=20
        LX1=15
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
        '''
        a=1
       
        ### x
        x_red=x[0,0,:,:]+a*backward(forward(masked_data[0,0,:,:],mask)-forward(x[0,0,:,:],mask),mask)
        x_green=x[0,1,:,:]+a*backward(forward(masked_data[0,1,:,:],mask)-forward(x[0,1,:,:],mask),mask)
        x_blue=x[0,2,:,:]+a*backward(forward(masked_data[0,2,:,:],mask)-forward(x[0,2,:,:],mask),mask)
        x=torch.stack((x_red,x_green,x_blue),dim=0).unsqueeze(0)
        #x=torch.as_tensor(x,dtype=torch.float32).cuda()
       
       
        ###x_mean  这是在ncsn中用的保真项 x=x_mean(单独使用）
        x_red_mean=x[0,0,:,:]+a*backward(data_ob[:,:,0]-forward(x[0,0,:,:],mask),mask)
        x_green_mean=x[0,1,:,:]+a*backward(data_ob[:,:,1]-forward(x[0,1,:,:],mask),mask)
        x_blue_mean=x[0,2,:,:]+a*backward(data_ob[:,:,2]-forward(x[0,2,:,:],mask),mask)
        x_mean=torch.stack((x_red_mean,x_green_mean,x_blue_mean),dim=0).unsqueeze(0)

        #x = x * (1. - mask) + masked_data * mask
        #x_mean=torch.as_tensor(x_mean,dtype=torch.float32).cuda()
        #x=x_mean
        #cv2.imwrite('fza_inpainting.png',x_mean[0,:,:,:].cpu().numpy().transpose(1,2,0)*255)
        '''
        plt.ion()
        plt.axis('off')
        plt.imshow(x_mean[0,:,:,:].cpu().numpy().transpose(1,2,0),cmap='gray')
        plt.show()
        plt.pause(0.1)
        plt.close()
        '''
        #x_mean = x * (1. - mask) + masked_data_mean * mask
        return x, x_mean

    return inpaint_update_fn

  projector_inpaint_update_fn = get_inpaint_update_fn(predictor_update_fn)
  corrector_inpaint_update_fn = get_inpaint_update_fn(corrector_update_fn)

  def pc_inpainter(model, data, mask,data_ob):

    with torch.no_grad():
      '''
      # Initial sample
      #x = data * mask + sde.prior_sampling(data.shape).to(data.device) * (1. - mask)  #torch.randn(*shape) * self.sigma_max=sde.prior_sampling
      x_red=sde.prior_sampling(data.shape).to(data.device)[0,0,:,:]+backward(data_ob[:,:,0]-forward(sde.prior_sampling(data.shape).to(data.device)[0,0,:,:],mask),mask)
      x_green=sde.prior_sampling(data.shape).to(data.device)[0,1,:,:]+backward(data_ob[:,:,1]-forward(sde.prior_sampling(data.shape).to(data.device)[0,1,:,:],mask),mask)
      x_blue=sde.prior_sampling(data.shape).to(data.device)[0,2,:,:]+backward(data_ob[:,:,2]-forward(sde.prior_sampling(data.shape).to(data.device)[0,2,:,:],mask),mask)
      x=torch.stack((x_red,x_green,x_blue),dim=0).unsqueeze(0)
      #x=torch.as_tensor(x,dtype=torch.float32).cuda() #-1722.2703--1961.4119
      '''
      
      x_red=backward(data_ob[:,:,0],mask)
      x_green=backward(data_ob[:,:,1],mask)
      x_blue=backward(data_ob[:,:,2],mask) 
      x=torch.stack((x_red,x_green,x_blue),dim=0).unsqueeze(0) #(-1,1)
      '''
      x=(x-x.min())/(x.max()-x.min())
      data_ob=(data_ob-data_ob.min())/(data_ob.max()-data_ob.min())
      '''
      '''
      plt.axis('off')
      plt.imshow(x[0,:,:,:].cpu().numpy().transpose(1,2,0),cmap='gray')
      plt.show()
      '''
      #x=torch.as_tensor(x,dtype=torch.float32).cuda() #3张ob的灰度图  (-1,1)
      
      timesteps = torch.linspace(sde.T, eps, sde.N) #1 1e-05 N #sde.T
      psnr_max=0
      ssim_max=0

      for i in range(sde.N):
        t = timesteps[i]
        x, x_mean = corrector_inpaint_update_fn(model, data, mask, x, t,data_ob)
        x, x_mean = projector_inpaint_update_fn(model, data, mask, x, t,data_ob)
        
        x_mean_save=x_mean.cpu().numpy()
        #x_mean_save=(x_mean_save-x_mean_save.min())/(x_mean_save.max()-x_mean_save.min()) #数值比较高
        #x_mean_save=np.clip(x_mean_save,0,1)
        x_mean_save_cv=x_mean_save[0,:,:,:].transpose(1,2,0)
        data_ori=data[0,:,:,:].cpu().numpy().transpose(1,2,0)
        psnr=compare_psnr(np.abs(x_mean_save_cv[49:208,49:208,:])*255,np.abs(data_ori[49:208,49:208,:])*255,data_range=255)
        ssim=compare_ssim(np.abs(x_mean_save_cv[49:208,49:208,:]),np.abs(data_ori[49:208,49:208,:]),multichannel=True,data_range=1)
        filename = 'psnr_ssim.txt'
        with open (filename,'a') as file_object:
          file_object.write("迭代次数:"+str(i)+"  psnr:"+str(psnr)+"  ssim:"+str(ssim)+"\n") 
        print('外循环:',i,'psnr:',psnr,'ssim:',ssim)
        if psnr>psnr_max:
          psnr_max=psnr
          ssim_max=ssim
          print('psnr_max',psnr_max)
          x_mean_save_img=np.stack((x_mean_save_cv[:,:,2],x_mean_save_cv[:,:,1],x_mean_save_cv[:,:,0]),axis=2)
          #x_mean_save_img=(x_mean_save_img-x_mean_save_img.min())/(x_mean_save_img.max()-x_mean_save_img.min())
          data_ori_img=np.stack((data_ori[:,:,2],data_ori[:,:,1],data_ori[:,:,0]),axis=2)
          cv2.imwrite('fza_inpainting.png',x_mean_save_img*255)
          cv2.imwrite('59.png',data_ori_img*255)
        
      return inverse_scaler(x_mean_save_img if denoise else x),psnr_max,ssim_max
  return pc_inpainter


