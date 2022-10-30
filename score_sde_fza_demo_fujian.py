#@title Autoload all modules
#%load_ext autoreload
#%autoreload 2

from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
import functools
import itertools
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import io
import likelihood
from utils import restore_checkpoint
sns.set(font_scale=2)
sns.set(style="whitegrid")
import cv2
import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling_fza
import sampling_pc
#import sampling_3noise
import sampling_3noise_fujian
import sampling_3noise_fujian_mask
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling_3noise_fujian_mask import (ReverseDiffusionPredictor, 
                      LangevinCorrector, 
                      EulerMaruyamaPredictor, 
                      AncestralSamplingPredictor, 
                      NoneCorrector, 
                      NonePredictor,
                      AnnealedLangevinDynamics)
import datasets
import scipy.io as io
from operator_fza import forward,backward,forward_torch,backward_torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# @title Load the score-based model
sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde.lower() == 'vesde':
  from configs.ve import church_ncsnpp_continuous as configs
  ckpt_filename = "exp_train_church_max380_N1000/checkpoints/checkpoint_9.pth" #(9:(20.2,0.5)
  sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
  sampling_eps = 1e-5
elif sde.lower() == 'vpsde':
  from configs.vp import cifar10_ddpmpp_continuous as configs  
  ckpt_filename = "exp/vp/cifar10_ddpmpp_continuous/checkpoint_8.pth"
  config = configs.get_config()
  sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  sampling_eps = 1e-3
elif sde.lower() == 'subvpsde':
  from configs.subvp import cifar10_ddpmpp_continuous as configs
  ckpt_filename = "exp/subvp/cifar10_ddpmpp_continuous/checkpoint_26.pth"
  config = configs.get_config()
  sde = subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  sampling_eps = 1e-3

batch_size =  1 #64#@param {"type":"integer"}
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = 0 #@param {"type": "integer"}

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device)

ema.copy_to(score_model.parameters())

#@title Visualization code

def image_grid(x):
  size = config.data.image_size
  channels = config.data.num_channels
  img = x.reshape(-1, size, size, channels)
  w = int(np.sqrt(img.shape[0]))
  img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
  #img = img.reshape(( size, size, channels*2))
  return img

def show_samples(x):
  x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
  img = image_grid(x)
  plt.figure(figsize=(8,8))
  plt.axis('off')
  plt.imshow(img)
  plt.show()



#@title PC inpainting

predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
snr = 0.16 #@param {"type": "number"}
n_steps = 1 #@param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}


psnr_result=[ ]
ssim_result=[ ]
for j in range(0,1,1):
  print('****************'+'第{}张图'.format(j+1)+'******************')
  img=io.loadmat('./input_output/input/church/church_Img.mat')['Img']

  img_ob=io.loadmat('./input_output/input/church/church_ob.mat')['ob']
  img_ob=torch.tensor(img_ob).cuda()
  img=np.expand_dims(img,axis=0)

  img = torch.from_numpy(img).permute(0, 3, 1, 2).to(config.device) #1,3,128,128

  dp=0.014
  di=3
  z1=300
  r1=0.23
  M=di/z1
  ri=(1+M)*r1



  NX,NY=256,256

  fu_max,fv_max=0.5/dp,0.5/dp
  du,dv=2*fu_max/NX,2*fv_max/NY
  u,v=np.mgrid[-fu_max:fu_max:du,-fv_max:fv_max:dv]
  u=u.T
  v=v.T
  H=1j*(np.exp(-1j*(np.dot(np.pi,ri**2))*(u**2+v**2)))
  H=np.array(H,dtype=np.complex128)
  #H=torch.tensor(H,dtype=torch.complex128).cuda()



  img_forward=backward(img_ob[:,:,0],H).cpu().numpy()#(-0.6,0.6)

  psnr_max_1=0
  for i in range(1):
    print('##################'+str(i)+'#######################')
     
    img_size = config.data.image_size
    channels = config.data.num_channels
    shape = (batch_size, channels, img_size, img_size)

    sampling_fn = sampling_3noise_fujian_mask.get_pc_sampler(sde, shape, predictor, corrector,
                                    inverse_scaler, snr, n_steps=n_steps,
                                    probability_flow=probability_flow,
                                    continuous=config.training.continuous,
                                    eps=sampling_eps, device=config.device)

    x,psnr_max,ssim_max = sampling_fn(score_model,img,H,img_ob)

    cv2.imwrite('./input_output/output/fza_{}_USAF_I_035.png'.format(j),x*255)
    #print('psnr_max_1',psnr_max)
  psnr_result.append(psnr_max)
  ssim_result.append(ssim_max)
  print('psnr_result',psnr_result)
  print('ssim_result',ssim_result)
psnr_result=sum(psnr_result)/(len(psnr_result))
ssim_result=sum(ssim_result)/(len(ssim_result))
print(psnr_result,ssim_result)





