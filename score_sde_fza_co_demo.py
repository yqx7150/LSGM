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
import controllable_generation_fza_co
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
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling_fza import (ReverseDiffusionPredictor, 
                      LangevinCorrector, 
                      EulerMaruyamaPredictor, 
                      AncestralSamplingPredictor, 
                      NoneCorrector, 
                      NonePredictor,
                      AnnealedLangevinDynamics)
import datasets
import scipy.io as io
from operator_fza import forward,backward,forward_torch,backward_torch


# @title Load the score-based model
sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde.lower() == 'vesde':
  #from configs.ve import cifar10_ncsnpp as configs
  #ckpt_filename = "exp/ve/cifar10_ncsnpp/checkpoint_16.pth"
  #from configs.ve import ffhq_256_ncsnpp_continuous as configs
  #ckpt_filename = "exp/ve/ffhq_256_ncsnpp_continuous/checkpoint_48.pth"
  from configs.ve import bedroom_ncsnpp_continuous as configs
  ckpt_filename = "exp/ve/bedroom_ncsnpp_continuous/checkpoint_127.pth"
  config = configs.get_config() 
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



#@title PC coloring
#train_ds, eval_ds, _ = datasets.get_dataset(config)
#eval_iter = iter(eval_ds)
#bpds = []

predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
snr = 0.16 #@param {"type": "number"}
n_steps = 1 #@param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}

img=io.loadmat('./test_img/Img_bed_256.mat')['Img']
img_ob=io.loadmat('./test_img/ob_bed_256.mat')['ob']

img=np.expand_dims(img,axis=0)
img_ob=np.expand_dims(img_ob,axis=0)

img = torch.from_numpy(img).permute(0, 3, 1, 2).to(config.device) #1,3,256,256
img_ob = torch.from_numpy(img_ob).permute(0, 3, 1, 2).to(config.device) #1,3,256,256

show_samples(img)
dp=0.014
di=3
z1=20
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

img_forward=forward(img[0,0,:,:],H).cpu().numpy()
plt.figure(figsize=(8,8))
plt.axis('off')
plt.imshow(img_forward,cmap='gray')
plt.show()

#gray_scale_img = torch.mean(img, dim=1, keepdims=True).repeat(1, 3, 1, 1)
img_ob_gray=torch.mean(img_ob, dim=1, keepdims=True).repeat(1, 3, 1, 1)
img_backward=backward(img_ob_gray[0,0,:,:],H).cpu().numpy()
plt.figure(figsize=(8,8))
plt.axis('off')
plt.imshow(img_backward,cmap='gray')
plt.show()

show_samples(img_ob_gray)
#gray_scale_img = scaler(gray_scale_img)
pc_colorizer = controllable_generation_fza_co.get_pc_colorizer(
    sde, predictor, corrector, inverse_scaler,
    snr=snr, n_steps=n_steps, probability_flow=probability_flow,
    continuous=config.training.continuous, denoise=True
)
x = pc_colorizer(score_model, img_ob_gray.float(),H,img)

 
show_samples(x)





