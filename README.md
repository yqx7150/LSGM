# LSGM
**Paper**: Lens-less imaging via score-based generative model https://www.opticsjournal.net/M/Articles/OJf1842c2819a4fa2e/Abstract
**Authors**: Chunhua Wu, Hong Peng, Qiegen Liu, Senior Member, IEEE, Wenbo Wan, Yuhao Wang, Senior Member, IEEE

Date : October-31-2022  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2022, Department of Electronic Information Engineering, Nanchang University. 


Lens-less imaging is affected by twinning noise in in-line holograms, and the reconstructed results has always faced the poor reconstruction signal-to-noise ratio and low imaging resolution. In this paper, a lens-less imaging via score-based generation model is proposed. In the training phase, the proposed model perturbs the data distribution by slowly adding Gaussian noise using a continuous stochastic differential equation (SDE). Acontinuous-time dependent score-based function with denoising score matching is then trained and used to solve the inverse SDE to generate object sample data. In the test phase, a single Fresnel ZoneAperture is usedasamasktoachievelens-lessencodingmodulationunderincoherentillumination.Theprediction-correctionmethodisthenused to alternate iterations between the numerical SDE solver and data-fidelity term steps to achieve lens-less imaging reconstruction. Validation results on LSUN-bedroom and LSUN-church datasets show that the proposed algorithm can effectively eliminate twin image noise, andthepeaksignal-to-noiseratioand structural similarity ofthereconstructionresultscanreach 25.23dBand 0.65.The PSNR values of the reconstruction results are 17.49 dB and 7.16 dB higher than lens-less imaging algorithms based on traditional Back Propagation or Compressed Sensing, respectively. The corresponding SSIM values were 0.42 and 0.35 higher, respectively. Thereby,thereconstructionqualityofthelens-lessimagingiseffectivelyimproved. 

## Requirements and Dependencies
    python==3.7.11
    Pytorch==1.7.0
    tensorflow==2.4.0
    torchvision==0.8.0
    tensorboard==2.7.0
    scipy==1.7.3
    numpy==1.19.5
    ninja==1.10.2
    matplotlib==3.5.1
    jax==0.2.26

## Checkpoints
We provide pretrained checkpoints. You can download pretrained models from  [Baidu cloud] (https://pan.baidu.com/s/12SzKZRNefjNqOx_nW-2RAA))

## Dataset

The dataset used to train the model in this experiment is  LSUN-bedroom and  LSUN-church.

place the dataset in the train file under the church folder.

## Train:

python main.py --config=configs/ve/church_ncsnpp_continuous.py  --workdir=exp_train_church_max1_N1000 --mode=train --eval_folder=result


## Test:

python score_sde_fza_demo_fujian.py
