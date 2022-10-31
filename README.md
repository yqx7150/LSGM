# LSGM
Paper: Lens-less imaging via score-based generative model 
https://www.opticsjournal.net/M/Articles/OJf1842c2819a4fa2e/Abstract

dataset

The dataset used to train the model in this experiment is  LSUN-bedroom and  LSUN-church.

place the dataset in the train file under the church folder.

train:

python main.py --config=configs/ve/church_ncsnpp_continuous.py  --workdir=exp_train_church_max1_N1000 --mode=train --eval_folder=result


test:

python score_sde_fza_demo_fujian.py
