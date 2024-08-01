#!/bin/bash

MODEL_PATH='./models/MAE_5.83_MSE_9.63_mae_5.83_mse_9.63_Ep_451.pth'

python3 test.py --dataset_name='SHB' --pretrain_model=${MODEL_PATH} --gpu_ids='0,1' --name='test' --net_name='reg_seg'
