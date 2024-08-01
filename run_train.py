#!/bin/bash
MODEL_PATH='pre-trained_segmodel'
DATASET_NAME='SHA'
SAVE_NAME=${DATASET_NAME}'_train'

python train.py --dataset_name=${DATASET_NAME} \
                 --gpu_ids='0,1,2,3' \
                 --optimizer='adam' \
                 --test_eval=0 \
                 --start_eval_epoch=400 \
                 --lr=1e-4 \
                 --lr_policy='step' \
                 --lr_decay_iters=300 \
                 --pretrain_model=${MODEL_PATH} \
                 --name=${SAVE_NAME} \
                 --net_name='reg_seg' \
                 --batch_size=16 \
                 --nThreads=32 \
                 --max_epochs=700 \
                 --eval_per_epoch=1
