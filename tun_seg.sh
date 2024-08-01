#!/bin/bash
python train_seg.py --dataset_name='NWPU' \
                 --gpu_ids='0,1,2,3' \
                 --optimizer='adam' \
                 --lr=5e-5 \
                 --lr_decay_iters=500 \
                 --name='test' \
                 --net_name='reg_seg' \
                 --batch_size=16 \
                 --nThreads=8 \
                 --max_epoch=500
