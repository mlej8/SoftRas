#! /bin/bash

CUDA_VISIBLE_DEVICES=0

python examples/recon/continual_learning.py -eid cl \ 
    --model-directory data/results/models \ 
    --dataset-directory /mnt/e/Data/mesh_reconstruction \ 
    --consolidate > continual_learning.log 2>&1 &
