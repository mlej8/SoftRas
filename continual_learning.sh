#! /bin/bash

CUDA_VISIBLE_DEVICES=0

python examples/recon/continual_learning.py -eid cl \
    -md data/results/models \
    -dd /mnt/e/Data/mesh_reconstruction > continual_learning.log 2>&1 &
