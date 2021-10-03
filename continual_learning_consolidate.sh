#! /bin/bash

CUDA_VISIBLE_DEVICES=0

python examples/recon/continual_learning.py -eid cl \
    -md "/mnt/e/Continual Learning/results/models" \
    -dd /mnt/e/Data/mesh_reconstruction \
    --consolidate > continual_learning_consolidate.log 2>&1 &
