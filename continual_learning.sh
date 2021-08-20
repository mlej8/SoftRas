#! /bin/sh

CUDA_VISIBLE_DEVICES=0

nohup python examples/recon/continual_learning.py -eid cl > continual_learning.log 2>&1 &
