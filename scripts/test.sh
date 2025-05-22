#!/bin/bash
# no cfg
torchrun --nnodes=1 --nproc_per_node=2 --master_port=29500 generate.py \
  --model SiT-XL/2 \
  --ckpt /0500000.pt \
  --sample-dir=/samples \
  --encoder-depth=8 \
  --num-fid-samples 50000 \
  --path-type=linear \
  --projector-embed-dims=768 \
  --per-proc-batch-size=64 \
  --mode=sde \
  --num-steps=250 \
  --cfg-scale=1.0 \
  --guidance-high=1.0\
  --guidance-low=0.0
# cfg
torchrun --nnodes=1 --nproc_per_node=2 --master_port=29500 generate.py \
  --model SiT-XL/2 \
  --ckpt /2500000.pt \
  --sample-dir=/samples \
  --encoder-depth=8 \
  --num-fid-samples 50000 \
  --path-type=linear \
  --projector-embed-dims=768 \
  --per-proc-batch-size=64 \
  --mode=sde \
  --num-steps=250 \
  --cfg-scale=1.65 \
  --guidance-high=0.72 \
  --guidance-low=0.0