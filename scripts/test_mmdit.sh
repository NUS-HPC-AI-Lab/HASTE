#!/bin/bash
torchrun --nnodes=1 --nproc_per_node=2 --master_port=29500 generate_t2i.py \
  --ckpt /[YOUR_CKPT_PATH] \
  --sample-dir=/samples \
  --encoder-depth=8 \
  --path-type=linear \
  --projector-embed-dims=768 \
  --per-proc-batch-size=32 \
  --mode=sde \
  --num-steps=250 \
  --cfg-scale=2.25 \
  --guidance-high=1.0 