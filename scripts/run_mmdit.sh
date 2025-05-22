#!/bin/bash
accelerate launch train_t2i.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --enc-type="dinov2-vit-b" \
  --proj-coeff=0.5 \
  --attn-coeff=0.5 \
  --encoder-depth=8 \
  --output-dir="MMDiT" \
  --exp-name="t2i_haste" \
  --data-dir=[YOUR_DATA_PATH] \
  --early-stop-point=150000 \
  --checkpointing-steps=25000 \
  --max-train-steps=150000
