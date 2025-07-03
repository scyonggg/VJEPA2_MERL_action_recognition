#!/bin/bash

python vjepa2_gradio_demo.py \
  --config configs/eval/vitl/merl.yaml \
  --encoder_ckpt ckpts/vjepa2_vitl.pt \
  --classifier_ckpt ./ckpts/video_classification_frozen/merl-vitl16-256-16f-5classes-temporal/best_val.pt \
  --device cuda
