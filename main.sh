#!/bin/bash
#time python static_scm_discovery.py \
#  --splits 4,3,2 \
#  --num_epochs 100 \
#  --num_runs 1 \
##  --logtostderr
#
#time python dynamic_scm_discovery.py \
#  --splits 3,3,3 \
#  --num_epochs 600 \
#  --num_runs 1 \
#  #--logtostderr
#
#time python coda_forward_model.py \
#  --num_sprites 4 \
#  --imagedim 20 \
#  --seed 0 \
#  --num_epochs 100 \
#  --num_pairs 500 \
#  --max_episode_length 5000 \
#  --weight_decay 0. \
#  --results_dir /scratch/gobi1/creager/disentangled_transitions/coda_forward_model

## small
time python model_based_rollouts.py \
  --model_type neural \
  --num_sprites 3 \
  --imagedim 20 \
  --seed 0 \
  --num_examples 100 \
  --patience_epochs 100 \
  --num_frames 100 \
  --max_episode_length 500 \
  --results_dir /tmp/neural_model_based_rollouts

#time python model_based_rollouts.py \
#  --model_type neural \
#  --num_sprites 6 \
#  --imagedim 20 \
#  --seed 0 \
#  --num_examples 100 \
#  --max_episode_length 500 \
#  --results_dir /scratch/gobi1/creager/disentangled_transitions/model_based_rollouts
