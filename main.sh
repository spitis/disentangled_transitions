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

time python linear_dynamics_model.py \
  --num_sprites 6 \
  --imagedim 20 \
  --seed 0 \
  --num_examples 10000 \
  --max_episode_length 500 \
  --results_dir /scratch/gobi1/creager/disentangled_transitions/linear_dynamics_model

