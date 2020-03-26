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

time python coda_forward_model.py \
  --num_sprites 4 \
  --imagedim 20 \
  --seed 0 \
  --num_epochs 5 \
  --num_pairs 100 \
  --max_episode_length 1000 \

