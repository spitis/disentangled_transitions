#!/bin/bash
#time python static_scm_discovery.py \
#  --splits 4,3,2 \
#  --num_epochs 100 \
#  --num_runs 5 \
#  --logtostderr

time python dynamic_scm_discovery.py \
  --splits 3,3,3 \
  --num_epochs 250 \
  --num_runs 5 \
  --logtostderr
