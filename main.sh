#!/bin/bash
time python static_scm_discovery.py \
  --splits 3,3,3 \
  --num_epochs 30 \
  --logtostderr
