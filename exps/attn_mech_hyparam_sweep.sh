#!/bin/bash

for lr in 0.001 0.0001; do
  for wd in 0.00001 0.00005; do
    for mr in 0.001 0.0001; do
      for ar in 0.001 0.0001; do
        for wr in 0.001 0.0001; do
          bash exps/launch_attn_mech.sh gpu spriteworld_scm_discovery_lr${lr}_wd${wd}_mr${mr}_ar${ar}_wr${wr} 1 "python spriteworld_scm_discovery.py --lr ${lr} --weight_decay ${wd} --mask_reg ${mr} --attn_reg ${ar} --weight_reg ${wr} --num_epochs 200 --num_runs 5 --num_sprites 4 --batch_size 512 --num_examples 50000"
        done
      done
    done
  done
done


