#!/bin/bash
#time python static_scm_discovery.py \
#  --splits 4,3,2 \
#  --num_epochs 100 \
#  --num_runs 1 \
##  --logtostderr
#
#time python dynamic_scm_discovery.py \
#  --splits 3,3,3 \
#  --num_epochs 50 \
#  --batch_size 64 \
#  --num_runs 5 \

#################################################################################
## train SSA attn mech
#MODEL_TYPE=SSA
#RESULTS_DIR=/scratch/gobi1/creager/disentangled_transitions/attn2_$MODEL_TYPE
#time python spriteworld_scm_discovery.py \
#  --num_epochs 125 \
#  --num_runs 5 \
#  --num_sprites 4 \
#  --batch_size 1000 \
#  --num_examples 50000 \
#  --mask_reg 0. \
#  --weight_reg 0. \
#  --attn_reg 0. \
#  --weight_decay 0. \
#  --model_type $MODEL_TYPE \
#  --results_dir $RESULTS_DIR
#################################################################################

################################################################################
# train MMN attn mech
MODEL_TYPE=MMN
RESULTS_DIR=/scratch/gobi1/creager/disentangled_transitions/attn2_$MODEL_TYPE
time python spriteworld_scm_discovery.py \
  --num_epochs 50 \
  --num_runs 5 \
  --num_sprites 4 \
  --batch_size 256 \
  --num_examples 50000 \
  --mask_reg 2e-3 \
  --weight_reg 1e-5 \
  --attn_reg 3e-4 \
  --weight_decay 0. \
  --mask_reg 2e-3 \
  --model_type $MODEL_TYPE \
  --results_dir $RESULTS_DIR
################################################################################


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
#model_type=lstm
#RESULTS_DIR=/tmp/model_based_rollouts/$model_type
#time python model_based_rollouts.py \
#  --model_type $model_type \
#  --num_sprites 3 \
#  --imagedim 20 \
#  --seed 0 \
#  --num_examples 5000 \
#  --num_epochs 100 \
#  --patience_epochs 25 \
#  --batch_size 64 \
#  --num_frames 100 \
#  --max_episode_length 500 \
#  --results_dir $RESULTS_DIR

## vws
#for model_type in linear neural lstm
#do
#  GOBI_DIR=/scratch/gobi1/creager/disentangled_transitions
#  RESULTS_DIR=$GOBI_DIR/model_based_rollouts/$model_type
#  time python model_based_rollouts.py \
#  --model_type $model_type \
#  --num_sprites 6 \
#  --imagedim 20 \
#  --seed 0 \
#  --num_examples 10000 \
#  --patience_epochs 100 \
#  --max_episode_length 500 \
#  --results_dir $RESULTS_DIR
#done

#time python train_RL_agent.py \
#  --attn_mech_dir /tmp/spriteworld_scm_discovery \
#  --relabel_every 1000 \
#  --batch_size 32 \
#  --num_pairs 100 \
#  --coda_samples_per_pair 2 \
#  --max_timesteps 3000 \
#  --relabel_type ground_truth \
#  --results_dir /tmp/train_RL_agent \
#  --save_model

#python eval_utils.py

#for thresh in 0. 0.1 0.2 0.3 0.4 0.5 0.6
#do
#  time python rollouts_with_attn.py \
#    --thresh $thresh
#done
