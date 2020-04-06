#!/bin/bash -x
#SBATCH --gres=gpu:0
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -o ./slurm_output/%J.out # this is where the output goes

python model_based_rollouts.py --model_type lstm --num_sprites 6 --imagedim 20 --seed 0 --num_examples 10000 --patience_epochs 100 --max_episode_length 500 --results_dir /scratch/gobi1/creager/disentangled_transitions/model_based_rollouts/lstm2 --num_lstm_layers 2