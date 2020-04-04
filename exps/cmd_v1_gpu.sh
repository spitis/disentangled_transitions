#!/bin/bash -x
#SBATCH --ntasks=1 # Note that ntasks=1 runs multiple jobs in an array
#SBATCH --array=1-48%48
#SBATCH --gres=gpu:0
#SBATCH -p p100
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o ./slurm_output/%J.out # this is where the output goes

# run with sbatch path_to_cmd.sh path_to_cmd.txt path_to_results_folder

cmd_line=$(sed "${SLURM_ARRAY_TASK_ID}q;d" ${1})
PYTHONPATH=./ $cmd_line
