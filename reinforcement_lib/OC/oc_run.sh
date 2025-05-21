#!/bin/bash -l
#SBATCH -p general
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --output=./output_logs/%x-%j.out
#srun singularity run --nv ../ppo_tests/webots_v2.sif bash sleep_test.sh $@
srun singularity run --nv ../ppo_tests/webots_v2.sif bash oc_script.sh $@
