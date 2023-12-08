#!/bin/bash
#SBATCH -t 3-00:00
#SBATCH --job-name=midigen
#SBATCH --time=4:00:00 # walltime
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4 # number of cores
#SBATCH --mem-per-cpu=100g # memory per CPU core
#SBATCH --constraint=40g

# load the modules
module load python/ondemand-jupyter-python3.8
module load cuda/11.8
source activate gpu_midi_transformer_env

python /ix/djishnu/Aaron/2_misc/PGM_Project/scripts/generate_improv_midis.py
