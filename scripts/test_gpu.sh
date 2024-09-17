#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gpus=a100:2
#SBATCH --job-name=test_dyna
#SBATCH -t 5:00:00
#SBATCH --mem=10000MB
#SBATCH -o ./out_gpu
#SBATCH -e ./err_gpu

# Activate Conda environment based on the user's configuration
source $(conda info --base)/etc/profile.d/conda.sh
conda activate temp_dyna

# Use python from the activated environment
python VEP_nt_siamese_nt.py \
  --index 3 \
  --template 'False' \
  --population 100 \
  --generations 100 \
  --njobs 5

