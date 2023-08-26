#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gpus=a100:2
#SBATCH --job-name=test_zhanh
#SBATCH -t 5:00:00
#SBATCH --mem=10000MB
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=huixin.zhan@cshs.org
#SBATCH -o /common/zhanh/out_gpu
#SBATCH -e /common/zhanh/err_gpu
source /common/zhanh/anaconda3/etc/profile.d/conda.sh
conda activate dna_lora
/common/zhanh/anaconda3/envs/dna_lora/bin/python
python \
VEP_nt_siamese_nt.py \
--index 3 \
--template 'False' \
--population 100 \
--generations 100 \
--njobs 5
