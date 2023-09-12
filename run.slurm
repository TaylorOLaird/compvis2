#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00:00
#SBATCH --error=myjobresults-%J.err
#SBATCH --output=myjobresults-%J.out
#SBATCH --job-name=cifar10vit16
#SBATCH --gres=gpu:2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user ta187904@ucf.edu
#SBATCH --constraint=gpu32


# Load modules
echo "Slurm nodes assigned :$SLURM_JOB_NODELIST"
module purge
module load cuda
module load gcc/gcc-9.1.0
module load oneapi/mkl
source ~/.bashrc

conda activate vitenv
export LD_LIBRARY_PATH=/home/cap6411.student28/anaconda3/envs/env/lib

python cifar10vit16.py
