#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --output=myjobresults-%J.out
#SBATCH --job-name=cifar10vit16-envsetup
#SBATCH --gres=gpu:2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user ta187904@ucf.edu


# Load modules
echo "Slurm nodes assigned :$SLURM_JOB_NODELIST"
module purge
module load cuda
module load gcc/gcc-9.1.0
module load oneapi/mkl
source ~/.bashrc

conda update conda

pip install requirements.txt