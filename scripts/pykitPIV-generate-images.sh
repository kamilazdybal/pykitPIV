#!/bin/bash -l
#
#SBATCH --account=em13
#SBATCH --partition=normal
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kamilazdybal@gmail.com
#SBATCH --job-name=pykitPIV-generate-images
#SBATCH --output=pykitPIV-generate-images.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --constraint=gpu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

conda activate pykitPIV

export NCCL_DEBUG=INFO
export HDF5_USE_FILE_LOCKING=FALSE

srun python pykitPIV-generate-images.py --densities 0.01 0.01
srun python pykitPIV-generate-images.py --densities 0.05 0.05
srun python pykitPIV-generate-images.py --densities 0.1 0.1
srun python pykitPIV-generate-images.py --densities 0.2 0.2
srun python pykitPIV-generate-images.py --densities 0.3 0.3
srun python pykitPIV-generate-images.py --densities 0.4 0.4

