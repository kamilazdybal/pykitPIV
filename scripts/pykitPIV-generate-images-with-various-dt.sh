#!/bin/bash -l
#
#SBATCH --account=em13
#SBATCH --partition=normal
#SBATCH --time=8:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kamilazdybal@gmail.com
#SBATCH --job-name=pykitPIV-various-dt
#SBATCH --output=pykitPIV-various-dt.txt
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

srun python pykitPIV-generate-images-with-various-dt.py --dt 0.5 1 1.5 2 2.5 3 3.5 4 5 6 7 8 9 10 --n_images 357 --size_buffer 80 --image_height 256 --image_width 256