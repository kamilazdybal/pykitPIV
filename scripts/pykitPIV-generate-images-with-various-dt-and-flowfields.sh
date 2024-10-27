#!/bin/bash -l
#
#SBATCH --account=em13
#SBATCH --partition=normal
#SBATCH --time=4:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kamilazdybal@gmail.com
#SBATCH --job-name=pykitPIV-generate-images-various-dt-and-flowfields
#SBATCH --output=pykitPIV-generate-images-various-dt-and-flowfields.txt
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

srun python pykitPIV-generate-images-with-various-dt-and-flowfields.py --particle_loss 40 40 --gaussian_filters 100 100 --dt 1 --n_images 1 --size_buffer 10 --image_height 2048 --image_width 2048 --densities 0.1 0.11
srun python pykitPIV-generate-images-with-various-dt-and-flowfields.py --particle_loss 60 60 --gaussian_filters 100 100 --dt 1 --n_images 1 --size_buffer 10 --image_height 2048 --image_width 2048 --densities 0.1 0.11
srun python pykitPIV-generate-images-with-various-dt-and-flowfields.py --particle_loss 70 70 --gaussian_filters 100 100 --dt 1 --n_images 1 --size_buffer 10 --image_height 2048 --image_width 2048 --densities 0.1 0.11
srun python pykitPIV-generate-images-with-various-dt-and-flowfields.py --particle_loss 80 80 --gaussian_filters 100 100 --dt 1 --n_images 1 --size_buffer 10 --image_height 2048 --image_width 2048 --densities 0.1 0.11
srun python pykitPIV-generate-images-with-various-dt-and-flowfields.py --particle_loss 90 90 --gaussian_filters 100 100 --dt 1 --n_images 1 --size_buffer 10 --image_height 2048 --image_width 2048 --densities 0.1 0.11
srun python pykitPIV-generate-images-with-various-dt-and-flowfields.py --particle_loss 95 95 --gaussian_filters 100 100 --dt 1 --n_images 1 --size_buffer 10 --image_height 2048 --image_width 2048 --densities 0.1 0.11
srun python pykitPIV-generate-images-with-various-dt-and-flowfields.py --particle_loss 99 99 --gaussian_filters 100 100 --dt 1 --n_images 1 --size_buffer 10 --image_height 2048 --image_width 2048 --densities 0.1 0.11



