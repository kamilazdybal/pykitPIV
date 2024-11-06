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

srun python pykitPIV-generate-images.py --dt 0.5 --n_images 5 --size_buffer 80 --image_height 1024 --image_width 1024 --densities 0.1 0.1 --gaussian_filters 50 50
srun python pykitPIV-generate-images.py --dt 1 --n_images 5 --size_buffer 80 --image_height 1024 --image_width 1024 --densities 0.1 0.1 --gaussian_filters 50 50
srun python pykitPIV-generate-images.py --dt 2 --n_images 5 --size_buffer 80 --image_height 1024 --image_width 1024 --densities 0.1 0.1 --gaussian_filters 50 50
srun python pykitPIV-generate-images.py --dt 3 --n_images 5 --size_buffer 80 --image_height 1024 --image_width 1024 --densities 0.1 0.1 --gaussian_filters 50 50
srun python pykitPIV-generate-images.py --dt 4 --n_images 5 --size_buffer 80 --image_height 1024 --image_width 1024 --densities 0.1 0.1 --gaussian_filters 50 50
srun python pykitPIV-generate-images.py --dt 5 --n_images 5 --size_buffer 80 --image_height 1024 --image_width 1024 --densities 0.1 0.1 --gaussian_filters 50 50
srun python pykitPIV-generate-images.py --dt 6 --n_images 5 --size_buffer 80 --image_height 1024 --image_width 1024 --densities 0.1 0.1 --gaussian_filters 50 50
srun python pykitPIV-generate-images.py --dt 7 --n_images 5 --size_buffer 80 --image_height 1024 --image_width 1024 --densities 0.1 0.1 --gaussian_filters 50 50
srun python pykitPIV-generate-images.py --dt 8 --n_images 5 --size_buffer 80 --image_height 1024 --image_width 1024 --densities 0.1 0.1 --gaussian_filters 50 50
srun python pykitPIV-generate-images.py --dt 9 --n_images 5 --size_buffer 80 --image_height 1024 --image_width 1024 --densities 0.1 0.1 --gaussian_filters 50 50
srun python pykitPIV-generate-images.py --dt 10 --n_images 5 --size_buffer 80 --image_height 1024 --image_width 1024 --densities 0.1 0.1 --gaussian_filters 50 50
