#!/bin/bash -l
#
#SBATCH --account=em13
#SBATCH --partition=debug
#SBATCH --time=00:30:00
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

srun python pykitPIV-generate-images.py --dt 0.001 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 0.01 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 0.1 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 0.5 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 1.0 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 1.5 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 2.0 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 3.0 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 4.0 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 5.0 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 6.0 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 7.0 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 8.0 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 9.0 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 10.0 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256