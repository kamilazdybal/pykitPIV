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

srun python pykitPIV-generate-images.py --dt 0.2 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 0.3 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 0.4 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 0.6 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 0.7 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 0.8 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 0.9 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 1.1 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 1.2 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 1.3 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 1.4 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 1.6 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 1.7 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 1.8 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 1.9 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 2.5 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256
srun python pykitPIV-generate-images.py --dt 3.5 --n_images 10 --size_buffer 60 --image_height 256 --image_width 256