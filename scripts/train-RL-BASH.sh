#!/bin/sh
#SBATCH --account=sutherland-np
#SBATCH --partition=sutherland-shared-np
#SBATCH --export=NONE
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=kamilazdybal@gmail.com
#SBATCH --job-name=64-64-relu
#SBATCH --output=64-64-relu.txt

# Set environment:
source $HOME/.bashrc
module use $HOME/MyModules
module load anaconda3/2019.03
conda activate pykitPIV

# Run Python script:
python train-RL.py --case_name "64-64-relu" --wt_size 200 200 --n_episodes 2000 --n_iterations 10 --epsilon_start 0.8 --epsilon_end 0 --discount_factor 0.95 --batch_size 32 --n_epochs 1 --memory_size 32 --initial_learning_rate 0.1 --alpha_lr 0.01 --magnify_step 10 --sample_every_n 10 --no-normalize_displacement_vectors --interrogation_window_size_buffer 0 --interrogation_window_size 40 40
