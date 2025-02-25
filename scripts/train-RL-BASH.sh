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
#SBATCH --job-name=inspecting-Q-values-gamma-03
#SBATCH --output=inspecting-Q-values-gamma-03.txt

# Set environment:
source $HOME/.bashrc
module use $HOME/MyModules
module load anaconda3/2019.03
conda activate pykitPIV

# Run Python script:
python train-RL.py --case_name "inspecting-Q-values-gamma-03" --discount_factor 0.3 --normalize_displacement_vectors --interrogation_window_size_buffer 0
