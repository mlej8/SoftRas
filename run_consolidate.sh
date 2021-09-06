#!/bin/bash
#SBATCH --gres=gpu:v100l:1        # Number of GPU(s) per node
#SBATCH --cpus-per-task=8         # CPU cores/threads
#SBATCH --mem=128G                # memory
#SBATCH --time=3-0                # A time limit of zero requests that no time limit be imposed. Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".
#SBATCH --job-name=cl         
#SBATCH --output=logs/%j
#SBATCH --mail-user=er.li@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --nodelist=gra1337

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# confirm gpu available
nvidia-smi

# activate env
source env/bin/activate

# validation python version
python --version

python examples/recon/continual_learning.py -eid cl --consolidate > continual_learning_consolidate.log 2>&1
