#! /bin/bash
#SBATCH --gres=gpu:v100:1        # Number of GPU(s) per node
#SBATCH --time=0:15:00                # A time limit of zero requests that no time limit be imposed. Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".
#SBATCH --nodelist=gra1337
#SBATCH --mem=16G   

salloc --time=0:15:00  --nodelist=gra1337 --gres=gpu:v100:1 --mem=16G   