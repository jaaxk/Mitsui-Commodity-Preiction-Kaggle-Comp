#!/bin/bash
#SBATCH -p hbm-long-96core
#SBATCH --time 48:00:00
#SBATCH --output res.txt

python tune_hyperparams.py --tune_method bayesian
