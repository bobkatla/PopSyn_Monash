#!/bin/bash
#SBATCH --job-name=run_saa
#SBATCH --partition=comp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --time=160:00:00
#SBATCH --output=run_saa.out

module load miniforge3
conda activate psim
python --version
psim synthesize -runs runs.yml -o out/
