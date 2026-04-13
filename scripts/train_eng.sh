#!/bin/sh
#SBATCH -J flower_fixes
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -G 4
#SBATCH -p mit_preemptable
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH -o /home/ptim/orcd/scratch/logs/FlowER/flower_fixes_%j.out
#SBATCH --mem=128G

module load miniforge
conda activate flower

sh run_FlowER_large_newData.sh