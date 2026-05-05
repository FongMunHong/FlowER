#!/bin/sh
#SBATCH -J flower_fixes
#SBATCH -n 1
#SBATCH -t 4-00:00:00
#SBATCH -G h100:2
#SBATCH -c 16
#SBATCH -p pi_ccoley
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ptim@mit.edu
#SBATCH -o /home/ptim/orcd/scratch/logs/FlowER/flower_fixes_test_%j.out
#SBATCH --error=/home/ptim/orcd/scratch/logs/FlowER/flower_test_%j.err
#SBATCH --mem=256G

set -euo pipefail 

REPO_DIR="/home/ptim/FlowER/FlowERrs"
cd "$REPO_DIR"


module load miniforge
conda activate flower


sh run_FlowER_large_newData.sh
