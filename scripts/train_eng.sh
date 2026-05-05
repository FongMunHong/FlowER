#!/bin/sh
#SBATCH -J flower_fixes
#SBATCH -n 1
#SBATCH -t 2-48:00:00
#SBATCH -G 4
#SBATCH -c 16
#SBATCH -p pi_ccoley
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ptim@mit.edu
#SBATCH -o /home/ptim/orcd/scratch/logs/FlowER/flower_fixes_%j.out
#SBATCH --error=/home/ptim/orcd/scratch/logs/FlowER/flower_fixes_%j.err
#SBATCH --mem=512G

set -euo pipefail

REPO_DIR="/home/ptim/FlowER/FlowERrs"
cd "$REPO_DIR"


module load miniforge
conda activate flower


sh run_FlowER_large_newData.sh