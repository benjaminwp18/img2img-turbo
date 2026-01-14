#!/bin/bash
#SBATCH -p gpu
#SBATCH -C gpul40s
#SBATCH -gres=gpu:1
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --time=0-36:00:00
#SBATCH --mem=32gb
#SBATCH --output=/home/bwp18/img2img-turbo/logs/%x_%4j.out
#SBATCH --mail-user=bwp18@case.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="uhaze_img2img"

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.8.0
. /home/bwp18/img2img-turbo/venv/bin/activate
. /home/bwp18/img2img-turbo/train.sh
