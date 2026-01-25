#!/bin/bash
#SBATCH -p gpu
#SBATCH -C gpul40s
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --time=0-36:00:00
#SBATCH --mem=32gb
#SBATCH --output=/home/bwp18/img2img-turbo/logs/%x_%4j.out
#SBATCH --mail-user=bwp18@case.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="phaze_img2img"

. /home/bwp18/img2img-turbo/prep_env.sh
. /home/bwp18/img2img-turbo/train_paired.sh
