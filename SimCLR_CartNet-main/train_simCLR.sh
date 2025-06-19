#!/bin/bash
#SBATCH -p gpi.compute            # Partition to submit to
#SBATCH --mem=20G 
#SBATCH -c6     # Max CPU Memory
#SBATCH --gres=gpu:1,gpumem:11G
#SBATCH --time=1:00:00


python run.py 
