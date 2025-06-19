#!/bin/bash
#SBATCH -p gpi.compute            # Partition to submit to
#SBATCH --mem=16G 
#SBATCH -c6     # Max CPU Memory
#SBATCH --gres=gpu:1,gpumem:16G
#SBATCH --time=24:00:00



cd ..
echo "--figshare_target can be any of this for Jarvis dataset: [e_form, gap pbe, mbj_bandgap, bulk modulus, shear modulus]"

python test_metrics.py --path "./results/megnet_shear_modulus_carnet"