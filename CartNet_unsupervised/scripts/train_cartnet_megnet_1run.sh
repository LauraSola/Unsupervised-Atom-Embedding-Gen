#!/bin/bash
#SBATCH -p gpi.compute            # Partition to submit to
#SBATCH --mem=25G 
#SBATCH -c6     # Max CPU Memory
#SBATCH --gres=gpu:1,gpumem:16G
#SBATCH --time=24:00:00



cd ..
echo "--figshare_target can be any of this for Jarvis dataset: [e_form, gap pbe, mbj_bandgap, bulk modulus, shear modulus]"

python main.py --seed 0 --figshare_target "gap pbe" --name "megnet_shear_modulus_carnet batch 128" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
                                    --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 128 --batch_accumulation 1 --lr 0.03 --epochs 500 --unsupervised_cartnet True


python test_metrics.py --path "./results/megnet_shear_modulus_carnet"