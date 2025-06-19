#!/bin/bash
#SBATCH -p gpi.compute            # Partition to submit to
#SBATCH --mem=30G 
#SBATCH -c6     # Max CPU Memory
#SBATCH --gres=gpu:1,gpumem:48G
#SBATCH --time=24:00:00



cd ..
echo "--figshare_target can be any of this for Jarvis dataset: [e_form, gap pbe, mbj_bandgap, bulk modulus, shear modulus]"


python main.py --seed 1 --figshare_target "gap pbe" --name "Dim 64 join training" --model "CartNet"    \
                                    --wandb_project "CartNet Paper MP" --batch 128 --dim_in 64 --batch_accumulation 1 --lr 0.03 --epochs 50 --regularization_cartnet True  \
                                    --alpha 2 --beta 0.08 --gamma 1 --unsup_weight 1 --supervised_weight 2  \
                                    --dataset_unlabeled "MP" --dataset_unlabeled_path "./dataset/mp_139K/" --dataset_labeled "megnet" --dataset_labeled_path "./dataset/megnet/"



# python main.py --seed 0 --figshare_target "gap pbe" --name "Dim 128 contrastive pes edge aug part 2" --model "CartNet" --dataset "MP" --dataset_path "./dataset/mp_139K/"  \
#                                     --wandb_project "CartNet Paper MP" --batch 128 --dim_in 128 --batch_accumulation 1 --lr 0.03 --epochs 60 --unsupervised_cartnet True --loss "combined" \
#                                     --alpha 225 --beta 4 --gamma 3 --stored_training "./results/Dim 128 contrastive pes edge aug part 1/0/ckpt/best.ckpt"


#

# python main.py --seed 0 --figshare_target "gap pbe" --name "dim 32 nomes barlow" --model "CartNet" --dataset "MP" --dataset_path "./dataset/mp_139K/"  \
#                                     --wandb_project "CartNet Paper MP" --batch 128 --dim_in 32 --batch_accumulation 1 --lr 0.03 --epochs 110 --unsupervised_cartnet True --loss "combined" \
#                                     --alpha 250 --beta 4 --gamma 3

# python main.py --seed 0 --figshare_target "gap pbe" --name "dim 32 proves branca dalt aug beta2" --model "CartNet" --dataset "MP" --dataset_path "./dataset/mp_139K/"  \
#                                     --wandb_project "CartNet Paper MP" --batch 128 --dim_in 32 --batch_accumulation 1 --lr 0.03 --epochs 110 --unsupervised_cartnet True --loss "branca dalt" \
#                                     --alpha 250 --beta 2 --gamma 3

# python main.py --seed 0 --figshare_target "gap pbe" --name "dim 32 proves combined aug" --model "CartNet" --dataset "MP" --dataset_path "./dataset/mp_139K/"  \
#                                     --wandb_project "CartNet Paper MP" --batch 128 --dim_in 32 --batch_accumulation 1 --lr 0.03 --epochs 110 --unsupervised_cartnet True --loss "combined" \
#                                     --alpha 250 --beta 2 --gamma 3

# python main.py --seed 0 --figshare_target "gap pbe" --name "guardar embeddings b128 dim 32 epochs 200_" --model "CartNet" --dataset "MP" --dataset_path "./dataset/mp_139K/"  \
#                                     --wandb_project "CartNet Paper MP" --batch 128 --dim_in 32 --batch_accumulation 1 --lr 0.03 --epochs 200 --unsupervised_cartnet True --loss "combined"


#python test_metrics.py --path "./results/guardar embeddings b128 dim 32"