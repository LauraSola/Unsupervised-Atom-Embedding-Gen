#!/bin/bash
#SBATCH -p gpi.compute            # Partition to submit to
#SBATCH --mem=20G 
#SBATCH -c6     # Max CPU Memory
#SBATCH --gres=gpu:1,gpumem:11G
#SBATCH --time=24:00:00

cd ..
echo "--figshare_target can be any of this for Jarvis dataset: [e_form, gap pbe, mbj_bandgap, bulk modulus, shear modulus]"


# "../CartNet/dense_vector/dense_vector_dim 32 pesos finals guardar embd.pkl"
#"../CartNet/dense_vector/dense_vector_mean_dim_32_pesos_finals_SI.pkl" 
# "../CartNet/dense_vector/dense_vector_mean_dim_32_pesos_finals_pes_edge_2.pkl"

##################
#### BASELINE ####
##################

# python main.py --seed 1 --figshare_target "gap pbe" --name "megnet b 128 dim 32 baseline fc head" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
#                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 32 --batch_accumulation 1 --lr 0.001 --epochs 400 \
#                                     --input_filter True --dense_vector "../CartNet/dense_vector/dense_vector_mean_dim_32_pesos_finals_SI.pkl"

# python main.py --seed 1 --figshare_target "gap pbe" --name "megnet b 128 dim 64 baseline" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
#                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 64 --batch_accumulation 1 --lr 0.001 --epochs 500 

# python main.py --seed 2 --figshare_target "gap pbe" --name "megnet b 128 dim 64 baseline" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
#                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 64 --batch_accumulation 1 --lr 0.001 --epochs 500 

# python main.py --seed 3 --figshare_target "gap pbe" --name "megnet b 128 dim 64 baseline" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
#                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 64 --batch_accumulation 1 --lr 0.001 --epochs 500 


#########################################################################
### input = embeddings     edge encoder only / linear atom encoder  #####
#########################################################################

python main.py --seed 0 --figshare_target "gap pbe" --name "megnet b 128 dim 32 embd linear combined aug" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
                                    --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 32 --batch_accumulation 1 --lr 0.001 --epochs 400 \
                                    --input "embeddings" --input_filter True --encoder "Linear" --dense_vector "../CartNet/dense_vector/dense_vector_mean_triplet.pkl" 
#                                    --path_model "../CartNet/results/dim 32 pesos finals/0/ckpt/best.ckpt" --copiar_pesos True

python main.py --seed 1 --figshare_target "gap pbe" --name "megnet b 128 dim 32 embd linear combined aug" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
                                    --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 32 --batch_accumulation 1 --lr 0.001 --epochs 400 \
                                    --input "embeddings" --input_filter True --encoder "Linear" --dense_vector "../CartNet/dense_vector/dense_vector_mean_triplet.pkl" 
#                                    --path_model "../CartNet/results/dim 32 pesos finals/0/ckpt/best.ckpt" --copiar_pesos True

# python main.py --seed 2 --figshare_target "gap pbe" --name "megnet b 128 dim 32 embd no enc branca dalt aug" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
#                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 32 --batch_accumulation 1 --lr 0.001 --epochs 400 \
#                                     --input "embeddings" --input_filter True --encoder "No encoder" --dense_vector "../CartNet/dense_vector/dense_vector_mean_dim_32_branca_dalt_aug.pkl" 
# #                                    --path_model "../CartNet/results/dim 32 pesos finals/0/ckpt/best.ckpt" --copiar_pesos True

# python main.py --seed 3 --figshare_target "gap pbe" --name "megnet b 128 dim 32 embd no enc branca dalt aug" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
#                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 32 --batch_accumulation 1 --lr 0.001 --epochs 400 \
#                                     --input "embeddings" --input_filter True --encoder "No encoder" --dense_vector "../CartNet/dense_vector/dense_vector_mean_dim_32_branca_dalt_aug.pkl" 
# #                                    --path_model "../CartNet/results/dim 32 pesos finals/0/ckpt/best.ckpt" --copiar_pesos True

# python main.py --seed 0 --figshare_target "gap pbe" --name "megnet b 128 dim 64 embd no enc filtrat" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
#                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 64 --batch_accumulation 1 --lr 0.001 --epochs 500 \
#                                     --input "embeddings filtrats" --encoder "No encoder" --dense_vector "../CartNet/dense_vector/dense_vector_b128 dim 64 e 110.pkl" 
# #                                    --path_model "../CartNet/results/b128 dim 64 e 110/0/ckpt/best.ckpt" --copiar_pesos

# python main.py --seed 1 --figshare_target "gap pbe" --name "megnet b 128 dim 64 embd no enc filtrat" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
#                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 64 --batch_accumulation 1 --lr 0.001 --epochs 500 \
#                                     --input "embeddings filtrats" --encoder "No encoder" --dense_vector "../CartNet/dense_vector/dense_vector_b128 dim 64 e 110.pkl" 
# #                                    --path_model "../CartNet/results/b128 dim 64 e 110/0/ckpt/best.ckpt" --copiar_pesos True


###########################################################
### input = num atomic     copiar embd atom encoder   #####
###########################################################


python main.py --seed 1 --figshare_target "gap pbe" --name "megnet b 128 dim 32 embd al encoder embds pes" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
                                    --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 32 --batch_accumulation 1 --lr 0.001 --epochs 400 \
                                    --dense_vector "../CartNet/dense_vector/dense_vector_mean_dim_32_combined_aug.pkl"  --input_filter True --experiment "Copy dense vector" 
#                                    --path_model "../CartNet/results/dim 32 pesos finals/0/ckpt/best.ckpt" --copiar_pesos True

python main.py --seed 2 --figshare_target "gap pbe" --name "megnet b 128 dim 32 embd al encoder embds pes" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
                                    --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 32 --batch_accumulation 1 --lr 0.001 --epochs 400 \
                                    --dense_vector "../CartNet/dense_vector/dense_vector_mean_dim_32_combined_aug.pkl"  --input_filter True --experiment "Copy dense vector" 
#                                    --path_model "../CartNet/results/dim 32 pesos finals/0/ckpt/best.ckpt" --copiar_pesos True


python main.py --seed 0 --figshare_target "gap pbe" --name "megnet b 128 dim 32 embd al encoder embds pes 2" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
                                    --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 32 --batch_accumulation 1 --lr 0.001 --epochs 400 \
                                    --dense_vector "../CartNet/dense_vector/dense_vector_mean_dim_32_branca_dalt_aug.pkl"  --input_filter True --experiment "Copy dense vector" 
#                                    --path_model "../CartNet/results/dim 32 pesos finals/0/ckpt/best.ckpt" --copiar_pesos True

python main.py --seed 3 --figshare_target "gap pbe" --name "megnet b 128 dim 32 embd al encoder embds pes 2" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
                                    --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 32 --batch_accumulation 1 --lr 0.001 --epochs 400 \
                                    --dense_vector "../CartNet/dense_vector/dense_vector_mean_dim_32_branca_dalt_aug.pkl"  --input_filter True --experiment "Copy dense vector" 
#                                    --path_model "../CartNet/results/dim 32 pesos finals/0/ckpt/best.ckpt" --copiar_pesos True

# python main.py --seed 0 --figshare_target "gap pbe" --name "megnet b 128 dim 32 COPY f" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
#                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 32 --batch_accumulation 1 --lr 0.001 --epochs 400 \
#                                     --dense_vector "../CartNet/dense_vector/dense_vector_dim 32 pesos finals guardar embd.pkl" --input_filter True \
#                                     --path_model "../CartNet/results/dim 32 pesos finals/0/ckpt/best.ckpt" --copiar_pesos True

# python main.py --seed 1 --figshare_target "gap pbe" --name "megnet b 128 dim 64 embd al encoder" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
#                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 64 --batch_accumulation 1 --lr 0.001 --epochs 500 \
#                                     --dense_vector "../CartNet/dense_vector/dense_vector_b128 dim 64 e 110.pkl"  --experiment "Copy dense vector" 
# #                                   --path_model "../CartNet/results/b128 dim 64 e 110/0/ckpt/best.ckpt" --copiar_pesos True

# python main.py --seed 2 --figshare_target "gap pbe" --name "megnet b 128 dim 64 embd al encoder" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
#                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 64 --batch_accumulation 1 --lr 0.001 --epochs 500 \
#                                     --dense_vector "../CartNet/dense_vector/dense_vector_b128 dim 64 e 110.pkl"  --experiment "Copy dense vector" 
# #                                   --path_model "../CartNet/results/b128 dim 64 e 110/0/ckpt/best.ckpt" --copiar_pesos True

# python main.py --seed 3 --figshare_target "gap pbe" --name "megnet b 128 dim 64 embd al encoder" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
#                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 64 --batch_accumulation 1 --lr 0.001 --epochs 500 \
#                                     --dense_vector "../CartNet/dense_vector/dense_vector_b128 dim 64 e 110.pkl"  --experiment "Copy dense vector" 
# #                                   --path_model "../CartNet/results/b128 dim 64 e 110/0/ckpt/best.ckpt" --copiar_pesos True

                                    

# ##############################################################################################################################

##################
#### BASELINE ####
##################

# python main.py --seed 0 --figshare_target "gap pbe" --name "megnet b 128 dim 32 baseline f" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
#                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 32 --batch_accumulation 1 --lr 0.001 --epochs 400 \
#                                     --input_filter True --dense_vector "../CartNet/dense_vector/dense_vector_mean_dim_32_pesos_finals_SI.pkl" 

# # python main.py --seed 1 --figshare_target "gap pbe" --name "megnet b 128 dim 64 baseline" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
# #                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 64 --batch_accumulation 1 --lr 0.001 --epochs 500 

# # python main.py --seed 2 --figshare_target "gap pbe" --name "megnet b 128 dim 64 baseline" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
# #                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 64 --batch_accumulation 1 --lr 0.001 --epochs 500 

# # python main.py --seed 3 --figshare_target "gap pbe" --name "megnet b 128 dim 64 baseline" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
# #                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 64 --batch_accumulation 1 --lr 0.001 --epochs 500 


# #########################################################################
# ### input = embeddings     edge encoder only / linear atom encoder  #####
# #########################################################################

# python main.py --seed 0 --figshare_target "gap pbe" --name "megnet b 128 dim 32 embd linear embds f" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
#                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 32 --batch_accumulation 1 --lr 0.001 --epochs 400 \
#                                     --input "embeddings" --input_filter True --encoder "Linear" --dense_vector "../CartNet/dense_vector/dense_vector_mean_dim_32_pesos_finals_SI.pkl" 
# #                                    --path_model "../CartNet/results/b128 dim 64 e 110/0/ckpt/best.ckpt" --copiar_pesos

# python main.py --seed 1 --figshare_target "gap pbe" --name "megnet b 128 dim 32 embd linear embds f" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
#                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 32 --batch_accumulation 1 --lr 0.001 --epochs 400 \
#                                     --input "embeddings" --input_filter True --encoder "Linear" --dense_vector "../CartNet/dense_vector/dense_vector_mean_dim_32_pesos_finals_SI.pkl" 
# #                                    --path_model "../CartNet/results/b128 dim 64 e 110/0/ckpt/best.ckpt" --copiar_pesos

# python main.py --seed 0 --figshare_target "gap pbe" --name "megnet b 128 dim 32 embd no enc embds f" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
#                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 32 --batch_accumulation 1 --lr 0.001 --epochs 400 \
#                                     --input "embeddings" --input_filter True --encoder "No encoder" --dense_vector "../CartNet/dense_vector/dense_vector_mean_dim_32_pesos_finals_SI.pkl" 
# #                                    --path_model "../CartNet/results/b128 dim 64 e 110/0/ckpt/best.ckpt" --copiar_pesos

# python main.py --seed 1 --figshare_target "gap pbe" --name "megnet b 128 dim 32 embd no enc embds f" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
#                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 32 --batch_accumulation 1 --lr 0.001 --epochs 400 \
#                                     --input "embeddings" --input_filter True --encoder "No encoder" --dense_vector "../CartNet/dense_vector/dense_vector_mean_dim_32_pesos_finals_SI.pkl" 
# #                                    --path_model "../CartNet/results/b128 dim 64 e 110/0/ckpt/best.ckpt" --copiar_pesos

# # python main.py --seed 0 --figshare_target "gap pbe" --name "megnet b 128 dim 64 embd no enc filtrat" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
# #                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 64 --batch_accumulation 1 --lr 0.001 --epochs 500 \
# #                                     --input "embeddings filtrats" --encoder "No encoder" --dense_vector "../CartNet/dense_vector/dense_vector_b128 dim 64 e 110.pkl" 
# # #                                    --path_model "../CartNet/results/b128 dim 64 e 110/0/ckpt/best.ckpt" --copiar_pesos

# # python main.py --seed 1 --figshare_target "gap pbe" --name "megnet b 128 dim 64 embd no enc filtrat" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
# #                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 64 --batch_accumulation 1 --lr 0.001 --epochs 500 \
# #                                     --input "embeddings filtrats" --encoder "No encoder" --dense_vector "../CartNet/dense_vector/dense_vector_b128 dim 64 e 110.pkl" 
# # #                                    --path_model "../CartNet/results/b128 dim 64 e 110/0/ckpt/best.ckpt" --copiar_pesos True


# ###########################################################
# ### input = num atomic     copiar embd atom encoder   #####
# ###########################################################


# python main.py --seed 0 --figshare_target "gap pbe" --name "megnet b 128 dim 32 embd al encoder embds f" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
#                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 32 --batch_accumulation 1 --lr 0.001 --epochs 400 \
#                                     --dense_vector "../CartNet/dense_vector/dense_vector_mean_dim_32_pesos_finals_SI.pkl"  --input_filter True --experiment "Copy dense vector" 
# #                                   --path_model "../CartNet/results/b128 dim 64 e 110/0/ckpt/best.ckpt" --copiar_pesos True


# python main.py --seed 1 --figshare_target "gap pbe" --name "megnet b 128 dim 32 embd al encoder embds f" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
#                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 32 --batch_accumulation 1 --lr 0.001 --epochs 400 \
#                                     --dense_vector "../CartNet/dense_vector/dense_vector_mean_dim_32_pesos_finals_SI.pkl"  --input_filter True --experiment "Copy dense vector" 
# #                                   --path_model "../CartNet/results/b128 dim 64 e 110/0/ckpt/best.ckpt" --copiar_pesos True

# # python main.py --seed 1 --figshare_target "gap pbe" --name "megnet b 128 dim 64 embd al encoder" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
# #                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 64 --batch_accumulation 1 --lr 0.001 --epochs 500 \
# #                                     --dense_vector "../CartNet/dense_vector/dense_vector_b128 dim 64 e 110.pkl"  --experiment "Copy dense vector" 
# # #                                   --path_model "../CartNet/results/b128 dim 64 e 110/0/ckpt/best.ckpt" --copiar_pesos True

# # python main.py --seed 2 --figshare_target "gap pbe" --name "megnet b 128 dim 64 embd al encoder" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
# #                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 64 --batch_accumulation 1 --lr 0.001 --epochs 500 \
# #                                     --dense_vector "../CartNet/dense_vector/dense_vector_b128 dim 64 e 110.pkl"  --experiment "Copy dense vector" 
# # #                                   --path_model "../CartNet/results/b128 dim 64 e 110/0/ckpt/best.ckpt" --copiar_pesos True

# # python main.py --seed 3 --figshare_target "gap pbe" --name "megnet b 128 dim 64 embd al encoder" --model "CartNet" --dataset "megnet" --dataset_path "./dataset/megnet/"  \
# #                                     --wandb_project "CartNet Paper Megnet" --batch 128 --dim_in 64 --batch_accumulation 1 --lr 0.001 --epochs 500 \
# #                                     --dense_vector "../CartNet/dense_vector/dense_vector_b128 dim 64 e 110.pkl"  --experiment "Copy dense vector" 
# # #                                   --path_model "../CartNet/results/b128 dim 64 e 110/0/ckpt/best.ckpt" --copiar_pesos True

                                    





python test_metrics.py --path "./results/megnet b 128 dim 64 baseline"