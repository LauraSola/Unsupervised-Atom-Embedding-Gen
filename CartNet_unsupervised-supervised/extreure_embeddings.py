import torch
import argparse
import os
import numpy as np
import pickle
from tqdm import tqdm
from collections import defaultdict
from models.master import create_model
from loader.loader import create_loader
from torch_geometric import seed_everything
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.graphgym.logger import set_printing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Seed for the experiment')
    parser.add_argument('--name', type=str, default="MP_carnet complet", help="name of the Wandb experiment" )
    parser.add_argument("--batch", type=int, default=128, help="Batch size")
    parser.add_argument("--batch_accumulation", type=int, default=1, help="Batch Accumulation")
    parser.add_argument("--dataset", type=str, default="MP", help="Dataset name. Available: ADP, jarvis, megnet, MP")
    parser.add_argument("--dataset_path", type=str, default="./dataset/mp_139K/")
    parser.add_argument("--inference", action="store_true", help="Inference")
    parser.add_argument("--montecarlo", action="store_true", help="Montecarlo")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path of the checkpoints of the model")
    parser.add_argument("--inference_output", type=str, default="./inference.pkl", help="Path to the inference output")
    parser.add_argument("--figshare_target", type=str, default="formation_energy_peratom", help="Figshare dataset target")
    parser.add_argument("--wandb_project", type=str, default="CartNet Paper MP", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="laura-sola-garcia-universitat-polit-cnica-de-catalunya", help="Name of the wandb entity")
    parser.add_argument("--loss", type=str, default="combined", help="Loss function")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate") #1e-3
    parser.add_argument("--warmup", type=float, default=0.01, help="Warmup")
    parser.add_argument('--model', type=str, default="CartNet", help="Model Name")
    parser.add_argument("--max_neighbours", type=int, default=25, help="Max neighbours (only for iComformer/eComformer)")
    parser.add_argument("--radius", type=float, default=5.0, help="Radius for the Radius Graph Neighbourhood")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--dim_in", type=int, default=128, help="Input dimension")
    parser.add_argument("--dim_rbf", type=int, default=64, help="Number of RBF")
    parser.add_argument('--augment', action='store_true', help='augment')
    parser.add_argument("--invariant", action="store_true", help="Rotation Invariant model")
    parser.add_argument("--disable_temp", action="store_false", help="Disable Temperature")
    parser.add_argument("--no_standarize_temp", action="store_false", help="Standarize temperature")
    parser.add_argument("--disable_envelope", action="store_false", help="Disable envelope")
    parser.add_argument('--disable_H', action='store_false', help='Hydrogens')
    parser.add_argument('--disable_atom_types', action='store_false', help='Atom types')
    parser.add_argument("--threads", type=int, default=1, help="Number of threads")
    parser.add_argument("--workers", type=int, default=3, help="Number of workers") 
    parser.add_argument("--alpha", type=float, default=0.25, help="Weight of node reconstruction loss")
    parser.add_argument("--beta", type=float, default=0.25, help="Weight of adjacency reconstruction loss")
    parser.add_argument("--gamma", type=float, default=0.5, help="Weight of barlow twins loss")
    parser.add_argument("--lamda", type=float, default=0.0051, help="Weight of off-diagonal loss (redundancy loss) in BT loss")
    parser.add_argument("--unsupervised_cartnet", type=bool, default=True, help="adding node and edge reconstruction + graph embeddings")
    parser.add_argument("--pretrained", type=bool, default=True, help="adding node and edge reconstruction + graph embeddings")
    parser.add_argument("--normalize", type=str, default="", help="normalization for node reconstruction")
    parser.add_argument("--path_model", type=str, default="", help="path to the pre-trained model")



    
    set_cfg(cfg)

    args = parser.parse_args()
    cfg.seed = args.seed
    cfg.name = args.name
    cfg.run_dir = "results/"+cfg.name+"/"+str(cfg.seed)
    cfg.inference_output = args.inference_output
    cfg.dataset.task_type = "regression"
    cfg.batch = args.batch
    cfg.batch_accumulation = args.batch_accumulation
    cfg.dataset.name = args.dataset
    cfg.dataset_path = args.dataset_path
    cfg.figshare_target = args.figshare_target
    cfg.wandb_project = args.wandb_project
    cfg.wandb_entity = args.wandb_entity
    cfg.loss = args.loss
    cfg.optim.max_epoch = args.epochs
    cfg.lr = args.lr
    cfg.warmup = args.warmup
    cfg.model = args.model
    cfg.max_neighbours = -1 if cfg.model== "CartNet" else args.max_neighbours
    cfg.radius = args.radius
    cfg.num_layers = args.num_layers
    cfg.dim_in = args.dim_in
    cfg.dim_rbf = args.dim_rbf
    cfg.augment = False if cfg.model in ["icomformer", "ecomformer"] else args.augment
    cfg.invariant = args.invariant
    cfg.use_temp = False if cfg.dataset.name != "ADP" else args.disable_temp
    cfg.standarize_temp = args.no_standarize_temp
    cfg.envelope = args.disable_envelope
    cfg.use_H = args.disable_H
    cfg.use_atom_types = args.disable_atom_types
    cfg.workers = args.workers
    cfg.alpha = args.alpha
    cfg.beta = args.beta
    cfg.gamma = args.gamma
    cfg.lamda = args.lamda
    cfg.unsupervised_cartnet = args.unsupervised_cartnet
    cfg.path_model = args.path_model
    cfg.normalize = args.normalize


    torch.set_num_threads(args.threads)

    set_printing()

    #Seed
    seed_everything(cfg.seed)

    ##### CODI ###########
    model = create_model()
    loaders = create_loader()

    #carregar els pesos
    #file_path = "./results/dim 32 pesos finals/0/ckpt/best.ckpt"
    #file_path = "./results/dim 32 proves pes edge 2/0/ckpt/best.ckpt"
    #file_path = "./results/dim 32 proves pes edge/0/ckpt/best.ckpt"
    #file_path = "./results/dim 32 proves branca dalt aug/0/ckpt/best.ckpt"
    #file_path = "./results/dim 32 proves combined aug/0/ckpt/best.ckpt"
    #file_path = "./results/dim 32 contrastive pes edge/0/ckpt/best.ckpt"
    file_path = "./results/Dim 128 contrastive pes edge aug part 2/0/ckpt/best.ckpt"
    #file_path = "./results/noves proves dim 32 aug canvi pesos/0/ckpt/best.ckpt"
    ckpt = torch.load(file_path, weights_only=True)
    model.load_state_dict(ckpt["model_state"], strict=False)

    print('holi')
    
    model.eval()

    embedding_dict = defaultdict(list)
    mean_embedding_dict = {}

    loader = loaders[0] #train set

    with torch.no_grad():
        for iter, batch in tqdm(enumerate(loader), total=len(loader), ncols=50):
            batch.to("cuda:0")
            (pred_node_feat, pred_adj), (true_node_feat, true_adj), batch_atom_fea = model(batch.clone(), reconstr = True)

            atomic_numbers = torch.argmax(batch_atom_fea.y, dim=1)

            # Store embeddings grouped by atomic number
            for atom, emb in zip(atomic_numbers.cpu().numpy(), batch_atom_fea.x.detach().cpu().numpy()):
                embedding_dict[atom].append(emb)


    # Define a set of expected atomic numbers (example: range of common elements)
    expected_atomic_numbers = set(range(1, 119))  # 1 to 118, covering all known elements

    for atom in expected_atomic_numbers:
        if atom in embedding_dict:
            embds = np.array(embedding_dict[atom])
            mean_emb = np.mean(embds, axis=0)
            mean_embedding_dict[atom] = mean_emb / np.linalg.norm(mean_emb)  # normalize
        else:
            mean_embedding_dict[atom] = np.zeros((128,))  # Assuming 32-dim embeddings

    for atom in mean_embedding_dict:
        mean_embedding_dict[atom] = torch.tensor(mean_embedding_dict[atom]).unsqueeze(0)



    os.makedirs(os.path.dirname(f"./dense_vector/dense_vector_individual_dim_128_aug_contrastive.pkl"), exist_ok=True)
    os.makedirs(os.path.dirname(f"./dense_vector/dense_vector_mean_dim_128_aug_contrastive.pkl"), exist_ok=True)

    with open(f"./dense_vector/dense_vector_individual_dim_128_aug_contrastive.pkl", 'wb') as file_:
        pickle.dump(embedding_dict, file_)

    with open(f"./dense_vector/dense_vector_mean_dim_128_aug_contrastive.pkl", 'wb') as file_:
        pickle.dump(mean_embedding_dict, file_)



