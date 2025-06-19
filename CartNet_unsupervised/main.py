# Copyright Universitat Politècnica de Catalunya 2024 https://imatge.upc.edu
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import torch
import torch.nn.functional as F
import logging
import argparse
import pickle
import os
from tqdm import tqdm
from logger.logger import create_logger
from loader.loader import create_loader
from models.master import create_model
#### IMPORTANTT ########
from train.train_regularization import train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric import seed_everything
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.graphgym.logger import set_printing


def inference(model, loader):
    """
    Run inference using the trained model and data loader, compute metrics, and save the results.
    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset to perform inference on.
    This function sets the model to evaluation mode and disables gradient calculations.
    It iterates over the data loader, collects predictions and ground truths, and computes metrics such as IoU,
    Mean Absolute Error (MAE), and similarity index for each batch. The metrics are logged, and all inference outputs
    are saved to a pickle file specified by `cfg.inference_output`.tra
    """
    from train.metrics import compute_3D_IoU, get_similarity_index
    model.eval()
    
    with torch.no_grad():
        inference_output = {"pred": [], "true": [], "temp": [], "cell": [], "refcode": [], "pos": [], "atoms": [], "iou": [], "mae": [], "similarity_index": []}
        for iter, batch in tqdm(enumerate(loader), total=len(loader), ncols=50):
            batch.to("cuda:0")
            inference_output["cell"].append(batch.cell.detach().to("cpu"))
            inference_output["atoms"].append(batch.x[batch.non_H_mask].detach().to("cpu"))
            inference_output["pos"].append(batch.pos[batch.non_H_mask].detach().to("cpu"))
            inference_output["refcode"].append(batch.refcode[0])
            inference_output["temp"].append(batch.temperature_og.detach().to("cpu")[0])
            _pred, _true = model(batch)
            inference_output["pred"].append(_pred.detach().to("cpu"))
            inference_output["true"].append(_true.detach().to("cpu"))
            inference_output["iou"].append(compute_3D_IoU(_pred, _true).detach().to("cpu"))
            inference_output["mae"].append(F.l1_loss(_pred,_true, reduce="none").detach().to("cpu"))
            inference_output["similarity_index"].append(get_similarity_index(_pred, _true).detach().to("cpu"))
        
        
        iou = torch.cat(inference_output["iou"], dim=0)
        mae = torch.cat(inference_output["mae"], dim=0)
        similarity_index = torch.cat(inference_output["similarity_index"], dim=0)
        
        logging.info(f"Mean IoU: {iou.mean().item()} +/- {iou.std().item()}")
        logging.info(f"Mean MAE: {mae.mean().item()} +/- {mae.std().item()}")
        logging.info(f"Mean Similarity Index: {similarity_index.mean().item()} +/- {similarity_index.std().item()}")

        pickle.dump(inference_output, open(cfg.inference_output, "wb"))

def montecarlo(model, loader):
    """
    Performs Monte Carlo simulations to evaluate the model's performance under random rotations.
    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        loader (torch.utils.data.DataLoader): DataLoader providing the dataset for evaluation.
    The function runs multiple iterations (e.g., 100) where it:
    - Applies a random rotation to the input batch data.
    - Performs a forward pass to obtain predictions.
    - Computes evaluation metrics such as Intersection over Union (IoU), Mean Absolute Error (MAE), and similarity index.
    - Stores and logs the results for each iteration.
    After all iterations, it aggregates the metrics to compute the mean and standard deviation, providing insights into the model's robustness to rotations.
    Results are saved to output files specified in the configuration, and important metrics are logged for analysis.
    """
    from train.metrics import compute_3D_IoU, get_similarity_index
    import roma

    model.eval()
    iou_montecarlo = []
    similarity_index_montecarlo = []
    mae_montecarlo = []
    with torch.no_grad():
        for i in tqdm(range(100), ncols=50, desc="Montecarlo"):
            inference_output = {"pred": [], "true": [], "cell": [], "refcode": [], "pos": [], "atoms": [], "mae": [], "iou": [], "similarity_index": []}
            for iter, batch in tqdm(enumerate(loader), total=len(loader), ncols=50):
                batch_copy = batch.clone()
                batch.to("cuda:0")
                inference_output["cell"].append(batch.cell.detach().to("cpu"))
                inference_output["atoms"].append(batch.x[batch.non_H_mask].detach().to("cpu"))
                inference_output["pos"].append(batch.pos[batch.non_H_mask].detach().to("cpu"))
                inference_output["refcode"].append(batch.refcode[0])
                pseudo_true, _ = model(batch)
                R = roma.utils.random_rotmat(size=1, device=pseudo_true.device).squeeze(0)
                batch_copy.to("cuda:0")
                batch_copy.cart_dir = batch_copy.cart_dir @ R
                pseudo_true =  R.transpose(-1,-2) @ pseudo_true @ R
                pred, _ = model(batch_copy)
                inference_output["pred"].append(pred.detach().to("cpu"))
                inference_output["true"].append(pseudo_true.detach().to("cpu"))
                inference_output["iou"].append(compute_3D_IoU(pred, pseudo_true).detach().to("cpu"))
                inference_output["similarity_index"].append(get_similarity_index(pred, pseudo_true).detach().to("cpu"))
                inference_output["mae"].append(F.l1_loss(pred, pseudo_true, reduce="none").detach().to("cpu"))
            pickle.dump(inference_output, open(cfg.inference_output.replace(".pkl", "_montecarlo_"+str(i)+".pkl"), "wb"))
            logging.info(f"Montecarlo {i}")
            logging.info(f"IoU: {torch.cat(inference_output['iou'], dim=0).mean().item()}")
            logging.info(f"MAE: {torch.cat(inference_output['mae'], dim=0).mean().item()}")
            logging.info(f"Similarity Index: {torch.cat(inference_output['similarity_index'], dim=0).mean().item()}")
            iou_montecarlo+=inference_output["iou"]
            mae_montecarlo+=inference_output["mae"]
            similarity_index_montecarlo+=inference_output["similarity_index"]
    
    iou_montecarlo = torch.cat(iou_montecarlo, dim=0)
    mae_montecarlo = torch.cat(mae_montecarlo, dim=0)
    similarity_index_montecarlo = torch.cat(similarity_index_montecarlo, dim=0)

    logging.info(f"Montecarlo IoU: {iou_montecarlo.mean().item()} +/- {iou_montecarlo.std().item()}")
    logging.info(f"Montecarlo MAE: {mae_montecarlo.mean().item()} +/- {mae_montecarlo.std().item()}")
    logging.info(f"Montecarlo Similarity Index: {similarity_index_montecarlo.mean().item()} +/- {similarity_index_montecarlo.std().item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Seed for the experiment')
    parser.add_argument('--name', type=str, default="MP_carnet complet", help="name of the Wandb experiment" )
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
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
    parser.add_argument("--dim_in", type=int, default=32, help="Input dimension")
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
    parser.add_argument("--unsupervised_cartnet", type=bool, default=False, help="adding node and edge reconstruction + graph embeddings")
    parser.add_argument("--pretrained", type=bool, default=True, help="adding node and edge reconstruction + graph embeddings")
    parser.add_argument("--normalize", type=str, default="", help="normalization for node reconstruction")
    parser.add_argument("--stored_training", type=str, default="", help="path of the trained model and optimizer")
    parser.add_argument("--dataset_unlabeled", type=str, default="MP", help="")
    parser.add_argument("--dataset_labeled", type=str, default="megnet", help="")
    parser.add_argument("--dataset_unlabeled_path", type=str, default="", help="")
    parser.add_argument("--dataset_labeled_path", type=str, default="", help="")
    parser.add_argument("--regularization_cartnet", type=bool, default=True, help="training of supervised and unsupervised tasks simultaneously")
    parser.add_argument("--unsup_weight", type=float, default=0.25, help="Weight of node reconstruction loss")
    parser.add_argument("--supervised_weight", type=float, default=0.25, help="Weight of adjacency reconstruction loss")

    
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
    cfg.normalize = args.normalize
    cfg.stored_training = args.stored_training
    cfg.regularization_cartnet = args.regularization_cartnet
    cfg.dataset_unlabeled = args.dataset_unlabeled
    cfg.dataset_labeled = args.dataset_labeled
    cfg.dataset_unlabeled_path = args.dataset_unlabeled_path
    cfg.dataset_labeled_path = args.dataset_labeled_path
    cfg.unsup_weight = args.unsup_weight
    cfg.supervised_weight = args.supervised_weight


    torch.set_num_threads(args.threads)

    set_printing()

    #Seed
    seed_everything(cfg.seed)

    logging.info(f"Experiment will be saved at: {cfg.run_dir}")

    if cfg.regularization_cartnet:
        cfg.dataset.name = cfg.dataset_unlabeled
        cfg.dataset_path = cfg.dataset_unlabeled_path
        unlbl_loaders = create_loader()

        cfg.dataset.name = cfg.dataset_labeled
        cfg.dataset_path = cfg.dataset_labeled_path
        lbl_loaders = create_loader()

    else:
        loaders = create_loader()

    
    model = create_model()

    logging.info(model)
    cfg.params_count = params_count(model)
    logging.info(f"Number of parameters: {cfg.params_count}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    loggers = create_logger()
    
    if os.path.exists(cfg.stored_training):
        ckpt = torch.load(cfg.stored_training, map_location='cuda:0')  # or 'cpu' if needed
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        logging.info("Checkpoint loaded successfully.")
    else:
        logging.warning("Checkpoint not found. Starting from scratch.")

    if args.inference:
        assert args.checkpoint_path is not None, "Weights path not provided"
        assert cfg.dataset.name == "ADP", "Inference only for ADP dataset"
        ckpt = torch.load(args.checkpoint_path)
        model.load_state_dict(ckpt["model_state"])
        cfg.inference_output = args.inference_output
        inference(model, loaders[-1])
    elif args.montecarlo:
        assert args.checkpoint_path is not None, "Weights path not provided"
        assert cfg.dataset.name == "ADP", "Montecarlo only for ADP dataset"
        ckpt = torch.load(args.checkpoint_path)
        model.load_state_dict(ckpt["model_state"])
        cfg.inference_output = args.inference_output
        montecarlo(model, loaders[-1])
    else:
        train(model, unlbl_loaders, lbl_loaders , optimizer, loggers)










