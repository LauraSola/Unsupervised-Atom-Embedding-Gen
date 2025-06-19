# Copyright Universitat Politècnica de Catalunya 2024 https://imatge.upc.edu
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import pickle
import torch
import logging
import wandb
import time
import os
import random
import numpy as np
import os.path as osp
from tqdm import tqdm
import torch.nn.functional as F
from torch_scatter import scatter
from train.BarlowTwins import BarlowTwinsLoss
from torch_geometric.graphgym.config import cfg
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.data import Batch
from torch.amp import GradScaler, autocast
from itertools import zip_longest
from train.metrics import compute_metrics_and_logging, compute_loss, compute_loss_nodes, compute_loss_adj, compute_loss_contrastive


def flatten_dict(metrics):
    """Flatten a list of train/val/test metrics into one dict to send to wandb.

    Args:
        metrics: List of Dicts with metrics

    Returns:
        A flat dictionary with names prefixed with "train/" , "val/"
    """
    prefixes = ['train', 'val']
    result = {}
    for i in range(len(metrics)):
        # Take the latest metrics.
        stats = metrics[i][-1]
        result.update({f"{prefixes[i]}/{k}": v for k, v in stats.items()})
    return result

def fix_validation_dataset(loader):
   
    fixed_data = []  # List to store fixed augmented samples

    for batch in tqdm(loader, desc="Fixing validation/test set", ncols=50):
        batch = batch.to("cpu")

        # Generate augmentations
        augmented_batch_0 = augment_batch(batch.clone()).to("cpu")
        augmented_batch_1 = augment_batch(batch.clone()).to("cpu")
        augmented_batch_2 = augment_batch(batch.clone()).to("cpu")

        # Store modified batch
        fixed_data.append([augmented_batch_0, augmented_batch_1,augmented_batch_2, batch ])

    return fixed_data


def info_nce_loss(features, batch_size):

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    
    labels = labels.to("cuda:0")

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to("cuda:0")
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to("cuda:0")

    temperature = 0.07
    logits = logits / temperature
    return logits, labels


def train(model, unlbl_loaders, lbl_loaders, optimizer, loggers):
    """
    Train the model

    Args:
        model: PyTorch model
        loaders: List of PyTorch data loaders
        optimizer: PyTorch optimizer
        loggers: List of loggers

    Returns: None

    """

    
    run = wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project,
                    name=cfg.name, config=cfg)

    num_splits = len(loggers)
    full_epoch_times = []
    perf = [[] for _ in range(num_splits-1)]
    ckpt_dir = osp.join(cfg.run_dir,"ckpt/")

    #scheduler = OneCycleLR(optimizer, max_lr=cfg.lr, total_steps=cfg.optim.max_epoch * len(loaders[0]) // cfg.batch_accumulation + cfg.optim.max_epoch , pct_start=cfg.warmup)
    scheduler = None

    unlbl_val_fixed_data = fix_validation_dataset(unlbl_loaders[1]) #validation
    lbl_val_fixed_data = fix_validation_dataset(lbl_loaders[1])
   
    print('batches unlbl loader', len(unlbl_loaders[0]))
    print('batches lbl loader', len(lbl_loaders[0]))


    for cur_epoch in range(cfg.optim.max_epoch):
        start_time = time.perf_counter()
        
        train_epoch(loggers[0], lbl_loaders[0],unlbl_loaders[0],  model, optimizer, cfg.batch_accumulation, scheduler)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        eval_epoch(loggers[1], lbl_val_fixed_data ,unlbl_val_fixed_data, model)
        perf[1].append(loggers[1].write_epoch(cur_epoch))
       
        full_epoch_times.append(time.perf_counter() - start_time)    
        run.log(flatten_dict(perf), step=cur_epoch)

        
        # Log current best stats on eval epoch.     
        best_epoch = int(np.array([vp['loss'] for vp in perf[1]]).argmin()) #abans estava a MAE
        
        best_train = f"train_loss: {perf[0][best_epoch]['loss']:.4f}" #MAE
        
        best_val = f"val_loss: {perf[1][best_epoch]['loss']:.4f}" #MAE

        bstats = {"best/epoch": best_epoch}
        for i, s in enumerate(['train', 'val']): 
            bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
            #bstats[f"best/{s}_MAE"] = perf[i][best_epoch]['MAE']
        logging.info(bstats)
        run.log(bstats, step=cur_epoch)
        run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
        run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)

        # Checkpoint the best epoch params.
        if best_epoch == cur_epoch: #######################


            ckpt = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = osp.join(ckpt_dir, 'best.ckpt')
            
            torch.save(ckpt, ckpt_path)
        
            logging.info(f"Best checkpoint saved at {ckpt_path}")

        
        logging.info(
            f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
            f"(avg {np.mean(full_epoch_times):.1f}s) | "
            f"Best so far: epoch {best_epoch}\t"
            f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
            f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
        )
    
    
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model_state"])
    
    ### TEST ONLY ON LABELLED DATA
    test_epoch(loggers[-1], lbl_loaders[-1], model, test_metrics=True)
    
    perf_test = loggers[-1].write_epoch(best_epoch)
    best_test = f"test_loss: {perf_test['loss']:.4f}" #MAE
    run.log({f"test/{k}": v for k, v in perf_test.items()})
    bstats[f"best/test_loss"] = perf_test['loss']
    bstats[f"best/test_MAE"] = perf_test['MAE']
    logging.info(bstats)
    run.log(bstats)

    logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                #f"test_loss: {perf_test['loss']:.4f} {best_test}"
            )
    
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    

    for logger in loggers:
        logger.close()
   
    logging.info('Task done, results saved in %s', ckpt_dir)

    run.finish()

#Elimino 10% dels edges randomly
def remove_edges(batch, mask_factor = 0.1):
    num_edges = batch.edge_index.shape[1]
    num_edges_remove = int(mask_factor * num_edges)
    edges_to_remove = random.sample(range(num_edges), num_edges_remove)

    mask = torch.ones(num_edges, dtype=torch.bool)
    mask[edges_to_remove] = False
    
    batch.edge_index = batch.edge_index[:, mask]
    batch.cart_dist = batch.cart_dist[mask]
    batch.cart_dir = batch.cart_dir[mask]
    
#Poso a 0 els nombres atomics de 10% dels nodes
def mask_nodes(batch, mask_factor = 0.1):
    num_nodes = batch.x.shape[0]
    num_nodes_mask = int(mask_factor * num_nodes)
    nodes_to_mask = random.sample(range(num_nodes), num_nodes_mask)

    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[nodes_to_mask] = True

    batch.x[mask] = 0

def augment_batch(batch, mask_factor_edges = 0.1, mask_factor_nodes = 0.1):

    augmented_batch = batch.clone()

    remove_edges(augmented_batch, mask_factor_edges)
    mask_nodes(augmented_batch, mask_factor_nodes)

    return augmented_batch
    
def embedding_create(batch):

    atomic_numbers = torch.argmax(batch.y, dim=1)

    mean_embeddings = scatter(batch.x, atomic_numbers, dim=0, reduce='mean')

    embedding_dict = {} 

    for atom_idx in atomic_numbers.unique().tolist():  # Iterate only over unique indices
        mean_embedding = mean_embeddings[atom_idx]
        embedding_dict[int(atom_idx)] = F.normalize(mean_embedding.view(1, -1), dim=1, p=2)

    return embedding_dict   


def node_edge_reconstr_forward(batch, model):

    (pred_node_feat, pred_adj), (true_node_feat, true_adj), batch_atom_fea = model(batch.clone(), reconstr = True)
    node_loss = compute_loss_nodes(pred_node_feat, true_node_feat)
    adj_loss= compute_loss_adj(pred_adj, true_adj)

    return node_loss, adj_loss, pred_node_feat, true_node_feat


def contrastive_forward(batch_1, batch_2, model):

    batch_aug = Batch.from_data_list([batch_1, batch_2]).to("cuda:0")

    features, _, _ = model(batch_aug.clone(), graph_emb = True)

    batch_size = batch_1.batch.max().item() + 1
    logits, labels = info_nce_loss(features, batch_size)
    contr_loss = compute_loss_contrastive(logits, labels)

    return contr_loss


def property_pred_forward(batch, model):

    pred,true, _ = model(batch)
    MAE,MSE = compute_loss(pred, true)

    return MAE, MSE




def train_epoch(logger, labeled_loader, unlabeled_loader, model, optimizer, batch_accumulation, scheduler):
    """
    Train the model for one epoch.
    Args:
        logger (Logger): Logger object to log training information.
        loader (DataLoader): DataLoader object providing the training data.
        model (nn.Module): The model to be trained.
        optimizer (Optimizer): Optimizer for updating the model parameters.
        batch_accumulation (int): Number of batches to accumulate gradients before updating the model parameters.
        scheduler (Scheduler): Learning rate scheduler.
        barlos_twins_loss: Loss to compare two graph embeddings (correlation matrix wrt. identity)
    Raises:
        Exception: If the specified loss function is not implemented.
    Returns:
        None
    """
    model.train()
    optimizer.zero_grad()
    
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    total_steps = max(len(labeled_loader), len(unlabeled_loader))
    print(len(labeled_loader))
    print(len(unlabeled_loader))

    iters = 0
    for step, (labeled_batch, unlabeled_batch) in enumerate(
        tqdm(zip_longest(labeled_iter, unlabeled_iter, fillvalue=None), total=total_steps, ncols=50)
    ):
        time_start = time.time()

        # Recreate iterators and reshuffle if one runs out
        if labeled_batch is None: #since it is smaller, it will get reused
            labeled_iter = iter(labeled_loader)
            labeled_batch = next(labeled_iter)
       
        
        labeled_batch = labeled_batch.to("cuda:0")
        unlabeled_batch = unlabeled_batch.to("cuda:0")

        # --- Unlabelled forward ---
        
        unlbl_batch_aug_0 = augment_batch(unlabeled_batch.clone())
        u_node_loss, u_adj_loss, _, _ = node_edge_reconstr_forward(unlbl_batch_aug_0.clone(), model)

        unlbl_batch_aug_1 = augment_batch(unlabeled_batch.clone())
        unlbl_batch_aug_2 = augment_batch(unlabeled_batch.clone())
        u_contr_loss = contrastive_forward(unlbl_batch_aug_1.clone(), unlbl_batch_aug_2.clone(), model)

        unlbl_loss = cfg.alpha * u_node_loss + cfg.beta * u_adj_loss + cfg.gamma * u_contr_loss

        # --- Labelled forward ---
        
        lbl_batch_aug_0 = augment_batch(labeled_batch.clone())
        node_loss, adj_loss, pred_node_feat, true_node_feat = node_edge_reconstr_forward(lbl_batch_aug_0.clone(), model)

        lbl_batch_aug_1 = augment_batch(labeled_batch.clone())
        lbl_batch_aug_2 = augment_batch(labeled_batch.clone())
        contr_loss = contrastive_forward(lbl_batch_aug_1.clone(), lbl_batch_aug_2.clone(), model)

        MAE, MSE = property_pred_forward(labeled_batch.clone(), model)

        unsupervised_lbld_loss = cfg.alpha * node_loss + cfg.beta * adj_loss + cfg.gamma * contr_loss
        supervised_lbld_loss = MAE ## ??

        # --- Loss combination ---
        loss = (unlbl_loss + unsupervised_lbld_loss)*cfg.unsup_weight + supervised_lbld_loss*cfg.supervised_weight 

        loss.mean().backward()



        if ((iters + 1) % batch_accumulation == 0) or (iters + 1 == len(loader)):
            optimizer.step()
            #scheduler.step()
            optimizer.zero_grad()
        
        if cfg.regularization_cartnet:
            compute_metrics_and_logging(pred = pred_node_feat.detach(), 
                                        true = true_node_feat.detach(), 
                                        u_node_loss = u_node_loss.detach(), 
                                        u_edge_loss = u_adj_loss.detach(),
                                        u_contrastive_loss = u_contr_loss.detach(),
                                        l_node_loss = node_loss.detach(), 
                                        l_edge_loss = adj_loss.detach(),
                                        l_contrastive_loss = contr_loss.detach(),
                                        l_MAE = MAE.detach(),
                                        loss = loss.detach(),
                                        lr = optimizer.param_groups[0]['lr'], 
                                        time_used = time.time()-time_start, 
                                        logger = logger)
        
        
        elif cfg.unsupervised_cartnet:
            compute_metrics_and_logging(pred = pred_node_feat.detach(), 
                                        true = true_node_feat.detach(), 
                                        node_loss = node_loss.detach(), 
                                        edge_loss = adj_loss.detach(),
                                        bt_loss = bt_loss.detach(),
                                        contrastive_loss = contr_loss.detach(),
                                        loss = loss.detach(),
                                        lr = optimizer.param_groups[0]['lr'], 
                                        time_used = time.time()-time_start, 
                                        logger = logger)
        else:
            compute_metrics_and_logging(pred = pred.detach(), 
                                        true = true.detach(), 
                                        mae = MAE.detach(), 
                                        mse = MSE.detach(),
                                        loss = loss.detach(),
                                        lr = optimizer.param_groups[0]['lr'], 
                                        time_used = time.time()-time_start, 
                                        logger = logger)

        iters += 1
    


def eval_epoch(logger, labeled_loader, unlabeled_loader, model, test_metrics=False):
    """
    Evaluate the model for one epoch.
    Args:
        logger (Logger): Logger object for logging metrics and information.
        loader (DataLoader): DataLoader object providing the dataset.
        model (nn.Module): The model to be evaluated.
        test_metrics (bool, optional): Flag to indicate if test metrics should be computed. Defaults to False.
    Raises:
        Exception: If the specified loss function in the configuration is not implemented.
    Returns:
        None
    """

    model.eval()

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    total_steps = max(len(labeled_loader), len(unlabeled_loader))
    
    with torch.no_grad():
        for step, (labeled_batch, unlabeled_batch) in enumerate(
            tqdm(zip_longest(labeled_iter, unlabeled_iter, fillvalue=None), total=total_steps, ncols=50)
        ):
            time_start = time.time()

            # Recreate iterators and reshuffle if one runs out
            if labeled_batch is None: #since it is smaller, it will get reused
                labeled_iter = iter(labeled_loader)
                labeled_batch = next(labeled_iter)

            
            # Unlabelled forward
            unlbl_batch_aug_0 = unlabeled_batch[0].to("cuda:0")
            unlbl_batch_aug_1 = unlabeled_batch[1].to("cuda:0")
            unlbl_batch_aug_2 = unlabeled_batch[2].to("cuda:0")

            u_node_loss, u_adj_loss, _, _ = node_edge_reconstr_forward(unlbl_batch_aug_0.clone(), model)
            u_contr_loss = contrastive_forward(unlbl_batch_aug_1.clone(), unlbl_batch_aug_2.clone(), model)

            unlbl_loss = cfg.alpha * u_node_loss + cfg.beta * u_adj_loss + cfg.gamma * u_contr_loss


            # Labelled forward 
            lbl_batch_aug_0 = labeled_batch[0].to("cuda:0")
            lbl_batch_aug_1 = labeled_batch[1].to("cuda:0")
            lbl_batch_aug_2 = labeled_batch[2].to("cuda:0")
            lbl_batch_original = labeled_batch[3].to("cuda:0")
        
            node_loss, adj_loss, pred_node_feat, true_node_feat = node_edge_reconstr_forward(unlbl_batch_aug_0.clone(), model)
            contr_loss = contrastive_forward(lbl_batch_aug_1.clone(), lbl_batch_aug_2.clone(), model)

            MAE, MSE = property_pred_forward(lbl_batch_original.clone(), model)

            unsupervised_lbld_loss = cfg.alpha * node_loss + cfg.beta * adj_loss + cfg.gamma * contr_loss
            supervised_lbld_loss = MAE ## ??

            # loss combination
            loss = (unlbl_loss + unsupervised_lbld_loss)*cfg.unsup_weight + supervised_lbld_loss*cfg.supervised_weight 

        
            if cfg.regularization_cartnet:
                compute_metrics_and_logging(pred = pred_node_feat.detach(), 
                                            true = true_node_feat.detach(), 
                                            u_node_loss = u_node_loss.detach(), 
                                            u_edge_loss = u_adj_loss.detach(),
                                            u_contrastive_loss = u_contr_loss.detach(),
                                            l_node_loss = node_loss.detach(), 
                                            l_edge_loss = adj_loss.detach(),
                                            l_contrastive_loss = contr_loss.detach(),
                                            l_MAE = MAE.detach(),
                                            loss = loss.detach(),
                                            lr = 0, 
                                            time_used = time.time()-time_start, 
                                            logger = logger)
            elif cfg.unsupervised_cartnet:
                compute_metrics_and_logging(pred = pred_node_feat.detach(), 
                                            true = true_node_feat.detach(), 
                                            node_loss = node_loss.detach(), 
                                            edge_loss = adj_loss.detach(),
                                            bt_loss = bt_loss.detach(),
                                            contrastive_loss = contr_loss.detach(),
                                            loss = loss.detach(),
                                            lr = 0, 
                                            time_used = time.time()-time_start, 
                                            logger = logger,
                                            test_metrics=test_metrics)
            else:
                compute_metrics_and_logging(pred = pred.detach(), 
                                            true = true.detach(), 
                                            mae = MAE.detach(), 
                                            mse = MSE.detach(),
                                            loss = loss.detach(),
                                            lr = 0, 
                                            time_used = time.time()-time_start, 
                                            logger = logger, 
                                            test_metrics=test_metrics)


def test_epoch(logger, loader, model, test_metrics=False):
    """
    Evaluate the model for one epoch.
    Args:
        logger (Logger): Logger object for logging metrics and information.
        loader (DataLoader): DataLoader object providing the dataset.
        model (nn.Module): The model to be evaluated.
        test_metrics (bool, optional): Flag to indicate if test metrics should be computed. Defaults to False.
    Raises:
        Exception: If the specified loss function in the configuration is not implemented.
    Returns:
        None
    """

    model.eval()
    
    with torch.no_grad():

        for iter, batch in tqdm(enumerate(loader), total=len(loader), ncols=50):
            time_start = time.time()
            batch.to("cuda:0")

            pred, true = model(batch, pred = True)
                
            MAE,MSE = compute_loss(pred, true)

            if cfg.loss == "MAE":
                loss = MAE
            elif cfg.loss == "MSE":
                loss = MSE
            else:
                raise Exception("Loss not implemented")
            

            compute_metrics_and_logging(pred = pred.detach(), 
                                        true = true.detach(), 
                                        mae = MAE.detach(), 
                                        mse = MSE.detach(),
                                        loss = loss.detach(),
                                        lr = 0, 
                                        time_used = time.time()-time_start, 
                                        logger = logger,
                                        test_metrics=tests_metrics)