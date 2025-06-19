import os
import shutil
import random
import torch
import yaml


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

############ AUGMENTATIONS

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
