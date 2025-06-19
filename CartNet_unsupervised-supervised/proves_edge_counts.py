
import torch
#from dataset.utils import compute_knn, compute_inf_potnet, compute_levels
from torch_geometric.graphgym.config import cfg
from torch_geometric.loader import DataLoader
import random
import os.path as osp
import numpy as np
from dataset.datasetMP import DatasetMP
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm


def create_train_val_test(data, val_ratio=0.1, test_ratio=0.1, seed=123):
    ids = list(np.arange(len(data)))
    n = len(data)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test
    random.seed(seed)
    random.shuffle(ids)
    ids_train = ids[:n_train]
    ids_val = ids[-(n_val + n_test): -n_test]
    ids_test = ids[-n_test:]
    return ids_train, ids_val, ids_test


def main():
    dat = glob.glob(osp.join("./dataset/mp_139K/", "*.cif"))
    print(len(dat))
    seed = 123
    ids_train, ids_val, ids_test = create_train_val_test(dat, seed=seed) 
    dat_train = [dat[i] for i in ids_train] 
    dat_val = [dat[i] for i in ids_val]
    dat_test = [dat[i] for i in ids_test]

    dataset_train, dataset_val, dataset_test = (DatasetMP(root="./dataset/mp_139K/", data=dat_train, radius=5, max_neigh=-1, name="MP_train"),
                                                 DatasetMP(root="./dataset/mp_139K/", data=dat_val, radius=5, max_neigh=-1, name="MP_val"),
                                                 DatasetMP(root="./dataset/mp_139K/", data=dat_test, radius=5, max_neigh=-1, name="MP_test")
                                             )


    loaders = [
        DataLoader(dataset_train, batch_size=64, persistent_workers=True,
                                  shuffle=True, num_workers=4,
                                  pin_memory=True),
        DataLoader(dataset_val, batch_size=64, persistent_workers=True,
                                    shuffle=False, num_workers=4,
                                    pin_memory=True),
        DataLoader(dataset_test, batch_size=64, persistent_workers=False,
                                    shuffle=False, num_workers=4,
                                    pin_memory=True)
    ]

    edge_counts = []
    for loader in loaders:

        for iter, batch in tqdm(enumerate(loader), total=len(loader), ncols=50):
            adj_matrix = to_dense_adj(batch.edge_index, batch.batch)  # Convert to dense adjacency matrix
            adj_matrix = adj_matrix.long()  # Ensure integer format
            
            unique, counts = torch.unique(adj_matrix, return_counts=True)  # Count occurrences of values
            edge_histogram = torch.zeros(unique.max().item() + 1, dtype=torch.long)  # Prepare storage
            
            edge_histogram[unique] = counts  # Store counts in vector
            
            edge_counts.append(edge_histogram)

    #ensure consistent lengths in order to sum
    max_length = max(hist.size(0) for hist in edge_counts)  # Find the max length across all histograms
    edge_counts = [torch.cat([hist, torch.zeros(max_length - hist.size(0), dtype=torch.long)]) for hist in edge_counts]

    # Aggregate counts across all batches
    final_histogram = torch.sum(torch.stack(edge_counts), dim=0)

    hist_remove_0 = final_histogram[1:]
    # Compute percentages
    total_count = hist_remove_0.sum().item()
    percentages = (hist_remove_0.float() / total_count) * 100  # Convert to percentages

    # Convert to NumPy for plotting
    values = torch.arange(1,len(final_histogram)).numpy()
    counts = final_histogram.numpy()
    percentages = percentages.numpy()

    # Plot histogram
    plt.figure(figsize=(10, 6))
    sns.barplot(x=values, y=percentages, color="blue", alpha=0.7)
    plt.xlabel("Edge Multiplicity")
    plt.ylabel("Percentage (%)")
    plt.title("Distribution of Edge Multiplicity in Adjacency Matrices")
    # Improve x-axis readability
    plt.xticks(ticks=values[::5], labels=values[::5], rotation=45)  # Show every 5th value, rotate labels

    # Save figure
    plt.savefig("edge_distribution.png", dpi=300)
    plt.show()

    print("Final Histogram:", final_histogram)
    print("Percentage Vector:", percentages)

                


if __name__ == '__main__':
    main()

# Final Histogram: tensor([3243946751,   74660102,   21145352,    4505220,    1982965,     373518,
#             552024,      68714,      57787,      14546,      18757,        984,
#              53666,       1070,       4189,        890,       5764,        192,
#               2231,        586,        512,         54,         88,          2,
#                512,         14,        238,          0,        100,          0,
#                 96,          0,        190,          0,          3,          0,
#                 34,          0,        120,          0,          0,          0,
#                 52,          0,          6,          0,          4,          0,
#                  0,          0,          1,          0,          0,          0,
#                  3,          0,          1,          0,          9,          0,
#                  1,          0,          0,          0,          0,          0,
#                  0,          0,          0,          0,          1,          0,
#                  0,          0,          1,          0,          0,          0,
#                  1,          0,          1])


# adj_matrix = to_dense_adj(batch.edge_index, batch.batch)  # Convert to dense adjacency matrix
# adj_matrix = torch.clamp(adj_matrix, max=5)  # Limit values to a maximum of 5