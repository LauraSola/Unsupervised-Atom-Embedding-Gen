import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.manifold import TSNE
import random


#./dense_vector/dense_vector_dim 32 pesos finals guardar embd.pkl
#./dense_vector/dense_vector_individual_dim_32_pesos_finals_SI.pkl
#"./dense_vector/dense_vector_mean_dim_32_combined_aug.pkl"
# "/mnt/work/users/laura.sola.garcia/SimCLR/dense_vector/dense_vector_mean_triplet.pkl"


with open("/mnt/work/users/laura.sola.garcia/SimCLR/dense_vector/dense_vector_individual_triplet.pkl", 'rb') as f:
    atomic_embeddings = pickle.load(f)

######## CODI 1 ###################################################

# atomic_number_to_symbol = {
#     1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
#     11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
#     21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
#     31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr",
#     41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
#     51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd",
#     61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb",
#     71: "Lu", 72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg",
#     81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th",
#     91: "Pa", 92: "U", 93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf", 99: "Es", 100: "Fm",
#     101: "Md", 102: "No", 103: "Lr"
# }

# # Element classifications based on atomic number
# element_types = {
#     "Alkali metals": [3, 11, 19, 37, 55, 87],
#     "Alkaline earth metals": [4, 12, 20, 38, 56, 88],
#     "Transition metals": list(range(21, 31)) + list(range(39, 49)) + list(range(72, 81)),
#     "Post-transition metals": [13, 31, 49, 50, 81, 82, 83, 84],
#     "Metalloids": [5, 14, 32, 33, 51, 52],
#     "Reactive non-metals": [1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53],
#     "Noble gases": [2, 10, 18, 36, 54, 86],
#     "Lanthanides": list(range(57, 72)),
#     "Actinides": list(range(89, 104))
# }

# # Assign colors to each group
# type_colors = {etype: color for etype, color in zip(element_types.keys(), sns.color_palette("tab10", n_colors=len(element_types)))}


# # Ensure the data is in a proper format
# atomic_numbers = list(atomic_embeddings.keys())  # List of element symbols or atomic numbers
# vectors = np.array([v.cpu().detach().numpy().squeeze() for v in atomic_embeddings.values()]) # Corresponding embeddings
# #vectors = np.array([v.squeeze() for v in atomic_embeddings.values()]) 
# # Identify non-zero vectors
# non_zero_mask = ~np.all(vectors == 0, axis=1)

# # Filter elements and vectors
# filtered_atomic_numbers = [e for e, keep in zip(atomic_numbers, non_zero_mask) if keep]
# filtered_vectors = vectors[non_zero_mask]

# print("Filtered elements:", filtered_atomic_numbers)
# print("Filtered vectors shape:", filtered_vectors.shape)

# # Apply t-SNE
# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# tsne_results = tsne.fit_transform(filtered_vectors)

# # Convert results into x, y coordinates
# x, y = tsne_results[:, 0], tsne_results[:, 1]

# # Assign a color to each element based on its type
# element_colors = []
# for atomic_number in filtered_atomic_numbers:
#     for etype, num_list in element_types.items():
#         if atomic_number in num_list:
#             element_colors.append(type_colors[etype])
#             break
#     else:
#         element_colors.append("black")  # Default color for unknown elements


# # Create a scatter plot
# plt.figure(figsize=(10, 7))
# sns.set(style="whitegrid")

# # Scatter plot of elements
# plt.scatter(x, y,c=element_colors, alpha=0.7)

# # Annotate each point with its element symbol
# for i, atomic_number in enumerate(filtered_atomic_numbers):
#     element_symbol = atomic_number_to_symbol.get(atomic_number, str(atomic_number))  # Get symbol or fallback to number
#     plt.annotate(element_symbol, (x[i], y[i]), fontsize=9, ha="right", va="bottom")


# plt.xlabel("t-SNE Component 1")
# plt.ylabel("t-SNE Component 2")
# plt.title("t-SNE Visualization of Atomic Embeddings")

# plt.savefig("/mnt/work/users/laura.sola.garcia/SimCLR/tsne_atomic_embeddings_triplet.png", dpi=300, bbox_inches="tight")
# plt.close()

# print("Plot saved as tsne_atomic_embeddings.png")

######## CODI 2 ###################################################

# Select first 10 atomic numbers
#selected_atomic_numbers = sorted(atomic_embeddings.keys())[:20]
selected_atomic_numbers = [1, 6, 17, 9, 7, 16, 35, 53, 8]
filtered_embeddings = {}

# Randomly select up to 10 embeddings per atomic number
for atomic_number in selected_atomic_numbers:
    embeddings = atomic_embeddings[atomic_number]  # List of tensors or numpy arrays

    # Convert tensors to numpy arrays if necessary
    embeddings = [e.cpu().detach().numpy().squeeze() if isinstance(e, torch.Tensor) else e for e in embeddings]

    # Remove zero vectors
    non_zero_embeddings = [e for e in embeddings if not np.all(e == 0)]

    # Randomly sample up to 10 embeddings
    sampled_embeddings = random.sample(non_zero_embeddings, min(200, len(non_zero_embeddings)))

    # Store in dictionary
    if sampled_embeddings:
        filtered_embeddings[atomic_number] = sampled_embeddings

# Flatten data for t-SNE
all_elements = []
all_vectors = []
all_labels = []

for atomic_number, embeddings in filtered_embeddings.items():
    for emb in embeddings:
        all_elements.append(atomic_number)
        all_vectors.append(emb)
        all_labels.append(str(atomic_number))  # Convert atomic number to string for labels

# Convert to numpy array
all_vectors = np.array(all_vectors)

# Check that data is not empty
if all_vectors.shape[0] == 0:
    raise ValueError("No valid embeddings found after filtering. Check your data.")

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=10, random_state=42)
tsne_results = tsne.fit_transform(all_vectors)

# Convert results into x, y coordinates
x, y = tsne_results[:, 0], tsne_results[:, 1]

# Create scatter plot
plt.figure(figsize=(10, 7))
sns.set(style="whitegrid")

# Assign unique colors to each atomic number
unique_elements = list(set(all_labels))
colors = sns.color_palette("tab10", len(unique_elements))
color_map = {element: colors[i] for i, element in enumerate(unique_elements)}

# Plot each point
for i in range(len(x)):
    plt.scatter(x[i], y[i], color=color_map[all_labels[i]], alpha=0.7, 
                label=all_labels[i] if all_labels[i] not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.annotate(all_labels[i], (x[i], y[i]), fontsize=8, ha="right", va="bottom")

# Legend for atomic numbers
plt.legend(title="Atomic Number", loc="best", fontsize=9)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization of Atomic Embeddings (First 10 Elements, Max 10 Each)")

# Save plot
plt.savefig("/mnt/work/users/laura.sola.garcia/SimCLR/tsne_atomic_embeddings_individual_triplet.png", dpi=300, bbox_inches="tight")
plt.close()

print("Plot saved as tsne_atomic_embeddings_filtered.png")