# Unsupervised atom embedding generation for crystal property prediction

This project explores how **unsupervised datasets** can be leveraged to improve **crystal property prediction** using **Graph Neural Networks (GNNs)**. Specifically, we use CartNet as the base model and investigate various **pretraining** and **multitasking** strategies to improve performance on labeled datasets.

---

## 📁 Folder Structure

### `unsupervised_CartNet/`
Contains code and configuration files for using the **unsupervised dataset**. Supports two training modes:
- **Pretraining**: Uses the unsupervised data in a dual-branch architecture:
  - **Autoencoder branch**: Reconstructs input graph (node features and edges) (optionally denoised).
  - **Self-supervised branch**: Can use **Barlow Twins loss**, **contrastive learning (SimCLR)**, or **Deep Graph Infomax**.
- **Multitask training**: Simultaneously uses both unsupervised and labeled datasets.

After pretraining:
- The model weights are saved.
- Atom-level embeddings can be extracted and visualized with **t-SNE**.

---

### `supervised_CartNet/`
Contains code to train CartNet for the main task: **crystal property prediction** using **labeled datasets**.

Training options:
- **From scratch**: Baseline performance.
- **With pretraining**: Uses weights and/or embeddings from the unsupervised stage. Multiple strategies are supported for incorporating pretrained outputs.

---

## 🚀 How to Run

### 1. Set Up Environment
Each folder contains an environment file (e.g., `environment_2.yml`). Use `conda` to create the environment:
```bash
conda env create -f environment_2.yml
conda activate your_env_name
```

### 2. Training the models
Each folder contains multiple .sh scripts to execute the code. In order to reproduce the results of the paper, the recommended ones are:

- Unsupervised training:
  ```bash
  sbatch train_cartnet_MP_1run.sh
  ```

- Supervised training:
   ```bash
  sbatch train_cartnet_megnet.sh
  ```
The scripts can be modified in order to control the multiple hyperparameters and variants of the code.
  

