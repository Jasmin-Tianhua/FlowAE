# FlowAE: A Generative Representation Learning Framework via Autoencoder-Flow Integration for Single-Cell Clustering

FlowAE is a novel generative deep learning framework that integrates Autoencoder (AE) and Normalizing Flow (NF) for effective representation learning and clustering of single-cell RNA sequencing (scRNA-seq) data.

## Framework Overview

<p align="center">
  <img src="images/framework.png" alt="FlowAE Framework" width="900">
</p>

**Figure 1.** Overall architecture of FlowAE. The framework combines an Autoencoder for nonlinear representation learning with a Normalizing Flow prior to enhance latent space expressiveness, resulting in more discriminative embeddings for scRNA-seq clustering.

---

## 🚀 Features

* End-to-end generative representation learning.
* Integration of Autoencoder and Normalizing Flow for expressive latent distribution modeling.
* Superior clustering performance on multiple real scRNA-seq datasets.
* Visualization support with t-SNE and UMAP.
* Modular PyTorch implementation for easy extension and reproducibility.

## 🧩 Dependencies

This project was developed and tested with the following environment:

* **Python** >= 3.8
* **PyTorch** >= 1.10
* **NumPy**
* **Pandas**
* **scanpy**
* **scikit-learn**
* **matplotlib**
* **seaborn**
* **(Optional)** CUDA-enabled GPU for faster training

## 📂 Project Structure

```text
FlowAE/
├── images/
│   └── framework.png
├── main.py
├── prior_pytorch.py
└── vasc_pytorch.py
```


