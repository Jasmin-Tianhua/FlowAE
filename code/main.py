import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE

from vasc_pytorch import vasc, plot_metrics
from helpers_pytorch import clustering, measure, print_2D
from config_pytorch import config
from prior_pytorch import FlowPrior


def main():

    # ============================================================
    # Data Preparation
    # ============================================================

    expr = data_normalized_tensor.numpy()
    label = true_labels.numpy()

    # ============================================================
    # Flow Prior Configuration
    # ============================================================

    latent_dim = 16
    hidden_dim = 26

    nets = lambda: nn.Sequential(
        nn.Linear(latent_dim // 2, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, latent_dim // 2)
    )

    nett = lambda: nn.Sequential(
        nn.Linear(latent_dim // 2, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, latent_dim // 2)
    )

    flow_prior = FlowPrior(
        nets,
        nett,
        num_flows=6,
        D=latent_dim
    )

    # ============================================================
    # Autoencoder Configuration
    # ============================================================

    hidden_dim1 = 1024
    hidden_dim2 = 512
    hidden_dim3 = 64

    dropout = 0.13023409024310986
    beta = 0.9

    # ============================================================
    # Training
    # ============================================================

    for run_idx in range(1):

        print(f"Iteration: {run_idx}")

        (
            res,
            loss_list,
            rec_list,
            prior_list,
            nmi_list,
            acc_list,
            ari_list,
            evaluation_epochs
        ) = vasc(
            expr,
            epoch=300,
            latent=16,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            hidden_dim3=hidden_dim3,
            dropout=dropout,
            beta=beta,
            var=False,
            annealing=False,
            batch_size=256,
            prefix="test",
            label=label,
            scale=config["scale"],
            patience=5,
            flow_prior=flow_prior
        )

        print("Embedding shape:", res.shape)

        # ========================================================
        # Clustering Evaluation
        # ========================================================

        k = len(np.unique(label))

        clustered_labels, _ = clustering(
            res,
            k=k
        )

        metrics = measure(
            clustered_labels,
            label
        )

        print(metrics)

        # ========================================================
        # Save Results to AnnData
        # ========================================================

        adata.obsm["X_vasc"] = res

        adata.obs["vasc_clusters"] = pd.Categorical(
            values=clustered_labels
        )

        # Optional:
        # adata.write_h5ad("adata_with_flowae.h5ad")

        # ========================================================
        # Latent Space Visualization
        # ========================================================

        fig = print_2D(
            points=res,
            label=label,
            id_map={i: str(i) for i in range(k)}
        )

        plt.show()

        # ========================================================
        # Training Curves
        # ========================================================

        plot_metrics(
            loss_list,
            rec_list,
            prior_list,
            nmi_list,
            acc_list,
            ari_list,
            evaluation_epochs,
            filename=f"metrics_curve_{run_idx}.png"
        )

        # ========================================================
        # t-SNE Visualization
        # ========================================================

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate=200,
            random_state=42
        )

        res_tsne = tsne.fit_transform(res)

        plt.figure(
            figsize=(8, 6),
            dpi=600
        )

        sns.scatterplot(
            x=res_tsne[:, 0],
            y=res_tsne[:, 1],
            hue=label,
            palette="tab10",
            s=30,
            edgecolor="none",
            legend=False
        )

        plt.title("t-SNE of Qx_Muscle (FlowAE Model)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")

        plt.tight_layout()

        plt.savefig(
            "tsne_cluster_visualization.png",
            dpi=600
        )

        plt.show()


if __name__ == "__main__":

    print("Starting FlowAE training...")

    main()

    print("Finished")

