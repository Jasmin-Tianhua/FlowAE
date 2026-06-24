import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
from helpers_pytorch import clustering  # Ensure this is imported from your helpers file
import math
import helpers_pytorch
from helpers_pytorch import measure
class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

import torch.nn as nn
import math

import torch
import torch.nn as nn
import math

class VASC(nn.Module):
    def __init__(self, in_dim, latent, dropout, hidden_dim1, hidden_dim2, hidden_dim3, beta, var=False, flow_prior=None):
        super(VASC, self).__init__()
        self.in_dim = in_dim
        self.latent = latent
        self.var = var
        self.flow_prior = flow_prior
        self.beta = beta  
        
        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU()
        )
        
        self.z_mean = nn.Linear(hidden_dim3, latent)
        self.z_log_var = nn.Linear(hidden_dim3, latent) if var else None
        
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden_dim3),
            nn.ReLU(),
            nn.Linear(hidden_dim3, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, in_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder(x)
        z_mean = self.z_mean(h)
        z = z_mean
        if self.var:
            z_log_var = self.z_log_var(h)
            z = self.sampling(z_mean, z_log_var)
        x_decoded = self.decoder(z)
        return x_decoded, z_mean

    def prior_log_prob(self, z):
        if self.flow_prior:
            return self.flow_prior.log_prob(z)
        else:
            return -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(dim=1)



def loss_function(x, x_decoded, z_mean, z_log_var=None, prior_log_prob=None, beta=1.0):
    reconstruction_loss = F.binary_cross_entropy(x_decoded, x, reduction='sum')
    kl_loss = 0
    if z_log_var is not None:
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    elif prior_log_prob is not None:
        kl_loss = -prior_log_prob.sum()
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss  





import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def vasc(expr, epoch, latent, dropout, hidden_dim1, hidden_dim2, hidden_dim3, beta,
         patience=50, min_stop=500, batch_size=32, var=False, prefix='test', label=None,
         log=True, scale=True, annealing=False, tau0=1.0, min_tau=0.5, rep=0, flow_prior=None):

    if log:
        expr = np.log2(expr + 1)
    if scale:
        expr = expr / np.max(expr, axis=1, keepdims=True)

    expr_train = np.tile(expr, (rep, 1)) if rep > 0 else expr.copy()
    expr_tensor = torch.tensor(expr_train, dtype=torch.float32)
    dataset = TensorDataset(expr_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VASC(in_dim=expr.shape[1], latent=latent, dropout=dropout,
                 hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, hidden_dim3=hidden_dim3,
                 beta=beta, var=var, flow_prior=flow_prior)
    optimizer = optim.RMSprop(model.parameters(), lr=0.0001)

    best_loss = float('inf')
    best_model = None

    loss_list, rec_list, prior_list = [], [], []
    nmi_list, acc_list, ari_list, evaluation_epochs = [], [], [], []

    model.train()
    for e in range(epoch):
        epoch_total_loss, epoch_rec_loss, epoch_prior_loss = 0, 0, 0

        for batch in dataloader:
            optimizer.zero_grad()
            x = batch[0]
            x_decoded, z_mean = model(x)
            z_log_var = model.z_log_var(x) if var else None
            prior_log_prob = model.prior_log_prob(z_mean) if flow_prior else None

            loss, rec_loss, prior_loss = loss_function(x, x_decoded, z_mean, z_log_var, prior_log_prob, beta=beta)
            loss.backward()
            optimizer.step()

            # 
            epoch_total_loss += loss.item()
            epoch_rec_loss += rec_loss.item()
            epoch_prior_loss += prior_loss.item()

        # 
        epoch_total_loss /= len(dataloader)
        epoch_rec_loss /= len(dataloader)
        epoch_prior_loss /= len(dataloader)

        loss_list.append(epoch_total_loss)
        rec_list.append(epoch_rec_loss)
        prior_list.append(epoch_prior_loss)

        if epoch_total_loss < best_loss:
            best_loss = epoch_total_loss
            best_model = model.state_dict()

        # 
        if e % patience == 0 or e == epoch - 1:
            model.eval()
            with torch.no_grad():
                expr_tensor_full = torch.tensor(expr, dtype=torch.float32)
                x_decoded, z_mean = model(expr_tensor_full)

            predicted_labels = z_mean.numpy()
            clustered_labels, _ = clustering(predicted_labels, k=len(np.unique(label)), name='kmeans')
            metrics = measure(clustered_labels, label)

            nmi_list.append(metrics['NMI'])
            acc_list.append(metrics['ACC'])
            ari_list.append(metrics['ARI'])
            evaluation_epochs.append(e)

            print(f"Epoch {e+1}/{epoch}, NMI: {metrics['NMI']:.4f}, ARI: {metrics['ARI']:.4f}, ACC: {metrics['ACC']:.4f}")
            model.train()

            if e > min_stop and epoch_total_loss < 1e-3:
                break

    model.load_state_dict(best_model)

    model.eval()
    with torch.no_grad():
        expr_tensor_full = torch.tensor(expr, dtype=torch.float32)
        x_decoded, z_mean = model(expr_tensor_full)

    return z_mean.numpy(), loss_list, rec_list, prior_list, nmi_list, acc_list, ari_list, evaluation_epochs







import matplotlib.pyplot as plt

def plot_metrics(loss_list, rec_list, prior_list, nmi_list, acc_list, ari_list, evaluation_epochs, filename='metrics_curve.png'):
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=600)
    
    epochs = range(len(loss_list))  # 
    
    # 
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    l1, = ax1.plot(epochs, loss_list, label='Total Loss', color='black', linestyle='--', linewidth=2)
    l2, = ax1.plot(epochs, rec_list, label='Reconstruction Loss', color='blue', linewidth=2)
    l3, = ax1.plot(epochs, prior_list, label='Flow Prior Loss', color='red', linewidth=2)
    ax1.tick_params(axis='y')
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    # 
    ax2 = ax1.twinx()
    ax2.set_ylabel('Clustering Metrics', fontsize=12)
    l4, = ax2.plot(evaluation_epochs, nmi_list, label='NMI', color='green', marker='o', markersize=6, linewidth=2)
    l5, = ax2.plot(evaluation_epochs, acc_list, label='ACC', color='orange', marker='s', markersize=6, linewidth=2)
    l6, = ax2.plot(evaluation_epochs, ari_list, label='ARI', color='purple', marker='^', markersize=6, linewidth=2)
    ax2.tick_params(axis='y')
    
    # 
    lines = [l1, l2, l3, l4, l5, l6]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10, frameon=True)
    
    plt.title('Training Losses and Clustering Metrics', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.show()





