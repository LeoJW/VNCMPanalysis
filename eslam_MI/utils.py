import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from estimators import estimate_mutual_information
import os
import json


# Check if CUDA or MPS is running
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = "cpu"

def rho_to_mi(dim, rho):
    """Obtain the ground truth mutual information from rho."""
    return -0.5 * np.log2(1 - rho**2) * dim


def mi_to_rho(dim, mi):
    """Obtain the rho for Gaussian give ground truth mutual information."""
    return np.sqrt(1 - 2**(-2.0 / dim * mi))


def linear_mut_info(x, y, threshold=1e-10):
    try:
        # Combine x and y column-wise (variables are columns)
        xy = np.hstack((x, y))
        
        # Compute joint covariance matrix once
        c_tot = np.cov(xy, rowvar=False)
        n_x = x.shape[1]  # Number of features in X
        n_y = y.shape[1]  # Number of features in Y
        
        # Extract C_x and C_y from the joint covariance matrix
        c_x = c_tot[:n_x, :n_x]
        c_y = c_tot[n_x:, n_x:]
        
        # Compute eigenvalues using eigh (faster for symmetric matrices)
        eig_tot = np.linalg.eigh(c_tot)[0]  # Returns sorted eigenvalues (ascending)
        eig_x = np.linalg.eigh(c_x)[0]
        eig_y = np.linalg.eigh(c_y)[0]
        
        # Threshold eigenvalues (avoid log(0))
        eig_tot_thr = np.maximum(eig_tot, threshold)
        eig_x_thr = np.maximum(eig_x, threshold)
        eig_y_thr = np.maximum(eig_y, threshold)
        
        # Compute log determinants
        logdet_tot = np.sum(np.log2(eig_tot_thr))
        logdet_x = np.sum(np.log2(eig_x_thr))
        logdet_y = np.sum(np.log2(eig_y_thr))
        
        # Mutual information
        info = 0.5 * (logdet_x + logdet_y - logdet_tot)
        return info if not np.isinf(info) else np.nan
    except np.linalg.LinAlgError:
        return np.nan


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class mlp(nn.Module):
    def __init__(self, dim, hidden_dim, output_dim, layers, activation):
        """Create an mlp from the configurations."""
        super(mlp, self).__init__()
        activation_fn = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
            'silu': nn.SiLU,
        }[activation]
    
        # Initialize the layers list
        seq = []
    
        # Input layer
        seq.append(nn.Linear(dim, hidden_dim))
        seq.append(activation_fn())
        nn.init.xavier_uniform_(seq[0].weight)  # Xavier initialization for input layer
    
        # Hidden layers
        for _ in range(layers):
            layer = nn.Linear(hidden_dim, hidden_dim)
            nn.init.xavier_uniform_(layer.weight)  # Xavier initialization for hidden layers
            seq.append(layer)
            seq.append(activation_fn())
    
        # Connect all together before the output
        self.base_network = nn.Sequential(*seq)
    
        # Output layer
        self.out = nn.Linear(hidden_dim, output_dim)
        
        # Initialize the layer with Xavier initialization
        nn.init.xavier_uniform_(self.out.weight)
    
    def forward(self, x):
        x = self.base_network(x)
        
        # Get output
        out = self.out(x)
        
        return out


class var_mlp(nn.Module):
    def __init__(self, dim, hidden_dim, output_dim, layers, activation):
        """Create a variational mlp from the configurations."""
        super(var_mlp, self).__init__()
        activation_fn = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
            'silu': nn.SiLU,
        }[activation]
    
        # Initialize the layers list
        seq = []
    
        # Input layer
        seq.append(nn.Linear(dim, hidden_dim))
        seq.append(activation_fn())
        nn.init.xavier_uniform_(seq[0].weight)  # Xavier initialization for input layer
    
        # Hidden layers
        for _ in range(layers):
            layer = nn.Linear(hidden_dim, hidden_dim)
            nn.init.xavier_uniform_(layer.weight)  # Xavier initialization for hidden layers
            seq.append(layer)
            seq.append(activation_fn())
    
        # Connect all together before the output
        self.base_network = nn.Sequential(*seq)
    
        # Two heads for means and log variances
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_logvar = nn.Linear(hidden_dim, output_dim)
        
        # Initialize the heads with Xavier initialization
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.xavier_uniform_(self.fc_logvar.weight)
        
        # Normal distribution for sampling
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        
        # KL Divergence loss initialized to zero
        self.kl_loss = 0.0
        
        # Set limits for numerical stability
        self.logvar_min = -20  # Lower bound for logVar
        self.logvar_max = 20   # Upper bound for logVar

    def forward(self, x):
        x = self.base_network(x)
        
        # Get mean and log variance
        meanz = self.fc_mu(x)
        logVar = self.fc_logvar(x)

        # Clamp logVar to prevent extreme values
        logVar = torch.clamp(logVar, min=self.logvar_min, max=self.logvar_max)
        
        # Compute KL divergence loss
        kl_terms = 0.5 * (torch.square(meanz) + torch.exp(logVar) - 1 - logVar)
        self.kl_loss = torch.mean(torch.sum(kl_terms, dim=1))

        # Check for NaN in KL loss
        if torch.isnan(self.kl_loss):
            print("NaN detected in KL loss!")
            # Use a small default value instead of NaN
            self.kl_loss = torch.tensor(0.1, device=device, requires_grad=True)
        
        # Reparameterization trick
        epsilon = self.N.sample(meanz.shape)
        std = torch.exp(0.5 * logVar)
        samples = meanz + std * epsilon
        return [meanz, logVar, samples]


def log_prob_gaussian(x):
    return torch.sum(torch.distributions.Normal(0., 1.).log_prob(x), -1)


class decoder_INFO(nn.Module):
    def __init__(self, typeEstimator, mode="sep", baseline_fn=None):
        super(decoder_INFO, self).__init__()
        
        self.estimator = typeEstimator
        self.baseline_fn = baseline_fn
        self.mode = mode  # "sep" and "bi" use the same critic function

    def critic_fn(self, dataZX, dataZY, batch_size=None):
        if self.mode in ["sep", "bi"]:  
            return torch.matmul(dataZY, dataZX.t())
        elif self.mode == "concat":
            return torch.reshape(dataZX, [batch_size, batch_size]).t() # Here dataZX is really the final scores matrix
        else:
            raise ValueError("Invalid mode. Choose 'sep', 'bi', or 'concat'.")

    def forward(self, dataZX, dataZY, batch_size=None):
        return estimate_mutual_information(self.estimator, dataZX, dataZY,
                                        lambda x, y: self.critic_fn(x, y, batch_size),
                                        baseline_fn=self.baseline_fn)

def write_config(args):
    out_fn = "config.json"
    out_fp = os.path.join(args.save_dir, out_fn)
    with open(out_fp, 'w') as fh:
        json.dump(vars(args), fh)
