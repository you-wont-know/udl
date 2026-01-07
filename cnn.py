from torch import nn
import torch
import copy

def clone_and_freeze(network):
    cloned = copy.deepcopy(network)
    cloned.eval()
    for param in cloned.parameters():
        param.requires_grad = False
    return cloned

def to_one_hot(Y, num_classes):
    if Y.dim() == 1 or (Y.dim() == 2 and Y.shape[1] == 1):
        return nn.functional.one_hot(Y.view(-1), num_classes=num_classes)
    return Y

# Main CNN used within the paper
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(             # input is (1, 28, 28)
            nn.Conv2d(1, 32, kernel_size=4),  # now (32, 25, 25)
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4), # now (32, 22, 22)
            nn.ReLU(),
            nn.MaxPool2d(2),                  # now (32, 11, 11)
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(32 * 11 * 11, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)                # output logits only
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1))
    
    def forward_probs(self, x, t=1):
        # forward times and average the post-softmax ouputs
        # So here we output not logits but probabilities
        avg_probs = torch.zeros(x.shape[0], 10, device=x.device)
        for _ in range(t):
            logits = self.net(x.unsqueeze(1))
            probs = nn.functional.softmax(logits, dim=1)
            avg_probs += probs
        avg_probs /= t
        return avg_probs
    

class AnalyticalCNN(nn.Module):
    def __init__(self, s, sigma_epsilon, phi, k):
        super().__init__()
        self.phi = clone_and_freeze(phi)  # feature extractor, outputs (n, k) batches 

        self.s = s      # std for the prior on w
        self.k = k      # feature dimension
        self.sigma_epsilon = sigma_epsilon

        # These will be recalculated after "train()"
        self.V = torch.zeros((k, k))
        self.M = torch.zeros((10, k))

    def fit(self, X, Y):
        # Computes the parameters needed for predictions
        n = X.shape[0]
        assert(Y.shape[0] == n)
        # transform the data to one-hot if needed
        Y = to_one_hot(Y, num_classes=10)
        Y = Y.to(dtype=X.dtype)

        Phi = self.phi(X)
        eye = torch.eye(self.k, device=Phi.device, dtype=Phi.dtype)
        self.V = torch.linalg.inv(Phi.T @ Phi + eye / (self.s ** 2))
        self.M = Y.T @ Phi @ self.V

    def forward(self, X):
        # The paper derivation was for a column vector,
        # but now we have row vectors, hence everything is
        # transposed
        Phi = self.phi(X)                               # (n, k)
        mean = Phi @ self.M.T                           # (n, 10)
        scale = 1.0 + (Phi @ self.V * Phi).sum(dim=1)   # (n, 1)
        cov = self.sigma_epsilon.unsqueeze(0) * scale.view(-1, 1, 1)    # (n, 10, 10)
        return mean, cov

class VariationalCNN(nn.Module):
    def __init__(self, s, sigma, phi, k):
        super().__init__()
        self.phi = clone_and_freeze(phi)  # feature extractor, outputs (n, k) batches 

        self.s = s              # std for the prior on w
        self.sigma = sigma      # std for the likelihood
        self.k = k              # feature dimension
        self.o = 10             # output dimension

        # Define the learnable parameters
        self.M = nn.Parameter(torch.randn(self.o, k) * 0.01)
        self.log_S = nn.Parameter(torch.randn(self.o, k) * 0.01)

    def elbo(self, X, Y, scale=1.0):
        # This functions implements the loss for this model.
        # Not very standard to put loss into the model, but this works.
        # All constants are dropped here.

        Y = to_one_hot(Y, num_classes=self.o)
        Y = Y.to(dtype=X.dtype)

        S = torch.exp(self.log_S)
        Phi = self.phi(X)

        mean = Phi @ self.M.T
        squared_diff = (Y - mean).pow(2).sum()
        
        # equivalent way of doing Phi @ trace(S.T @ S) @ Phi.T, just faster
        diag_sts = (S * S).sum(dim=0)
        phi_sq = Phi * Phi
        trace_term = (phi_sq * diag_sts).sum()

        # Constants for the ELBO (can be dropped during optimization if desired).
        sigma2 = self.sigma ** 2
        s2 = self.s ** 2

        expected_ll = (
            -0.5 / sigma2 * (squared_diff + trace_term)
        ) * scale
        kl = (
            0.5 / s2 * (S * S).sum()
            + 0.5 / s2 * (self.M * self.M).sum()
            - self.log_S.sum()
        )
        return expected_ll - kl

    def forward(self, X):
        # The paper derivation was for a column vector,
        # but now we have row vectors, hence everything is
        # transposed.
        Phi = self.phi(X)
        mean = Phi @ self.M.T

        S = torch.exp(self.log_S) 
        S_sq = S * S
        phi_sq = Phi * Phi
        var = phi_sq @ S_sq.T
        diag = var + (self.sigma ** 2)
        cov = torch.diag_embed(diag)
    
        return mean, cov
