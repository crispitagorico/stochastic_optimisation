import torch
import numpy as np


def generate_gbm(n_samples: int, n_days: int, mu, vol, dt: torch.float64, dim: int = 1):

    voldt = np.sqrt(dt) * vol
    mudt = dt * mu
    z = torch.randn(n_samples, n_days, dim)
    paths = (voldt * z - voldt**2 / 2. + mudt).cumsum(1).exp()
    x = torch.cat([torch.ones(n_samples, 1, dim), paths], dim=1)
    return x


def generate_multi_gbm(n_samples: int, n_days: int, mu_vec: torch.tensor, cov_mat: torch.tensor, dt: torch.float64, dim: int):
    n_steps = n_days*dt

    # case: given correlation matrix, create paths for multiple correlated processes
    if cov_mat.shape[0] > 1:
        # paths = torch.zeros(size = (n_samples, n_steps, dim))
        # loop through number of paths
        dim = mu_vec.shape[0]

        choleskyMatrix = torch.linalg.cholesky(cov_mat)
        e = torch.normal(0, 1, size=(n_samples, n_steps, dim))
        noise = torch.matmul(e, choleskyMatrix)

        paths = torch.cumsum(noise, dim=1)

        for i in range(dim):
            paths[:, :, i] = torch.exp(torch.arange(n_steps) * mu_vec[i] + paths[:, :, i])

        paths = torch.cat([torch.ones(n_samples, 1, dim), paths], dim=1)

    # case: no given correlation matrix, create paths for a single process
    else:
        paths = generate_gbm(n_samples, n_days, mu_vec[0], cov_mat[0], dt, dim)
    return paths


def generate_heston_paths(n_samples: int,
                          n_days: int,
                          dt: torch.float64,
                          dim: int,
                          S_0: int,
                          V_0: torch.float64,
                          kappa: torch.float64,
                          theta: torch.float64,
                          nu: torch.float64,
                          rho: torch.float64):

    n_steps = n_days*dt

    # first, generate 2 correlated brownian motion paths

    dW1 = torch.normal(0, 1, size=(n_samples, n_steps, dim))
    dW2 = rho*dW1 + torch.sqrt(1-rho**2)*torch.normal(0, 1, size=(n_samples, n_steps, dim))

    paths = torch.ones(size=(n_samples, n_steps, 2))

    paths[:, 0, 0] = V_0
    paths[:, 0, 1] = S_0

    for i in range(1, n_steps):
        paths[:, i, 0] = paths[:, i-1, 0] + kappa*(theta - paths[:, i-1, 0])*dt + nu*torch.sqrt(paths[:, i-1, 0])*dW1[:, i-1, 0]  # Euler discretisation
        paths[:, i, 0] = torch.abs(paths[:, i, 0])  # Ensuring positive volatility by reflecting
        paths[:, i, 1] = paths[:, i-1, 1]*torch.exp(-0.5*paths[:, i-1, 0]*dt + torch.sqrt(paths[:, i-1, 0])*dW2[:, i-1, 0])  # Euler discretisation

    return paths


def generate_OU_paths(n_samples: int,
                      n_days: int,
                      dt: torch.float64,
                      dim: int,
                      S_0: int,
                      theta: torch.float64,
                      sigma: torch.float64,
                      mu: torch.float64):

    n_steps = int(n_days/dt)

    dW1 = torch.normal(0, 1, size=(n_samples, n_steps, dim))
    paths = torch.ones(size=(n_samples, n_steps, 1))

    paths[:, 0, 0] = S_0

    for i in range(1, n_steps):
        paths[:, i, 0] = paths[:, i-1, 0] + (theta*(mu - paths[:, i-1, 0])*dt + sigma * dW1[:, i-1, 0])

    return paths
