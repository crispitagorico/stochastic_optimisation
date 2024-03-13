import torch
from typing import Tuple


def augment_with_time(x: torch.Tensor,
                      T: torch.Tensor = 1.0,
                      grid: torch.Tensor = None) -> torch.Tensor:
    """
    Returns the time augmented (in dimension 0) paths i.e

        X_t --> (t,X_t)

    Parameters
        ----------
        x: (batch, timesteps, d)
        grid: (timesteps)

    Returns
        -------
        x_augmented : (batch, timesteps, 1+d)
    """

    if grid is None:
        grid = torch.linspace(0, T, (x.shape[1]))

    grid = grid.to(x.device.type)
    x_augmented = torch.cat((grid.expand(x.shape[0], -1).unsqueeze(-1), x),
                            dim=-1)

    return x_augmented


def Hoff_transform(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the forward and backward component of the Hoff transform of paths in tensor x.
    Recall how the transform of a path X sampled at times t_k has been defined as

        X^f_t = X_{t_k}                                                             if t \in (t_k, t_k + (t_{k+1} - t_k)/2]
              = X_{t_k} + 4(t - (t_k + (t_{k+1} - t_k)/2))(X_{t_{k+1}} - X_{t_k})   if t \in  (t_k + (t_{k+1} - t_k)/2, t_k + 3(t_{k+1} - t_k)/4]
              = X_{t_{k+1}}                                                         if t \in  (t_k + 3(t_{k+1} - t_k)/2, t_{k+1}]

        X^b_t = X^f_{t-"1/4"}
    Parameters
        ----------
        x: (batch, timesteps, d)

    Returns
        -------
        x_b : (batch, 4*(timesteps-1), d)
        x_f : (batch, 4*(timesteps-1), d)
    """

    x_rep = torch.repeat_interleave(x, repeats=4, dim=1)

    x_b = x_rep[:, 1:-3, :]
    x_f = x_rep[:, 2:-2, :]

    return x_b, x_f


def batch_dyadic_partition(X: torch.Tensor, dyadic_order: int) -> torch.Tensor:
    # X: (batch, times, d)

    dX = X.diff(dim=1)
    D = 2**dyadic_order

    new_shape = list(dX.shape)
    new_shape[1] *= D

    X_new = torch.empty(new_shape, dtype=X.dtype).to(X.device)

    for i in range(D):
        X_new[:, i::D] = X[:, :-1] + i*dX/D

    X_new = torch.cat((X_new, X[:, -1].unsqueeze(1)), dim=1)

    # X_new: (batch, times*(2^dyadic_order) + 1, d)
    return X_new


def batch_dyadic_recovery(XX: torch.Tensor, dyadic_order: int) -> torch.Tensor:
    # XX: (batch, times*(2^dyadic_order) + 1, d)

    # X_old: (batch, times, d)
    return XX[:, ::2**dyadic_order]


def batches_to_path(batches):
    """
    input: in batches of size [num_batches, path_length, dim]

    output: one continuous path of size [num_batches*(path_length-1), dim]
    """

    path = torch.cat([torch.ones(1, 1), torch.diff(batches, dim=1).flatten(end_dim=1).cumsum(0)+1], dim=0)

    return path


def compute_quadratic_var(x: torch.Tensor) -> torch.Tensor:
    """
    Computes batchwise the quadratic variation of tensor x.

    Parameters
        ----------
        x: (batch, timesteps, d)

    Returns
        -------
        x_q_var : (batch, timesteps, d, d)
    """

    batch, timesteps, d = x.shape

    # delta_x[n,t,k] = x[n,t+1,k] - x[n,t,k]
    # delta_x: (batch, timesteps-1, d)
    delta_x = x.diff(dim=1)

    # delta_x_2[n,t,i,j] = delta_x[n,t,i]*delta_x[n,t,j]
    # i.e. delta_x_2[n,t] = delta_x[n,t] \otimes delta_x[n,t]
    # delta_x_2: (batch, timesteps-1, d, d)
    delta_x_2 = (delta_x.unsqueeze(-1) @ delta_x.unsqueeze(-1).swapaxes(-1, -2))

    # x_var[n,t,i,j] = \sum_{s=0}^{t-1} delta_x_2[n,s,i,j]
    # x_var: (batch, timesteps, d, d)
    x_var = torch.cat((torch.zeros(batch, 1, d, d).to(x.device.type), delta_x_2.cumsum(dim=1)), dim=1)

    return x_var
