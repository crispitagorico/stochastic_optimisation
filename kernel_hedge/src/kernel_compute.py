import os.path
import torch
import gc

base_path = os.getcwd()


class KernelCompute:
    """
    This is a wrapper class in the sense that it adds functionality to the base Kernel
    in order to make it compatible with the RKHS framework.

    In essence this class defines the dot products of the Phi feature maps.
    """
    def __init__(self):
        pass

    def compute_Gram(self,
                     X: torch.Tensor,
                     Y: torch.Tensor,
                     sym=False) -> torch.Tensor:
        """
        Compute the Gram Matrix

        Parameters
            ----------
            X: torch.Tensor(batch_x, timesteps_x, d)
            Y: torch.Tensor(batch_y, timesteps_y, d)
            sym: bool

        Returns
            -------
            K : (batch_x, batch_y, timesteps_x, timesteps_y)
        """
        return NotImplementedError

    def Phi(self,
            X: torch.Tensor,
            Y: torch.Tensor,
            time_augmented=False) -> torch.Tensor:

        """
        Computes the evaluation of Phi_X on Y in the kappa RKHS sense
        Recall

            Phi_{x}(y|_{0,t})^k = \int_0^1 K_{s,t}(x,y) dx^k_s

        Parameters
            ----------
            X: torch.Tensor(batch_x, timesteps_x, d)
            Y: torch.Tensor(batch_y, timesteps_y, d)
            time_augmented: bool - True if the input paths are time augmented  (in dimension 0)

        Returns
            -------
            Phi: (batch_x, batch_y, timesteps_y, d or d-1)

            Phi[i, j, t, k] = Phi_{x_i}(y_j|_{0,t})^k
        """

        return self._compute_Phi(X, Y, time_augmented)

    def K_Phi(self,
              X: torch.Tensor,
              time_augmented=False,
              max_batch=50) -> torch.Tensor:
        """
        Compute the Gram matrix of K_Phi
        i.e. the matrix of dot products in H_Phi of the Phis

        Parameters
            ----------
            X: torch.Tensor(batch_x, timesteps_x, d)
            time_augmented: bool  - True if the input paths are time augmented  (in dimension 0)

        Returns
            -------
            K_Phi: (batch_x, batch_x)

            K_Phi[i,j] = K_Phi(x_i,x_j) = <Phi_{x_i},Phi_{x_j}>_{H_Phi}
        """

        return self._compute_K_Phi_batched(X, X, sym=True,
                                           time_augmented=time_augmented,
                                           max_batch=max_batch)

    def _compute_Phi(self,
                     X: torch.Tensor,
                     Y: torch.Tensor,
                     time_augmented=False) -> torch.Tensor:
        """
        Compute the eta tensor.
        Recall

            Phi[i, j, t, k] := Phi_{x_i}(y_j|_{0,t})^k
                             = \int_0^1 K_{s,t}(x_i,y_j) d(x_i)^k_s

        Parameters
            ----------
            X: torch.Tensor(batch_x, timesteps_x, d)
            Y: torch.Tensor(batch_y, timesteps_y, d)
            time_augmented: bool - True if the input paths are time augmented  (in dimension 0)

        Returns
            -------
            Phi: (batch_x, batch_y, timesteps_y, d or d-1)

        """

        # dx : (batch_x, timesteps_x-1, d or d-1)
        # dx[i,s,k] = x[i,s+1,k] - x[i,s,k]
        if time_augmented:
            dx = X[..., 1:].diff(dim=1)
        else:
            dx = X.diff(dim=1)

        # K[i, j, s, t] = K(x_i, y_j)_{s,t}
        K = self.compute_Gram(X, Y, sym=False)

        # Phi : (batch_x, batch_y, timesteps_y, d or d-1)
        # Phi[i,j,t,k] = Phi_{x_i}(y_j|_{[0,t]}) = \int_0^1 K(X,Y)[i,j,s,t] dx[i,s,k]
        #              = \sum_{s=0}^{timesteps_x-1} K(X,Y)[i,j,s,t,*] dx[i,*,s,*,k]
        Phi = (K[..., :-1, :].unsqueeze(-1)*dx.unsqueeze(1).unsqueeze(3)).sum(dim=-3)

        # Memory management
        del dx
        if X.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        return Phi

    def _compute_K_Phi_batched(self,
                               X: torch.Tensor,
                               Y: torch.Tensor,
                               sym=False,
                               time_augmented=False,
                               max_batch=50) -> torch.Tensor:
        """
        Compute the K_Phi Gram Matrix, in a batched manner.
        Recall

            K_Phi[i,j] = <Phi_{x_i},Phi_{y_j}>

        Parameters
            ----------
            X: torch.Tensor(batch_x, timesteps_x, d)
            Y: torch.Tensor(batch_y, timesteps_y, d)
            sym: bool - True if X == Y
            time_augmented: bool  - True if the input paths are time augmented  (in dimension 0)
            max_batch: int - The maximum batch size

        Returns
            -------
            K_Phi: (batch_x, batch_y)

        """

        batch_X, batch_Y = X.shape[0], Y.shape[0]

        if batch_X <= max_batch and batch_Y <= max_batch:
            return self._compute_K_Phi(X, Y, sym, time_augmented)

        elif batch_X <= max_batch and batch_Y > max_batch:
            cutoff = int(batch_Y/2)
            Y1, Y2 = Y[:cutoff], Y[cutoff:]
            K1 = self._compute_K_Phi_batched(X, Y1, sym=False, time_augmented=time_augmented, max_batch=max_batch)
            K2 = self._compute_K_Phi_batched(X, Y2, sym=False, time_augmented=time_augmented, max_batch=max_batch)
            return torch.cat((K1, K2), dim=1)

        elif batch_X > max_batch and batch_Y <= max_batch:
            cutoff = int(batch_X/2)
            X1, X2 = X[:cutoff], X[cutoff:]
            K1 = self._compute_K_Phi_batched(X1, Y, sym=False, time_augmented=time_augmented, max_batch=max_batch)
            K2 = self._compute_K_Phi_batched(X2, Y, sym=False, time_augmented=time_augmented, max_batch=max_batch)
            return torch.cat((K1, K2), dim=0)

        cutoff_X, cutoff_Y = int(batch_X/2), int(batch_Y/2)
        X1, X2 = X[:cutoff_X], X[cutoff_X:]
        Y1, Y2 = Y[:cutoff_Y], Y[cutoff_Y:]

        K11 = self._compute_K_Phi_batched(X1, Y1, sym=sym, time_augmented=time_augmented, max_batch=max_batch)

        K12 = self._compute_K_Phi_batched(X1, Y2, sym=False, time_augmented=time_augmented, max_batch=max_batch)
        # If X==Y then K21 is just the "transpose" of K12
        if sym:
            K21 = K12.T
        else:
            K21 = self._compute_K_Phi_batched(X2, Y1, sym=False, time_augmented=time_augmented, max_batch=max_batch)

        K22 = self._compute_K_Phi_batched(X2, Y2, sym=sym, time_augmented=time_augmented, max_batch=max_batch)

        K_top = torch.cat((K11, K12), dim=1)
        K_bottom = torch.cat((K21, K22), dim=1)
        return torch.cat((K_top, K_bottom), dim=0)

    def _compute_K_Phi(self,
                       X: torch.Tensor,
                       Y: torch.Tensor,
                       sym=False,
                       time_augmented=False) -> torch.Tensor:
        """
        Compute the K_Phi Gram Matrix.
        Recall

            K_Phi[i,j] = <Phi_{x_i},Phi_{y_j}>

        Parameters
            ----------
            X: torch.Tensor(batch_x, timesteps_x, d)
            Y: torch.Tensor(batch_y, timesteps_y, d)
            sym: bool - True if X == Y
            time_augmented: bool  - True if the input paths are time augmented  (in dimension 0)

        Returns
            -------
            K_Phi: (batch_x, batch_y)

        """

        # x_rev : (batch_x, timesteps_x-1, d or d-1)
        # dx : (batch_x, timesteps_x-1, d or d-1)
        # dy : (batch_y, timesteps_y-1, d or d-1)
        # dx[i,s,k] = x[i,s+1,k] - x[i,s,k]
        # dy[j,t,k] = y[j,t+1,k] - y[j,t,k]
        if time_augmented:
            dx = X[..., 1:].diff(dim=1)
            dy = Y[..., 1:].diff(dim=1)
        else:
            dx = X.diff(dim=1)
            dy = Y.diff(dim=1)

        # dxdy : (batch_x, batch_y, timesteps_x-1, timesteps_y-1)
        # dxdy[i,j,s,t] = <dx[i,*,s,*],dy[*,j,*,t]>_{\R^d}
        dxdy = (dx.unsqueeze(1).unsqueeze(3)*dy.unsqueeze(0).unsqueeze(2)).sum(dim=-1)

        # K_Phi: (batch_x, batch_y)
        # K_Phi[i,j] = <Phi_{x_i},Phi_{y_j}>_{\Hs}
        #            = \int_0^1 \int_0^1 K(X,Y)[i,j,s,t] <dx[i,s],dy[j,t]>
        #            = \sum_{s=0}^{timesteps_x-1} \sum_{t=0}^{timesteps_y-1} K(X,Y)[i,j,s,t] dxdy[i,j,s,t]
        G = self.compute_Gram(X, Y, sym=sym)
        # print(G.shape)
        K_Phi = (G[..., :-1, :-1]*dxdy).sum(dim=(-2, -1))

        # Memory management
        del dx, dy, dxdy
        if X.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        return K_Phi
