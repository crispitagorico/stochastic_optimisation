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
                     features_X: torch.Tensor,
                     features_Y: torch.Tensor,
                     sym=False) -> torch.Tensor:
        """
        Compute the Gram Matrix

        Parameters
            ----------
            features_X: torch.Tensor(batch_x, timesteps, d_psi)
            features_Y: torch.Tensor(batch_y, timesteps, d_psi)
            sym: bool

        Returns
            -------
            K : (batch_x, batch_y, timesteps, timesteps, d_xi, d_xi)
        """
        return NotImplementedError

    def Phi(self,
            xi_X: torch.Tensor,
            features_X: torch.Tensor,
            features_Y: torch.Tensor) -> torch.Tensor:

        """
        Computes the evaluation of Phi_X on features_Y in the kappa RKHS sense
        Recall

            < Phi_{x}(psi(y)|_{[0,t]}), z >_{xi} =  < Phi_{x}, kappa(psi(y)|_{[0,t]}, cdot)z >_{kappa}
                                                   =  int_0^1 < kappa(psi(y)|_{[0,t]},psi(x)|_{[0,s]})z , dxi(x)_s >_{xi}

            Phi_{x}(psi(y)|_{[0,t]}) = int_0^1 kappa(psi(x)|_{[0,s]},psi(y)|_{[0,t]}) dxi(x)_s

        Parameters
            ----------
            xi_X: torch.Tensor(batch_x, timesteps, d_xi)
            features_X: torch.Tensor(batch_x, timesteps, d_psi)
            features_Y: torch.Tensor(batch_y, timesteps, d_psi)

        Returns
            -------
            Phi: (batch_x, batch_y, timesteps, d_xi)

            Phi[i, j, t, k] = [ Phi_{x}(psi(y)|_{[0,t]}) ]_k
        """

        return self._compute_Phi(xi_X, features_X, features_Y)

    def K_Phi(self,
              xi_X: torch.Tensor,
              features_X: torch.Tensor,
              max_batch=50) -> torch.Tensor:
        """
        Compute the Gram matrix of K_Phi
        i.e. the matrix of dot products in H_Phi of the Phis

        Parameters
            ----------
            xi_X: torch.Tensor(batch_x, timesteps, d_xi)
            features_X: torch.Tensor(batch_x, timesteps, d_psi)

        Returns
            -------
            K_Phi: (batch_x, batch_x)
            K_Phi[i,j] = K_Phi(x_i,x_j) = <Phi_{x_i},Phi_{x_j}>_{H_Phi}
        """

        return self._compute_K_Phi_batched(xi_X, xi_X,
                                           features_X, features_X,
                                           sym=True,
                                           max_batch=max_batch)

    def _compute_Phi(self,
                     xi_X: torch.Tensor,
                     features_X: torch.Tensor,
                     features_Y: torch.Tensor) -> torch.Tensor:
        """
        Compute the eta tensor.
        Recall

            Phi[i, j, t, k] := [ Phi_{x}(\psi(y)|_{[0,t]}) ]_k
                             = int_0^1 < \kappa(\psi(y)|_{[0,t]},psi(x)|_{[0,s]})e_k, dxi(x)_s >_{xi}

        Parameters
            ----------
            xi_X: torch.Tensor(batch_x, timesteps, d_xi)
            features_X: torch.Tensor(batch_x, timesteps, d_psi)
            features_Y: torch.Tensor(batch_y, timesteps, d_psi)

        Returns
            -------
            Phi: (batch_x, batch_y, timesteps, d_xi)

        """

        # dxi_x : (batch_x, timesteps-1, d_xi)
        # dxi_x[i,s,k] = xi_X[i,s+1,k] - xi_X[i,s,k]
        dxi_x = xi_X.diff(dim=1)

        # K[i, j, s, t, :, :] = \kappa(\psi(x_i)|_{[0,s]}, \psi(y_j)|_{[0,t]}) \in \bR^{d_xi \times d_xi}
        K = self.compute_Gram(features_X, features_Y, sym=False)

        # Phi : (batch_x, batch_y, timesteps, d_xi)
        # Phi[i,j,t,:] = \int_0^1 \kappa(\psi(x_i)|_{[0,s]}, \psi(y_j)|_{[0,t]}) d\xi(x)_s
        #              = \sum_{m=1}^{d_xi} \sum_{s=0}^{timesteps-1} K[i,j,s,t,:,m]@dxi_x[i,*,s,*,*,m]
        Phi = (K[:, :, :-1]*dxi_x.unsqueeze(1).unsqueeze(3).unsqueeze(4)).sum(dim=(-1, -4))

        # Memory management
        del dxi_x
        if xi_X.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        return Phi

    def _compute_K_Phi_batched(self,
                               xi_X: torch.Tensor,
                               xi_Y: torch.Tensor,
                               features_X: torch.Tensor,
                               features_Y: torch.Tensor,
                               sym=False, max_batch=50) -> torch.Tensor:
        """
        Compute the K_Phi Gram Matrix, in a batched manner.
        Recall

            K_Phi[i,j] = <Phi_{x_i},Phi_{y_j}>

        Parameters
            ----------
            xi_X: torch.Tensor(batch_x, timesteps, d_xi)
            xi_Y: torch.Tensor(batch_y, timesteps, d_xi)
            features_X: torch.Tensor(batch_x, timesteps, d_psi)
            features_Y: torch.Tensor(batch_y, timesteps, d_psi)
            sym: bool - True if X == Y
            max_batch: int - The maximum batch size

        Returns
            -------
            K_Phi: (batch_x, batch_y)

        """

        batch_X, batch_Y = xi_X.shape[0], xi_Y.shape[0]

        if batch_X <= max_batch and batch_Y <= max_batch:
            return self._compute_K_Phi(xi_X, xi_Y, features_X, features_Y, sym)

        elif batch_X <= max_batch and batch_Y > max_batch:
            cutoff = int(batch_Y/2)
            xi_Y1, xi_Y2 = xi_Y[:cutoff], xi_Y[cutoff:]
            features_Y1, features_Y2 = features_Y[:cutoff], features_Y[cutoff:]
            K1 = self._compute_K_Phi_batched(xi_X, xi_Y1, features_X, features_Y1, sym=False,  max_batch=max_batch)
            K2 = self._compute_K_Phi_batched(xi_X, xi_Y2, features_X, features_Y2, sym=False,  max_batch=max_batch)
            return torch.cat((K1, K2), dim=1)

        elif batch_X > max_batch and batch_Y <= max_batch:
            cutoff = int(batch_X/2)
            xi_X1, xi_X2 = xi_X[:cutoff], xi_X[cutoff:]
            features_X1, features_X2 = features_X[:cutoff], features_X[cutoff:]
            K1 = self._compute_K_Phi_batched(xi_X1, xi_Y, features_X1, features_Y, sym=False,  max_batch=max_batch)
            K2 = self._compute_K_Phi_batched(xi_X2, xi_Y, features_X2, features_Y, sym=False,  max_batch=max_batch)
            return torch.cat((K1, K2), dim=0)

        cutoff_X, cutoff_Y = int(batch_X/2), int(batch_Y/2)
        xi_X1, xi_X2 = xi_X[:cutoff_X], xi_X[cutoff_X:]
        features_X1, features_X2 = features_X[:cutoff_X], features_X[cutoff_X:]
        xi_Y1, xi_Y2 = xi_Y[:cutoff_Y], xi_Y[cutoff_Y:]
        features_Y1, features_Y2 = features_Y[:cutoff_Y], features_Y[cutoff_Y:]

        K11 = self._compute_K_Phi_batched(xi_X1, xi_Y1, features_X1, features_Y1, sym=False,  max_batch=max_batch)
        K12 = self._compute_K_Phi_batched(xi_X1, xi_Y2, features_X1, features_Y2, sym=False,  max_batch=max_batch)

        # If X==Y then K21 is just the "transpose" of K12
        if sym:
            K21 = K12.T
        else:
            K21 = self._compute_K_Phi_batched(xi_X2, xi_Y1, features_X2, features_Y1, sym=False,  max_batch=max_batch)

        K22 = self._compute_K_Phi_batched(xi_X2, xi_Y2, features_X2, features_Y2, sym=False,  max_batch=max_batch)

        K_top = torch.cat((K11, K12), dim=1)
        K_bottom = torch.cat((K21, K22), dim=1)
        return torch.cat((K_top, K_bottom), dim=0)

    def _compute_K_Phi(self,
                       xi_X: torch.Tensor,
                       xi_Y: torch.Tensor,
                       features_X: torch.Tensor,
                       features_Y: torch.Tensor,
                       sym=False) -> torch.Tensor:
        """
        Compute the K_Phi Gram Matrix.
        Recall

            K_Phi[i,j] = <Phi_{x_i},Phi_{y_j}>

        Parameters
            ----------
            xi_X: torch.Tensor(batch_x, timesteps, d_xi)
            xi_Y: torch.Tensor(batch_y, timesteps, d_xi)
            features_X: torch.Tensor(batch_x, timesteps, d_psi)
            features_Y: torch.Tensor(batch_y, timesteps, d_psi)
            sym: bool - True if X == Y

        Returns
            -------
            K_Phi: (batch_x, batch_y)

        """

        # dxi_x : (batch_x, timesteps-1, d_xi)
        # dxi_y : (batch_y, timesteps-1, d_xi)
        dxi_x = xi_X.diff(dim=1)
        dxi_y = xi_Y.diff(dim=1)

        # K[i, j, s, t, :, :] = \kappa(\psi(x_i)|_{[0,s]}, \psi(y_j)|_{[0,t]}) \in \bR^{d_xi \times d_xi}
        K = self.compute_Gram(features_X, features_Y, sym=sym)

        # K_Phi: (batch_x, batch_y)
        # K_Phi[i,j] = <Phi_{x_i},Phi_{y_j}>_{\Hs}
        #            = \int_0^1 \int_0^1 <K[i,j,s,t]@dxi_x[i,s], dxi_y[j,t]>_{\xi}
        #            = \sum_{m,k=0}^{d_xi}\sum_{s,t=0}^{timesteps-1} K[i,j,s,t,m,k]  dxi_x[i,*,s,*,*,k]  dxi_y[*,j,*,t,m,*]
        K_Phi = (K[:, :, :-1, :-1]*dxi_x.unsqueeze(1).unsqueeze(3).unsqueeze(4)*dxi_y.unsqueeze(0).unsqueeze(2).unsqueeze(5)).sum(dim=(-1, -2, -3, -4))

        # Memory management
        del dxi_x, dxi_y
        if xi_X.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        return K_Phi
