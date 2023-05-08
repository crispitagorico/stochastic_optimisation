import os.path
from typing import Tuple
import torch
from tqdm import tqdm
import gc
from src.utils import Hoff_transform, compute_quadratic_var

base_path = os.getcwd()


class ItoKer:
    def __int__(self, dyadic_order=0):
        self.dyadic_order = dyadic_order

    def compatibility_check(self, x: torch.Tensor, y: torch.Tensor):
        """
        Check that x and y are compatible i.e.
             - d_x = d_y
             - timesteps_x = timesteps_y

        Parameters
            ----------
            x: (batch_x, timesteps_x, d_x)
            y: (batch_y, timesteps_y, d_y)

        Returns
            -------
            True  - If x and y pass all the preliminary compatibility checks
            False - otherwise
        """

        x_shape, y_shape = x.shape, y.shape

        assert x_shape[1] == y_shape[1], 'timesteps_x != timesteps_y'
        assert x_shape[2] == y_shape[2], 'd_x != d_y'
        assert x.device.type == y.device.type, 'x and y must be on the same device.'

    def compute_Gram(self, X: torch.Tensor, Y: torch.Tensor, sym=False, max_batch=50) -> torch.Tensor:

        batch_X, batch_Y = X.shape[0], Y.shape[0]

        if batch_X <= max_batch and batch_Y <= max_batch:
            K = self._compute_Gram(X, Y)

        elif batch_X <= max_batch and batch_Y > max_batch:
            cutoff = int(batch_Y/2)
            Y1, Y2 = Y[:cutoff], Y[cutoff:]
            K1 = self.compute_Gram(X, Y1, sym=False, max_batch=max_batch)
            K2 = self.compute_Gram(X, Y2, sym=False, max_batch=max_batch)
            K = torch.cat((K1, K2), dim=1)

        elif batch_X > max_batch and batch_Y <= max_batch:
            cutoff = int(batch_X/2)
            X1, X2 = X[:cutoff], X[cutoff:]
            K1 = self.compute_Gram(X1, Y, sym=False, max_batch=max_batch)
            K2 = self.compute_Gram(X2, Y, sym=False, max_batch=max_batch)
            K = torch.cat((K1, K2), dim=0)

        else:
            cutoff_X, cutoff_Y = int(batch_X/2), int(batch_Y/2)
            X1, X2 = X[:cutoff_X], X[cutoff_X:]
            Y1, Y2 = Y[:cutoff_Y], Y[cutoff_Y:]

            K11 = self.compute_Gram(X1, Y1, sym=sym, max_batch=max_batch)

            K12 = self.compute_Gram(X1, Y2, sym=False, max_batch=max_batch)
            # If X==Y then K21 is just the "transpose" of K12
            if sym:
                K21 = K12.swapaxes(0, 1).swapaxes(2, 3)
            else:
                K21 = self.compute_Gram(X2, Y1, sym=False, max_batch=max_batch)

            K22 = self.compute_Gram(X2, Y2, sym=sym, max_batch=max_batch)

            K_top = torch.cat((K11, K12), dim=1)
            K_bottom = torch.cat((K21, K22), dim=1)
            K = torch.cat((K_top, K_bottom), dim=0)
        return K

    def compute_Gram_and_eta(self, X: torch.Tensor, Y: torch.Tensor,
                             time_augment=False, max_batch=50) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_X, batch_Y = X.shape[0], Y.shape[0]

        if batch_X <= max_batch and batch_Y <= max_batch:
            K, eta = self._compute_Gram_and_eta(X, Y, time_augment)

        elif batch_X <= max_batch and batch_Y > max_batch:
            cutoff = int(batch_Y/2)
            Y1, Y2 = Y[:cutoff], Y[cutoff:]
            K1, eta1 = self.compute_Gram_and_eta(X, Y1, time_augment, max_batch)
            K2, eta2 = self.compute_Gram_and_eta(X, Y2, time_augment, max_batch)
            K = torch.cat((K1, K2), dim=1)
            eta = torch.cat((eta1, eta2), dim=1)

        elif batch_X > max_batch and batch_Y <= max_batch:
            cutoff = int(batch_X/2)
            X1, X2 = X[:cutoff], X[cutoff:]
            K1, eta1 = self.compute_Gram_and_eta(X1, Y, time_augment, max_batch)
            K2, eta2 = self.compute_Gram_and_eta(X2, Y, time_augment, max_batch)
            K = torch.cat((K1, K2), dim=0)
            eta = torch.cat((eta1, eta2), dim=0)

        else:
            cutoff_X, cutoff_Y = int(batch_X/2), int(batch_Y/2)
            X1, X2 = X[:cutoff_X], X[cutoff_X:]
            Y1, Y2 = Y[:cutoff_Y], Y[cutoff_Y:]

            K11, eta11 = self.compute_Gram_and_eta(X1, Y1, max_batch)
            K12, eta12 = self.compute_Gram_and_eta(X1, Y2, max_batch)
            K21, eta21 = self.compute_Gram_and_eta(X2, Y1, max_batch)
            K22, eta22 = self.compute_Gram_and_eta(X2, Y2, max_batch)

            K_top = torch.cat((K11, K12), dim=1)
            eta_top = torch.cat((eta11, eta12), dim=1)

            K_bottom = torch.cat((K21, K22), dim=1)
            eta_bottom = torch.cat((eta21, eta22), dim=1)

            K = torch.cat((K_top, K_bottom), dim=0)
            eta = torch.cat((eta_top, eta_bottom), dim=0)

        return K, eta

    def compute_Gram_and_eta_square(self, X: torch.Tensor, Y: torch.Tensor,
                                    sym=False, time_augment=False, max_batch=50) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_X, batch_Y = X.shape[0], Y.shape[0]

        if batch_X <= max_batch and batch_Y <= max_batch:
            K, eta = self._compute_Gram_and_eta_square(X, Y, time_augment)

        elif batch_X <= max_batch and batch_Y > max_batch:
            cutoff = int(batch_Y/2)
            Y1, Y2 = Y[:cutoff], Y[cutoff:]
            K1, eta1 = self.compute_Gram_and_eta_square(X, Y1, False, time_augment, max_batch)
            K2, eta2 = self.compute_Gram_and_eta_square(X, Y2, False, time_augment, max_batch)
            K = torch.cat((K1, K2), dim=1)
            eta = torch.cat((eta1, eta2), dim=1)

        elif batch_X > max_batch and batch_Y <= max_batch:
            cutoff = int(batch_X/2)
            X1, X2 = X[:cutoff], X[cutoff:]
            K1, eta1 = self.compute_Gram_and_eta_square(X1, Y, False, time_augment, max_batch)
            K2, eta2 = self.compute_Gram_and_eta_square(X2, Y, False, time_augment, max_batch)
            K = torch.cat((K1, K2), dim=0)
            eta = torch.cat((eta1, eta2), dim=0)

        else:
            cutoff_X, cutoff_Y = int(batch_X/2), int(batch_Y/2)
            X1, X2 = X[:cutoff_X], X[cutoff_X:]
            Y1, Y2 = Y[:cutoff_Y], Y[cutoff_Y:]

            K11, eta11 = self.compute_Gram_and_eta_square(X1, Y1, sym, time_augment, max_batch)

            K12, eta12 = self.compute_Gram_and_eta_square(X1, Y2, False, time_augment, max_batch)
            # If X==Y then K21 is just the "transpose" of K12
            if sym:
                K21, eta21 = K12.swapaxes(0, 1), eta12.swapaxes(0, 1)
            else:
                K21, eta21 = self.compute_Gram_and_eta_square(X2, Y1, False, time_augment, max_batch)

            K22, eta22 = self.compute_Gram_and_eta_square(X2, Y2, sym, time_augment, max_batch)

            K_top = torch.cat((K11, K12), dim=1)
            eta_top = torch.cat((eta11, eta12), dim=1)

            K_bottom = torch.cat((K21, K22), dim=1)
            eta_bottom = torch.cat((eta21, eta22), dim=1)

            K = torch.cat((K_top, K_bottom), dim=0)
            eta = torch.cat((eta_top, eta_bottom), dim=0)

        return K, eta

    def _compute_Gram(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the ItoKer Gram Matrix using only Stratonovich Integrals

        Parameters
            ----------
            x: (batch_x, timesteps_x, d_x)
            y: (batch_y, timesteps_y, d_x)

        Returns
            -------
            K : (batch_x, batch_y, timesteps_x, timesteps_y)
        """

        self.compatibility_check(x, y)

        batch_x, timesteps, d = x.shape
        batch_y = y.shape[0]

        # d*: (batch_*, timesteps-1, d)
        dx, dy = x.diff(dim=1), y.diff(dim=1)
        # *_var: (batch_*, timesteps-1, d, d)
        x_var, y_var = compute_quadratic_var(x)[:, 1:], compute_quadratic_var(y)[:, 1:]

        ## Compute useful quantities

        # dX_T: (batch_x, timesteps-1, d, d+1)
        dXT = torch.cat((dx.unsqueeze(-1), 0.5*x_var), dim=-1).type(torch.float64)
        # dY: (batch_y, timesteps-1, d+1, d)
        dY = torch.cat((dy.unsqueeze(-2), 0.5*y_var), dim=-2).type(torch.float64)
        # dYdXT: (batch_x, batch_y, timesteps-1, timesteps-1, d+1, d+1)
        # dYdXT[i,j,s,t] = dY[j,t]*dX[i,s]^T
        dYdXT = torch.matmul(dY.unsqueeze(0).unsqueeze(2), dXT.unsqueeze(1).unsqueeze(3))
        # trace_dYdXT: (batch_x, batch_y, timesteps-1, timesteps-1)
        # trace_dYdXT[i,j,s, t] = Tr(dY[j,t]*dX[i,s]^T)
        trace_dYdXT = torch.diagonal(dYdXT, dim1=-2, dim2=-1).sum(-1)

        # Memory management
        del dx, dy, x_var, y_var
        if x.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        # dZ : (batch_x, batch_y, timesteps-1, timesteps-1, 1+2d)
        # dZ[i, j, s, t, 0] = trace_dYdXT[i,j,s, t]
        # dZ[i, j, s, t, 1:(d+1)] = 2 * dYdXT[i,j,s, t, 1:(d+1), 0]
        # dZ[i, j, s, t, 1:(d+1)] = 2 * dYdXT[i,j,s, t, 0, 1:(d+1)]^T
        dZ = torch.cat((trace_dYdXT.unsqueeze(-1),
                        2 * dYdXT[:, :, :, :, 1:(d+1), 0],
                        2 * dYdXT[:, :, :, :, 0, 1:(d+1)]), dim=-1)

        # A_* : (2d+1, d+1)
        A_1 = torch.zeros((1+2*d, 1+d), device=x.device.type).type(torch.float64)
        A_2 = torch.zeros((1+2*d, 1+d), device=x.device.type).type(torch.float64)

        A_1[0, 0], A_2[0, 0] = 0.5, 0.5
        A_1[1:(d+1), 1:], A_2[(d+1):, 1:] = torch.eye(d, device=x.device.type), torch.eye(d, device=x.device.type)

        ## Initialise KGF := [K, G, F]

        # KGF: (batch_x, batch_y, timesteps, timesteps, 1 + 2*d)
        KGF = torch.zeros((batch_x, batch_y, timesteps, timesteps, 1 + 2*d), device=x.device.type).type(torch.float64)

        KGF[:, :, 0, :, 0] = 1
        KGF[:, :, :, 0, 0] = 1

        ## Helper function

        def compute_next(sigma, tau):

            ## Compute the contribution from the "past" values
            # past: (batch_x, batch_y, 2d+1)
            past = KGF[:, :, sigma, tau + 1] + KGF[:, :, sigma + 1, tau] - KGF[:, :, sigma, tau]

            ## Compute the innovation term
            # B_0: (batch_x, batch_y, 2d+1)
            B_0 = dZ[:, :, sigma, tau]
            # B_1: (1, batch_y, 2d+1, d)
            B_1 = torch.matmul(A_1, dY[:, tau]).unsqueeze(0)
            # B_2: (batch_x, 1, 2d+1, d)
            B_2 = torch.matmul(A_2, torch.swapaxes(dXT[:, sigma], -1, -2)).unsqueeze(1)

            innovation_K = B_0 * KGF[:, :, sigma, tau, 0].unsqueeze(-1)
            innovation_G = torch.matmul(B_2, (KGF[:, :, sigma, tau+1, 1:(d+1)] - KGF[:, :, sigma, tau, 1:(d+1)]).unsqueeze(-1)).squeeze(-1)
            innovation_F = torch.matmul(B_1, (KGF[:, :, sigma+1, tau, (d+1):] - KGF[:, :, sigma, tau, (d+1):]).unsqueeze(-1)).squeeze(-1)
            innovation = innovation_K - innovation_G - innovation_F

            # Return new value of size (batch_x, batch_y, 2d+1)
            return past + innovation

        ## Compute KGF

        for s in tqdm(range(timesteps-1)):
            # Compute KGF[:, :, s + 1, t + 1] and KGF[:, :, t + 1, s + 1] when t < s
            for t in range(s):
                KGF[:, :, s + 1, t + 1] = compute_next(s, t)
                KGF[:, :, t + 1, s + 1] = compute_next(t, s)
            # Compute K[:, :, s+1, t+1] when t==s
            KGF[:, :, s + 1, s + 1] = compute_next(s, s)

        # Return only Gram matrix K : (batch_x, batch_y, timesteps, timesteps)
        G_matrix = KGF[..., 0]

        # Memory management
        del dXT, dY, dYdXT, trace_dYdXT, dZ, KGF
        if x.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        return G_matrix

    def compute_next_fast(self, KGF, A_1, A_2, dY, dXT, dZ, d, sigma, tau):
        ### Compute K_{s+1,t+1}

        ## Compute the contribution from the "past" values
        # past: (batch_x, batch_y, 2d+1)
        past = KGF[:, :, sigma, 1] + KGF[:, :, sigma+1, 0] - KGF[:, :, sigma, 0]

        ## Compute the innovation term
        # B_0: (batch_x, batch_y, 2d+1)
        B_0 = dZ(sigma, tau)
        # B_1: (1, batch_y, 2d+1, d)
        B_1 = torch.matmul(A_1, dY[:, tau]).unsqueeze(0)
        # B_2: (batch_x, 1, 2d+1, d)
        B_2 = torch.matmul(A_2, torch.swapaxes(dXT[:, sigma], -1, -2)).unsqueeze(1)

        innovation_K = B_0 * KGF[:, :, sigma, 0, 0].unsqueeze(-1)
        # innovation_G = torch.matmul(B_2, KGF[:, :, sigma, 0, 1:(d+1)].unsqueeze(-1)).squeeze(-1)
        innovation_G = torch.matmul(B_2, (KGF[:, :, sigma, 1, 1:(d+1)] - KGF[:, :, sigma, 0, 1:(d+1)]).unsqueeze(-1)).squeeze(-1)
        # innovation_F = torch.matmul(B_1, KGF[:, :, sigma, 0, (d+1):].unsqueeze(-1)).squeeze(-1)
        innovation_F = torch.matmul(B_1, (KGF[:, :, sigma+1, 0, (d+1):] - KGF[:, :, sigma, 0, (d+1):]).unsqueeze(-1)).squeeze(-1)
        innovation = innovation_K - innovation_G - innovation_F

        # Return new value of size (batch_x, batch_y, 2d+1)
        return past + innovation

    def dZ_fast(self, dY, dXT, d, s, t):
        # dYdXT: (s,t) -> torch.Tensor(batch_x, batch_y, d+1, d+1)
        # dYdXT(s,t)[i,j] = dY[j,t]*dX[i,s]^T
        dYdXT = torch.matmul(dY[:, t].unsqueeze(0), dXT[:, s].unsqueeze(1))

        # trace_dYdXT: (s,t) -> torch.Tensor(batch_x, batch_y)
        # trace_dYdXT(s,t)[i,j] = Tr(dY[j,t]*dX[i,s]^T)
        trace_dYdXT = torch.diagonal(dYdXT, dim1=-2, dim2=-1).sum(-1)

        # dZ :  -> torch.Tensor(batch_x, batch_y, 1+2d)
        # dZ(s,t)[i, j, 0] = trace_dYdXT(s,t)[i,j]
        # dZ(s,t)[i, j, 1:(d+1)] = 2 * dYdXT(s,t)[i, j, 1:(d+1), 0]
        # dZ(s,t)[i, j, 1:(d+1)] = 2 * dYdXT(s,t)[i, j, 0, 1:(d+1)]^T
        return torch.cat((trace_dYdXT.unsqueeze(-1),
                          2 * dYdXT[..., 1:(d+1), 0],
                          2 * dYdXT[..., 0, 1:(d+1)]), dim=-1)

    def _compute_Gram_and_eta(self, x: torch.Tensor, y: torch.Tensor,
                              time_augment=False,  compute_eta=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the last value of the ItoKer Gram Matrix using only Stratonovich Integrals.
        Moreover return the etas at the same time.

        Recall how if H := [K, G, F] then H_{s+1, t+1} = past + innovation where

            past := H_{s+1,t} + H_{s,t+1} - H_{s,t}
            innovation := K_{s,t}*dZ_{s,t} + 2*(B1_t*F_{s,t} + B2_s*G_{s,t})

        Recall also how

            eta_x(y|_{0,t}}) := \int_0^1 K_{s,t}(x,y) dx_s

        so by proceeding in a row by row pattern
        i.e. K_{*,0} -> K_{*,1} -> ... -> K_{*,times_y}
        one needs only to store the current and past rows,
        thus the memory load is O(timesteps) instead of O(timesteps^2)!

        eta2[i,j] = <eta_{x_i},eta_{x_j}>_{\Hs} = \int_0^1\int_0^1 K[i,j,s,t](x,x) <dx[i,s],dx[j,t]>

        Parameters
            ----------
            x: (batch_x, timesteps_x, d_x)
            y: (batch_y, timesteps_y, d_x)
            compute_eta: Bool

        Returns
            -------
            K : (batch_x, batch_y)
            eta: (batch_x, batch_y, timesteps_y)
        """

        self.compatibility_check(x, y)

        batch_x, timesteps, d = x.shape
        batch_y = y.shape[0]

        # d*: (batch_*, timesteps-1, d)
        dx, dy = x.diff(dim=1), y.diff(dim=1)
        # *_var: (batch_*, timesteps-1, d, d)
        x_var, y_var = compute_quadratic_var(x)[:, 1:], compute_quadratic_var(y)[:, 1:]

        ## Compute useful quantities

        # dX_T: (batch_x, timesteps-1, d, d+1)
        dXT = torch.cat((dx.unsqueeze(-1), 0.5*x_var), dim=-1).type(torch.float64)
        # dY: (batch_y, timesteps-1, d+1, d)
        dY = torch.cat((dy.unsqueeze(-2), 0.5*y_var), dim=-2).type(torch.float64)

        # A_* : (2d+1, d+1)
        A_1 = torch.zeros((1+2*d, 1+d), device=x.device.type).type(torch.float64)
        A_2 = torch.zeros((1+2*d, 1+d), device=x.device.type).type(torch.float64)

        A_1[0, 0], A_2[0, 0] = 0.5, 0.5
        A_1[1:(d+1), 1:], A_2[(d+1):, 1:] = torch.eye(d, device=x.device.type), torch.eye(d, device=x.device.type)

        ## Initialise KGF := [K, G, F] in the first two rows

        # KGF_curr_past: (batch_x, batch_y, timesteps_x, 2, 1 + 2*d)
        KGF = torch.zeros((batch_x, batch_y, timesteps, 2, 1 + 2*d), device=x.device.type).type(torch.float64)

        # K_{*,0}(x,y) = K_{0,1}(x,y) = 1
        KGF[:, :, 0, :, 0] = 1
        KGF[:, :, :, 0, 0] = 1

        ## Initialise eta
        if compute_eta:
            # eta: (batch_x, batch_y, times_y, d or d-1)
            if time_augment:
                eta = torch.zeros((batch_x, batch_y, timesteps, d-1), device=x.device.type).type(torch.float64)
                eta[..., 0, :] = (x[:, -1, 1:] - x[:, 0, 1:]).unsqueeze(1)
            else:
                eta = torch.zeros((batch_x, batch_y, timesteps, d), device=x.device.type).type(torch.float64)
                eta[..., 0, :] = (x[:, -1] - x[:, 0]).unsqueeze(1)

        else:
            eta = None

        ## Helper functions
        def dZ(s, t):
            return self.dZ_fast(dY, dXT, d, s, t)

        def compute_next(sigma, tau):
            return self.compute_next_fast(KGF, A_1, A_2, dY, dXT, dZ, d, sigma, tau)

        ## Compute KGF

        for t in tqdm(range(timesteps-1)):
            for s in range(timesteps-1):
                # Compute KGF[:, :, s + 1, t + 1]
                KGF[:, :, s + 1, 1] = compute_next(s, t)

            if compute_eta:
                # Compute eta_{x}(y|_{0,t+1})
                # eta : (batch_x, batch_y, timesteps_y-1, d or d-1)
                # eta[i,j,t,k] = eta_{x_i}(y_j|_{[0,t]}) = \int_0^1 K(X,Y_t)[i,j,s] dx[i,s,k]
                #              = \sum_{s=0}^{timesteps_x-1} K(X,Y_t)[i,j,s,*] dx[i,*,s,k]
                if time_augment:
                    eta[:, :, t+1] = (KGF[:, :, :-1, 1, 0].unsqueeze(-1)*dx[..., 1:].unsqueeze(1)).sum(dim=2)
                else:
                    eta[:, :, t+1] = (KGF[:, :, :-1, 1, 0].unsqueeze(-1)*dx.unsqueeze(1)).sum(dim=2)

            # Reset KGF
            KGF[..., 0, :] = KGF[..., 1, :]
            KGF[..., 1, :] = 0
            KGF[..., 0, 1, 0] = 1

        # Return only Gram matrix K_{-1,-1}: (batch_x, batch_y)
        # and eta: (batch_x, batch_y, times_y, d)
        return KGF[..., -1, 0, 0], eta

    def _compute_Gram_and_eta_Hoff(self, x: torch.Tensor, y: torch.Tensor,
                                   time_augment=False,  compute_eta=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the last value of the ItoKer Gram Matrix using only Stratonovich Integrals.
        Moreover return the etas at the same time.

        Recall how if H := [K, G, F] then H_{s+1, t+1} = past + innovation where

            past := H_{s+1,t} + H_{s,t+1} - H_{s,t}
            innovation := K_{s,t}*dZ_{s,t} + 2*(B1_t*F_{s,t} + B2_s*G_{s,t})

        Recall also how

            eta_x(y|_{0,t}}) := \int_0^1 K_{s,t}(x,y) dx_s

        so by proceeding in a row by row pattern
        i.e. K_{*,0} -> K_{*,1} -> ... -> K_{*,times_y}
        one needs only to store the current and past rows,
        thus the memory load is O(timesteps) instead of O(timesteps^2)!

        eta2[i,j] = <eta_{x_i},eta_{x_j}>_{\Hs} = \int_0^1\int_0^1 K[i,j,s,t](x,x) <dx[i,s],dx[j,t]>


        Parameters
            ----------
            x: (batch_x, timesteps_x, d_x)
            y: (batch_y, timesteps_y, d_x)
            compute_eta: Bool

        Returns
            -------
            K : (batch_x, batch_y)
            eta: (batch_x, batch_y, timesteps_y)
            eta2: (batch_x, batch_x)
        """

        self.compatibility_check(x, y)

        batch_x, timesteps, d = x.shape
        batch_y = y.shape[0]

        # X_Hoff_b : (batch_x, 4*(timesteps_x-1), d)
        # X_Hoff_f : (batch_x, 4*(timesteps_x-1), d)
        X_Hoff_b, X_Hoff_f = Hoff_transform(x)
        # Y_Hoff_b : (batch_y, 4*(timesteps_y-1), d)
        Y_Hoff_b = Hoff_transform(y)[0]

        # dx : (batch_x, 4*(timesteps_x-1)-1, d or d-1)
        # dx[i,s,k] = X_Hoff_f[i,s+1,k] - X_Hoff_f[i,s,k]
        dx_f = X_Hoff_f.diff(dim=1)

        # d*: (batch_*, 4*(timesteps_*-1)-1, d)
        dx, dy = X_Hoff_b.diff(dim=1), Y_Hoff_b.diff(dim=1)
        # *_var: (batch_*, 4*(timesteps_*-1)-1, d, d)
        x_var, y_var = compute_quadratic_var(X_Hoff_b)[:, 1:], compute_quadratic_var(Y_Hoff_b)[:, 1:]

        ## Compute useful quantities

        # dX_T: (batch_x, 4*(timesteps_x-1)-1, d, d+1)
        dXT = torch.cat((dx.unsqueeze(-1), 0.5*x_var), dim=-1).type(torch.float64)
        # dY: (batch_y, 4*(timesteps_x-1)-1, d+1, d)
        dY = torch.cat((dy.unsqueeze(-2), 0.5*y_var), dim=-2).type(torch.float64)

        # A_* : (2d+1, d+1)
        A_1 = torch.zeros((1+2*d, 1+d), device=x.device.type).type(torch.float64)
        A_2 = torch.zeros((1+2*d, 1+d), device=x.device.type).type(torch.float64)

        A_1[0, 0], A_2[0, 0] = 0.5, 0.5
        A_1[1:(d+1), 1:], A_2[(d+1):, 1:] = torch.eye(d, device=x.device.type), torch.eye(d, device=x.device.type)

        ## Initialise KGF := [K, G, F] in the first two rows

        # KGF_curr_past: (batch_x, batch_y, 4*(timesteps_x-1), 2, 1 + 2*d)
        KGF = torch.zeros((batch_x, batch_y, 4*(timesteps-1), 2, 1 + 2*d), device=x.device.type).type(torch.float64)

        # K_{*,0}(x,y) = K_{0,1}(x,y) = 1
        KGF[:, :, 0, :, 0] = 1
        KGF[:, :, :, 0, 0] = 1

        ## Initialise eta
        if compute_eta:
            # eta: (batch_x, batch_y, 4*(timesteps_x-1), d or d-1)
            if time_augment:
                eta_Hoff = torch.zeros((batch_x, batch_y,  4*(timesteps-1), d-1), device=x.device.type).type(torch.float64)
                eta_Hoff[..., 0, :] = (x[:, -1, 1:] - x[:, 0, 1:]).unsqueeze(1)
            else:
                eta_Hoff = torch.zeros((batch_x, batch_y,  4*(timesteps-1), d), device=x.device.type).type(torch.float64)
                eta_Hoff[..., 0, :] = (x[:, -1] - x[:, 0]).unsqueeze(1)
        else:
            eta_Hoff = None

        ## Helper functions
        def dZ(s, t):
            return self.dZ_fast(dY, dXT, d, s, t)

        def compute_next(sigma, tau):
            return self.compute_next_fast(KGF, A_1, A_2, dY, dXT, dZ, d, sigma, tau)

        ## Compute KGF

        for t in tqdm(range(KGF.shape[2]-1)):
            for s in range(KGF.shape[2]-1):
                # Compute KGF[:, :, s + 1, t + 1]
                KGF[:, :, s + 1, 1] = compute_next(s, t)

            if compute_eta:
                # Compute eta_{x}(y|_{0,t+1})
                # eta_Hoff : (batch_x, batch_y, 4*(timesteps_y-1)-1, d or d-1)
                # eta_Hoff[i,j,t,k] = eta_{x_i}(y_j|_{[0,t]}) = \int_0^1 K(X,Y_t)[i,j,s] dx[i,s,k]
                #              = \sum_{s=0}^{timesteps_x-1} K(X,Y_t)[i,j,s,*] dx[i,*,s,k]
                if time_augment:
                    eta_Hoff[:, :, t+1] = (KGF[:, :, :-1, 1, 0].unsqueeze(-1)*dx_f[..., 1:].unsqueeze(1)).sum(dim=2)
                else:
                    eta_Hoff[:, :, t+1] = (KGF[:, :, :-1, 1, 0].unsqueeze(-1)*dx_f.unsqueeze(1)).sum(dim=2)

            # Reset KGF
            KGF[..., 0, :] = KGF[..., 1, :]
            KGF[..., 1, :] = 0
            KGF[..., 0, 1, 0] = 1

        # eta : (batch_x, batch_y, timesteps_y, d or d-1)
        if time_augment:
            eta = torch.zeros((batch_x, batch_y,  timesteps, d-1), device=x.device.type).type(torch.float64)
        else:
            eta = torch.zeros((batch_x, batch_y,  timesteps, d), device=x.device.type).type(torch.float64)

        eta[:, :, -1, :] = eta_Hoff[:, :, -1, :]
        eta[:, :, :-1, :] = eta_Hoff[:, :, 2::4, :]

        # Return only Gram matrix K_{-1,-1}: (batch_x, batch_y)
        # and eta: (batch_x, batch_y, times_y, d)
        return KGF[..., -1, 0, 0], eta

    def _compute_Gram_and_eta_(self, x: torch.Tensor, y: torch.Tensor,
                               time_augment=False,  compute_eta=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
       For testing purposes.

        Compute the last value of the ItoKer Gram Matrix using only Stratonovich Integrals.
        Moreover return the etas at the same time.

        Recall how if H := [K, G, F] then H_{s+1, t+1} = past + innovation where

            past := H_{s+1,t} + H_{s,t+1} - H_{s,t}
            innovation := K_{s,t}*dZ_{s,t} + 2*(B1_t*F_{s,t} + B2_s*G_{s,t})

        Recall also how

            eta_x(y|_{0,t}}) := \int_0^1 K_{s,t}(x,y) dx_s

        so by proceeding in a row by row pattern
        i.e. K_{*,0} -> K_{*,1} -> ... -> K_{*,times_y}
        one needs only to store the current and past rows,
        thus the memory load is O(timesteps) instead of O(timesteps^2)!

        eta2[i,j] = <eta_{x_i},eta_{x_j}>_{\Hs} = \int_0^1\int_0^1 K[i,j,s,t](x,x) <dx[i,s],dx[j,t]>


        Parameters
            ----------
            x: (batch_x, timesteps_x, d_x)
            y: (batch_y, timesteps_y, d_x)
            compute_eta: Bool

        Returns
            -------
            K : (batch_x, batch_y)
            eta: (batch_x, batch_y, timesteps_y)
            eta2: (batch_x, batch_x)
        """

        self.compatibility_check(x, y)

        batch_x, timesteps, d = x.shape
        batch_y = y.shape[0]

        # d*: (batch_*, timesteps-1, d)
        dx, dy = x.diff(dim=1), y.diff(dim=1)
        # *_var: (batch_*, timesteps-1, d, d)
        x_var, y_var = compute_quadratic_var(x)[:, 1:], compute_quadratic_var(y)[:, 1:]

        ## Compute useful quantities

        # dX_T: (batch_x, timesteps-1, d, d+1)
        dXT = torch.cat((dx.unsqueeze(-1), 0.5*x_var), dim=-1).type(torch.float64)
        # dY: (batch_y, timesteps-1, d+1, d)
        dY = torch.cat((dy.unsqueeze(-2), 0.5*y_var), dim=-2).type(torch.float64)

        # A_* : (2d+1, d+1)
        A_1 = torch.zeros((1+2*d, 1+d), device=x.device.type).type(torch.float64)
        A_2 = torch.zeros((1+2*d, 1+d), device=x.device.type).type(torch.float64)

        A_1[0, 0], A_2[0, 0] = 0.5, 0.5
        A_1[1:(d+1), 1:], A_2[(d+1):, 1:] = torch.eye(d, device=x.device.type), torch.eye(d, device=x.device.type)

        ## Initialise KGF := [K, G, F] in the first two rows

        # KGF_curr_past: (batch_x, batch_y, timesteps_x, 2, 1 + 2*d)
        KGF = torch.zeros((batch_x, batch_y, timesteps, 2, 1 + 2*d), device=x.device.type).type(torch.float64)
        KGF_ = torch.zeros((batch_x, batch_y, timesteps, timesteps, 1 + 2*d), device=x.device.type).type(torch.float64)

        # K_{*,0}(x,y) = K_{0,1}(x,y) = 1
        KGF[:, :, 0, :, 0] = 1
        KGF[:, :, :, 0, 0] = 1

        KGF_[:, :, 0, :, 0] = 1
        KGF_[:, :, :, 0, 0] = 1

        ## Initialise eta
        if compute_eta:
            # eta: (batch_x, batch_y, times_y, d or d-1)
            if time_augment:
                eta = torch.zeros((batch_x, batch_y, timesteps, d-1), device=x.device.type).type(torch.float64)
                eta[..., 0, :] = (x[:, -1, 1:] - x[:, 0, 1:]).unsqueeze(1)
            else:
                eta = torch.zeros((batch_x, batch_y, timesteps, d), device=x.device.type).type(torch.float64)
                eta[..., 0, :] = (x[:, -1] - x[:, 0]).unsqueeze(1)

        else:
            eta = None

        ## Helper functions
        def dZ(s, t):

            # dYdXT: (s,t) -> torch.Tensor(batch_x, batch_y, d+1, d+1)
            # dYdXT(s,t)[i,j] = dY[j,t]*dX[i,s]^T
            dYdXT = torch.matmul(dY[:, t].unsqueeze(0), dXT[:, s].unsqueeze(1))

            # trace_dYdXT: (s,t) -> torch.Tensor(batch_x, batch_y)
            # trace_dYdXT(s,t)[i,j] = Tr(dY[j,t]*dX[i,s]^T)
            trace_dYdXT = torch.diagonal(dYdXT, dim1=-2, dim2=-1).sum(-1)

            # dZ :  -> torch.Tensor(batch_x, batch_y, 1+2d)
            # dZ(s,t)[i, j, 0] = trace_dYdXT(s,t)[i,j]
            # dZ(s,t)[i, j, 1:(d+1)] = 2 * dYdXT(s,t)[i, j, 1:(d+1), 0]
            # dZ(s,t)[i, j, 1:(d+1)] = 2 * dYdXT(s,t)[i, j, 0, 1:(d+1)]^T
            return torch.cat((trace_dYdXT.unsqueeze(-1),
                              2 * dYdXT[..., 1:(d+1), 0],
                              2 * dYdXT[..., 0, 1:(d+1)]), dim=-1)

        def compute_next(sigma, tau):
            ### Compute K_{s+1,t+1}

            ## Compute the contribution from the "past" values
            # past: (batch_x, batch_y, 2d+1)
            past = KGF[:, :, sigma, 1] + KGF[:, :, sigma+1, 0] - KGF[:, :, sigma, 0]

            ## Compute the innovation term
            # B_0: (batch_x, batch_y, 2d+1)
            B_0 = dZ(sigma, tau)
            # B_1: (1, batch_y, 2d+1, d)
            B_1 = torch.matmul(A_1, dY[:, tau]).unsqueeze(0)
            # B_2: (batch_x, 1, 2d+1, d)
            B_2 = torch.matmul(A_2, torch.swapaxes(dXT[:, sigma], -1, -2)).unsqueeze(1)

            innovation_K = B_0 * KGF[:, :, sigma, 0, 0].unsqueeze(-1)
            # innovation_G = torch.matmul(B_2, KGF[:, :, sigma, 0, 1:(d+1)].unsqueeze(-1)).squeeze(-1)
            innovation_G = torch.matmul(B_2, (KGF[:, :, sigma, 1, 1:(d+1)] - KGF[:, :, sigma, 0, 1:(d+1)]).unsqueeze(-1)).squeeze(-1)
            # innovation_F = torch.matmul(B_1, KGF[:, :, sigma, 0, (d+1):].unsqueeze(-1)).squeeze(-1)
            innovation_F = torch.matmul(B_1, (KGF[:, :, sigma+1, 0, (d+1):] - KGF[:, :, sigma, 0, (d+1):]).unsqueeze(-1)).squeeze(-1)
            innovation = innovation_K - innovation_G - innovation_F

            # Return new value of size (batch_x, batch_y, 2d+1)
            return past + innovation

        ## Compute KGF

        for t in tqdm(range(timesteps-1)):
            for s in range(timesteps-1):
                # Compute KGF[:, :, s + 1, t + 1]
                KGF[:, :, s + 1, 1] = compute_next(s, t)

            if compute_eta:
                # Compute eta_{x}(y|_{0,t+1})
                # eta : (batch_x, batch_y, timesteps_y-1, d or d-1)
                # eta[i,j,t,k] = eta_{x_i}(y_j|_{[0,t]}) = \int_0^1 K(X,Y_t)[i,j,s] dx[i,s,k]
                #              = \sum_{s=0}^{timesteps_x-1} K(X,Y_t)[i,j,s,*] dx[i,*,s,k]
                if time_augment:
                    eta[:, :, t+1] = (KGF[:, :, :-1, 1, 0].unsqueeze(-1)*dx[..., 1:].unsqueeze(1)).sum(dim=2)
                else:
                    eta[:, :, t+1] = (KGF[:, :, :-1, 1, 0].unsqueeze(-1)*dx.unsqueeze(1)).sum(dim=2)

            # Reset KGF
            KGF[..., 0, :] = KGF[..., 1, :]
            KGF_[..., t+1, :] = KGF[..., 1, :]
            KGF[..., 1, :] = 0
            KGF[..., 0, 1, 0] = 1

        # Return only Gram matrix K_{-1,-1}: (batch_x, batch_y)
        # and eta: (batch_x, batch_y, times_y, d)
        return KGF[..., -1, 0, 0], eta, KGF_

    def _compute_Gram_and_eta_square(self, x: torch.Tensor, y: torch.Tensor,
                                     time_augment=False,
                                     compute_eta_square=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the last value of the ItoKer Gram Matrix using only Stratonovich Integrals.
        Moreover return the etas at the same time.

        Recall how if H := [K, G, F] then H_{s+1, t+1} = past + innovation where

            past := H_{s+1,t} + H_{s,t+1} - H_{s,t}
            innovation := K_{s,t}*dZ_{s,t} + 2*(B1_t*F_{s,t} + B2_s*G_{s,t})

        Recall also how

            eta2[i,j] = <eta_{x_i},eta_{x_j}>_{\Hs} = \int_0^1\int_0^1 K[i,j,s,t] <dx[i,s],dx[j,t]>

        so by proceeding in a row by row pattern
        i.e. K_{*,0} -> K_{*,1} -> ... -> K_{*,times_y}
        one needs only to store the current and past rows,
        thus the memory load is O(timesteps) instead of O(timesteps^2)!

        Parameters
            ----------
            x: (batch_x, timesteps_x, d_x)
            time_augment: Bool
            compute_eta: Bool

        Returns
            -------
            K : (batch_x, batch_y)
            eta2: (batch_x, batch_x)
        """

        batch_x, timesteps, d = x.shape
        batch_y = y.shape[0]

        # d*: (batch_*, timesteps-1, d)
        dx = x.diff(dim=1)
        dy = y.diff(dim=1)
        # *_var: (batch_*, timesteps-1, d, d)
        x_var = compute_quadratic_var(x)[:, 1:]
        y_var = compute_quadratic_var(y)[:, 1:]

        ## Compute useful quantities

        # dX_T: (batch_x, timesteps-1, d, d+1)
        dXT = torch.cat((dx.unsqueeze(-1), 0.5*x_var), dim=-1).type(torch.float64)
        # dY: (batch_y, timesteps-1, d+1, d)
        dY = torch.cat((dy.unsqueeze(-2), 0.5*y_var), dim=-2).type(torch.float64)

        # A_* : (2d+1, d+1)
        A_1 = torch.zeros((1+2*d, 1+d), device=x.device.type).type(torch.float64)
        A_2 = torch.zeros((1+2*d, 1+d), device=x.device.type).type(torch.float64)

        A_1[0, 0], A_2[0, 0] = 0.5, 0.5
        A_1[1:(d+1), 1:], A_2[(d+1):, 1:] = torch.eye(d, device=x.device.type), torch.eye(d, device=x.device.type)

        ## Initialise KGF := [K, G, F] in the first two rows

        # KGF_curr_past: (batch_x, batch_y, timesteps_x, 2, 1 + 2*d)
        KGF = torch.zeros((batch_x, batch_y, timesteps, 2, 1 + 2*d), device=x.device.type).type(torch.float64)

        # K_{*,0}(x,y) = K_{0,1}(x,y) = 1
        KGF[:, :, :, 0, 0] = 1
        KGF[:, :, 0, 1, 0] = 1

        ## Initialise eta_square
        if compute_eta_square:
            # eta: (batch_x, batch_x)
            eta2 = torch.zeros((batch_x, batch_y), device=x.device.type).type(torch.float64)

            # dxdy: (t) -> [batch, batch, times-1]
            # dxdy(t)[i,j,s] = <dx[i,s],dy[j,t]>_{\R^d}
            if time_augment:
                dxdy = lambda t: (dx[..., 1:].unsqueeze(1)*dy[:, t, 1:].unsqueeze(0).unsqueeze(2)).sum(dim=-1)
            else:
                dxdy = lambda t: (dx.unsqueeze(1)*dy[:, t].unsqueeze(0).unsqueeze(2)).sum(dim=-1)
        else:
            eta2 = None

        ## Helper functions
        def dZ(s, t):
            return self.dZ_fast(dY, dXT, d, s, t)

        def compute_next(sigma, tau):
            return self.compute_next_fast(KGF, A_1, A_2, dY, dXT, dZ, d, sigma, tau)

        ## Compute KGF

        for t in tqdm(range(timesteps-1)):
            for s in range(timesteps-1):
                # Compute KGF[:, :, s + 1, t + 1]
                KGF[:, :, s + 1, 1] = compute_next(s, t)

            if compute_eta_square:
                # Compute eta2[i,j] = \int_0^1\int_0^1 K[i,j,s,t] <dx[i,s],dy[j,t]>
                # KGF[:, :, :-1, 1, 0]: (batch_x, batch_y, times-1)
                # dxdy(t): (batch_x, batch_y, times-1)
                eta2 += (KGF[:, :, :-1, 0, 0]*dxdy(t)).sum(dim=2)

            # Reset KGF
            KGF[..., 0, :] = KGF[..., 1, :]
            KGF[..., 1, :] = 0
            KGF[..., 0, 1, 0] = 1

        # Return only Gram matrix K_{-1,-1}: (batch_x, batch_y)
        # and eta2: (batch_x, batch_y)
        return KGF[..., -1, 0, 0], eta2

    def _compute_Gram_and_eta_square_Hoff(self, x: torch.Tensor,
                                          time_augment=False,
                                          compute_eta_square=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the last value of the ItoKer Gram Matrix using only Stratonovich Integrals.
        Moreover return the etas at the same time.

        Recall how if H := [K, G, F] then H_{s+1, t+1} = past + innovation where

            past := H_{s+1,t} + H_{s,t+1} - H_{s,t}
            innovation := K_{s,t}*dZ_{s,t} + 2*(B1_t*F_{s,t} + B2_s*G_{s,t})

        Recall also how

            eta2[i,j] = <eta_{x_i},eta_{x_j}>_{\Hs} = \int_0^1\int_0^1 K[i,j,s,t] <dx[i,s],dx[j,t]>

        so by proceeding in a row by row pattern
        i.e. K_{*,0} -> K_{*,1} -> ... -> K_{*,times_y}
        one needs only to store the current and past rows,
        thus the memory load is O(timesteps) instead of O(timesteps^2)!

        Parameters
            ----------
            x: (batch_x, timesteps_x, d_x)
            time_augment: Bool
            compute_eta: Bool

        Returns
            -------
            K : (batch_x, batch_y)
            eta2: (batch_x, batch_x)
        """

        batch_x, timesteps, d = x.shape

        # X_Hoff_b : (batch_x, 4*(timesteps_x-1), d)
        # X_Hoff_f : (batch_x, 4*(timesteps_x-1), d)
        X_Hoff_b, X_Hoff_f = Hoff_transform(x)

        # dx : (batch_x, 4*(timesteps_x-1)-1, d or d-1)
        # dx[i,s,k] = X_Hoff_f[i,s+1,k] - X_Hoff_f[i,s,k]
        dx_f = X_Hoff_f.diff(dim=1)

        # d*: (batch_*, 4*(timesteps_x-1)-1, d)
        dx = X_Hoff_b.diff(dim=1)
        # *_var: (batch_*, 4*(timesteps_x-1)-1, d, d)
        x_var = compute_quadratic_var(X_Hoff_b)[:, 1:]

        ## Compute useful quantities

        # dX_T: (batch_x, 4*(timesteps_x-1)-1, d, d+1)
        dXT = torch.cat((dx.unsqueeze(-1), 0.5*x_var), dim=-1).type(torch.float64)
        # dY: (batch_y, 4*(timesteps_x-1)-1, d+1, d)
        dY = torch.cat((dx.unsqueeze(-2), 0.5*x_var), dim=-2).type(torch.float64)

        # A_* : (2d+1, d+1)
        A_1 = torch.zeros((1+2*d, 1+d), device=x.device.type).type(torch.float64)
        A_2 = torch.zeros((1+2*d, 1+d), device=x.device.type).type(torch.float64)

        A_1[0, 0], A_2[0, 0] = 0.5, 0.5
        A_1[1:(d+1), 1:], A_2[(d+1):, 1:] = torch.eye(d, device=x.device.type), torch.eye(d, device=x.device.type)

        ## Initialise KGF := [K, G, F] in the first two rows

        # KGF_curr_past: (batch_x, batch_y, 4*(timesteps_x-1), 2, 1 + 2*d)
        KGF = torch.zeros((batch_x, batch_x, 4*(timesteps-1), 2, 1 + 2*d), device=x.device.type).type(torch.float64)

        # K_{*,0}(x,y) = K_{0,1}(x,y) = 1
        KGF[:, :, :, 0, 0] = 1
        KGF[:, :, 0, 1, 0] = 1

        ## Initialise eta_square
        if compute_eta_square:
            # eta: (batch_x, batch_x)
            eta2 = torch.zeros((batch_x, batch_x), device=x.device.type).type(torch.float64)

            # dx2: (t) -> [batch, batch, 4*(timesteps_x-1)-1]
            # dx2(t)[i,j,s] = <dx[i,s],dx[j,t]>_{\R^d}
            if time_augment:
                dx2 = lambda t: (dx_f[..., 1:].unsqueeze(1)*dx_f[:, t, 1:].unsqueeze(0).unsqueeze(2)).sum(dim=-1)
            else:
                dx2 = lambda t: (dx_f.unsqueeze(1)*dx_f[:, t].unsqueeze(0).unsqueeze(2)).sum(dim=-1)
        else:
            eta2 = None

        ## Helper functions
        def dZ(s, t):
            return self.dZ_fast(dY, dXT, d, s, t)

        def compute_next(sigma, tau):
            return self.compute_next_fast(KGF, A_1, A_2, dY, dXT, dZ, d, sigma, tau)

        ## Compute KGF

        for t in tqdm(range(KGF.shape[2]-1)):
            for s in range(KGF.shape[2]-1):
                # Compute KGF[:, :, s + 1, t + 1]
                KGF[:, :, s + 1, 1] = compute_next(s, t)

            if compute_eta_square:
                # Compute eta2[i,j] = \int_0^1\int_0^1 K[i,j,s,t] <dx[i,s],dx[j,t]>
                # KGF[:, :, :-1, 1, 0]: (batch, batch, 4*(timesteps_x-1)-1)
                # dx(t): (batch, batch, 4*(timesteps_x-1)-1)
                eta2 += (KGF[..., :-1, 0, 0]*dx2(t)).sum(dim=2)

            # Reset KGF
            KGF[..., 0, :] = KGF[..., 1, :]
            KGF[..., 1, :] = 0
            KGF[..., 0, 1, 0] = 1

        # Return only Gram matrix K_{-1,-1}: (batch_x, batch_y)
        # and eta: (batch_x, batch_y, times_y, d)
        return KGF[..., -1, 0, 0], eta2
