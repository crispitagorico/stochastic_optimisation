import os.path
import torch
import time
import gc
from scipy.optimize import minimize
from src.utils import augment_with_time, Hoff_transform, batch_dyadic_partition, batch_dyadic_recovery

base_path = os.getcwd()


class KernelCompute:
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

    def eta(self,
            X: torch.Tensor,
            Y: torch.Tensor,
            time_augmented=False) -> torch.Tensor:

        """
        Compute the eta tensor.
        Recall

            \eta_{x}(y|_{0,t})^k = \int_0^1 K_{s,t}(x,y) dx^k_s

        Parameters
            ----------
            X: torch.Tensor(batch_x, timesteps_x, d)
            Y: torch.Tensor(batch_y, timesteps_y, d)
            time_augmented: bool - True if the input paths are time augmented  (in dimension 0)

        Returns
            -------
            eta: (batch_x, batch_y, timesteps_y, d or d-1)

            eta[i, j, t, k] = \eta_{x_i}(y_j|_{0,t})^k
        """

        return self._compute_eta(X, Y, time_augmented)

    def eta_square(self,
                   X: torch.Tensor,
                   time_augmented=False,
                   max_batch=50) -> torch.Tensor:
        """
        Compute the eta_square tensor
        i.e. the matrix of dot products in H_K of the etas

        Parameters
            ----------
            X: torch.Tensor(batch_x, timesteps_x, d)
            time_augmented: bool  - True if the input paths are time augmented  (in dimension 0)

        Returns
            -------
            eta_square: (batch_x, batch_x)

            eta_square[i,j] = <eta_{x_i},eta_{x_j}>_{H_K}
        """

        return self._compute_eta_square_batched(X, X, sym=True,
                                                time_augmented=time_augmented,
                                                max_batch=max_batch)

    def _compute_eta(self,
                     X: torch.Tensor,
                     Y: torch.Tensor,
                     time_augmented=False) -> torch.Tensor:
        """
        Compute the eta tensor.
        Recall

            eta[i, j, t, k] = \eta_{x_i}(y_j|_{0,t})^k
                         = \int_0^1 K_{s,t}(x_i,y_j) d(x_i)^k_s

        Parameters
            ----------
            X: torch.Tensor(batch_x, timesteps_x, d)
            Y: torch.Tensor(batch_y, timesteps_y, d)
            time_augmented: bool - True if the input paths are time augmented  (in dimension 0)

        Returns
            -------
            eta: (batch_x, batch_y, timesteps_y, d or d-1)

        """

        # dx : (batch_x, timesteps_x-1, d or d-1)
        # dx[i,s,k] = x[i,s+1,k] - x[i,s,k]
        if time_augmented:
            dx = X[..., 1:].diff(dim=1)
        else:
            dx = X.diff(dim=1)

        # eta : (batch_x, batch_y, timesteps_y, d or d-1)
        # eta[i,j,t,k] = eta_{x_i}(y_j|_{[0,t]}) = \int_0^1 K(X,Y)[i,j,s,t] dx[i,s,k]
        #              = \sum_{s=0}^{timesteps_x-1} K(X,Y)[i,j,s,t,*] dx[i,*,s,*,k]
        eta = (self.compute_Gram(X, Y, sym=False)[..., :-1, :].unsqueeze(-1)*dx.unsqueeze(1).unsqueeze(3)).sum(dim=-3)

        # Memory management
        del dx
        if X.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        return eta

    def _compute_eta_Hoff(self,
                          X: torch.Tensor,
                          Y: torch.Tensor,
                          time_augmented=False) -> torch.Tensor:
        """
        Compute the eta tensor.
        Recall

            eta[i, j, t, k] = \eta_{x_i}(y_j|_{0,t})^k
                         = \int_0^1 K_{s,t}(x_i,y_j) d(x_i)^k_s

        Parameters
            ----------
            X: torch.Tensor(batch_x, timesteps_x, d)
            Y: torch.Tensor(batch_y, timesteps_y, d)
            time_augmented: bool - True if the input paths are time augmented  (in dimension 0)

        Returns
            -------
            eta: (batch_x, batch_y, timesteps_y, d or d-1)

        """

        # X_Hoff_b : (batch_x, 4*(timesteps_x-1), d)
        # X_Hoff_f : (batch_x, 4*(timesteps_x-1), d)
        X_Hoff_b, X_Hoff_f = Hoff_transform(X)
        # Y_Hoff_b : (batch_y, 4*(timesteps_y-1), d)
        Y_Hoff_b = Hoff_transform(Y)[0]

        # dx : (batch_x, 4*(timesteps_x-1)-1, d or d-1)
        # dx[i,s,k] = X_Hoff_f[i,s+1,k] - X_Hoff_f[i,s,k]
        if time_augmented:
            dx = X_Hoff_f[..., 1:].diff(dim=1)
        else:
            dx = X_Hoff_f.diff(dim=1)

        # eta_Hoff : (batch_x, batch_y, 4*(timesteps_y-1), d or d-1)
        # eta[i,j,t,k] = eta_{x_i}(y_j|_{[0,t]}) = \int_0^1 K(X,Y)[i,j,s,t] dx[i,s,k]
        #              = \sum_{s=0}^{timesteps_x-1} K(X,Y)[i,j,s,t,*] dx[i,*,s,*,k]
        eta_Hoff = (self.compute_Gram(X_Hoff_b, Y_Hoff_b, sym=False)[..., :-1, :].unsqueeze(-1)*dx.unsqueeze(1).unsqueeze(3)).sum(dim=-3)

        # eta : (batch_x, batch_y, (timesteps_y), d or d-1)
        eta = torch.zeros((X.shape[0], Y.shape[0], Y.shape[1]), device=X.device.type)
        eta[:, :, 1:, :] = eta_Hoff[:, :, 2::4, :]

        # Memory management
        del X_Hoff_b, X_Hoff_f, Y_Hoff_b, dx, eta_Hoff
        if X.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        return eta

    def _compute_eta_square_batched(self,
                                    X: torch.Tensor,
                                    Y: torch.Tensor,
                                    sym=False,
                                    time_augmented=False,
                                    max_batch=50) -> torch.Tensor:
        """
        Compute the eta_square tensor, in a batched manner.
        Recall

            eta_square[i,j] = <eta_{x_i},eta_{y_j}>_{H_k}

        Parameters
            ----------
            X: torch.Tensor(batch_x, timesteps_x, d)
            Y: torch.Tensor(batch_y, timesteps_y, d)
            sym: bool - True if X == Y
            time_augmented: bool  - True if the input paths are time augmented  (in dimension 0)
            max_batch: int - The maximum batch size

        Returns
            -------
            eta_square: (batch_x, batch_y)

        """

        batch_X, batch_Y = X.shape[0], Y.shape[0]

        if batch_X <= max_batch and batch_Y <= max_batch:
            return self._compute_eta_square(X, Y, sym, time_augmented)

        elif batch_X <= max_batch and batch_Y > max_batch:
            cutoff = int(batch_Y/2)
            Y1, Y2 = Y[:cutoff], Y[cutoff:]
            K1 = self._compute_eta_square_batched(X, Y1, sym=False, time_augmented=time_augmented, max_batch=max_batch)
            K2 = self._compute_eta_square_batched(X, Y2, sym=False, time_augmented=time_augmented, max_batch=max_batch)
            return torch.cat((K1, K2), dim=1)

        elif batch_X > max_batch and batch_Y <= max_batch:
            cutoff = int(batch_X/2)
            X1, X2 = X[:cutoff], X[cutoff:]
            K1 = self._compute_eta_square_batched(X1, Y, sym=False, time_augmented=time_augmented, max_batch=max_batch)
            K2 = self._compute_eta_square_batched(X2, Y, sym=False, time_augmented=time_augmented, max_batch=max_batch)
            return torch.cat((K1, K2), dim=0)

        cutoff_X, cutoff_Y = int(batch_X/2), int(batch_Y/2)
        X1, X2 = X[:cutoff_X], X[cutoff_X:]
        Y1, Y2 = Y[:cutoff_Y], Y[cutoff_Y:]

        K11 = self._compute_eta_square_batched(X1, Y1, sym=sym, time_augmented=time_augmented, max_batch=max_batch)

        K12 = self._compute_eta_square_batched(X1, Y2, sym=False, time_augmented=time_augmented, max_batch=max_batch)
        # If X==Y then K21 is just the "transpose" of K12
        if sym:
            K21 = K12.T
        else:
            K21 = self._compute_eta_square_batched(X2, Y1, sym=False, time_augmented=time_augmented, max_batch=max_batch)

        K22 = self._compute_eta_square_batched(X2, Y2, sym=sym, time_augmented=time_augmented, max_batch=max_batch)

        K_top = torch.cat((K11, K12), dim=1)
        K_bottom = torch.cat((K21, K22), dim=1)
        return torch.cat((K_top, K_bottom), dim=0)

    def _compute_eta_square(self,
                            X: torch.Tensor,
                            Y: torch.Tensor,
                            sym=False,
                            time_augmented=False) -> torch.Tensor:
        """
        Compute the eta_square tensor.
        Recall

            eta_square[i,j] = <eta_{x_i},eta_{y_j}>_{H_k}

        Parameters
            ----------
            X: torch.Tensor(batch_x, timesteps_x, d)
            Y: torch.Tensor(batch_y, timesteps_y, d)
            sym: bool - True if X == Y
            time_augmented: bool  - True if the input paths are time augmented  (in dimension 0)

        Returns
            -------
            eta_square: (batch_x, batch_y)

        """

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

        # eta_square: (batch_x, batch_y)
        # eta_square[i,j] = <eta_{x_i},eta_{y_j}>_{\Hs} = \int_0^1 \int_0^1 K(X,Y)[i,j,s,t] <dx[i,s],dy[j,t]>
        #                 = \sum_{s=0}^{timesteps_x-1} \sum_{t=0}^{timesteps_y-1} K(X,Y)[i,j,s,t] dxdy[i,j,s,t]
        eta_square = (self.compute_Gram(X, Y, sym=sym)[..., :-1, :-1]*dxdy).sum(dim=(-2, -1))

        # Memory management
        del dx, dy, dxdy
        if X.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        return eta_square

    def _compute_eta_square_Hoff(self,
                                 X: torch.Tensor,
                                 time_augmented=False) -> torch.Tensor:
        """
        Compute the eta_square tensor.
        Recall

            eta_square[i,j] = <eta_{x_i},eta_{x_j}>_{H_k}

        Parameters
            ----------
            X: torch.Tensor(batch_x, timesteps_x, d)
            time_augmented: bool  - True if the input paths are time augmented  (in dimension 0)

        Returns
            -------
            eta_square: (batch_x, batch_x)

        """

        # X_Hoff_b : (batch_x, 4*(timesteps_x-1), d)
        # X_Hoff_f : (batch_x, 4*(timesteps_x-1), d)
        X_Hoff_b, X_Hoff_f = Hoff_transform(X)

        # dx : (batch_x, 4*(timesteps_x-1)-1, d or d-1)
        # dx[i,s,k] = X_Hoff_f[i,s+1,k] - X_Hoff_f[i,s,k]
        if time_augmented:
            dx = X_Hoff_f[..., 1:].diff(dim=1)
        else:
            dx = X_Hoff_f.diff(dim=1)

        # dx2 : (batch_x, batch_x, 4*(timesteps_x-1)-1, 4*(timesteps_x-1)-1)
        # dx2[i,j,s,t] = <dx[i,*,s,*],dx[*,j,*,t]>_{\R^d}
        dx2 = (dx.unsqueeze(1).unsqueeze(3)*dx.unsqueeze(0).unsqueeze(2)).sum(dim=-1)

        # eta_square: (batch_x, batch_x)
        # eta_square[i,j] = <eta_{x_i},eta_{x_j}>_{\Hs} = \int_0^1 \int_0^1 K(X,X)[i,j,s,t] <dx[i,s],dx[j,t]>
        #                 = \sum_{s=0}^{timesteps_x-1} \sum_{t=0}^{timesteps_x-1} K(X,X)[i,j,s,t] dx2[i,j,s,t]
        eta_square = (self.compute_Gram(X_Hoff_b, X_Hoff_b, sym=True)[..., :-1, :-1]*dx2).sum(dim=(-2, -1)).type(torch.float64)

        # Memory management
        del X_Hoff_b, X_Hoff_f, dx, dx2
        if X.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        return eta_square


class SigKernelHedger:
    def __init__(self,
                 kernel_fn,
                 payoff_fn,
                 price_initial,
                 device,
                 time_augment=True,
                 dyadic_order=0):
        """
        Parameters
        ----------

        kernel_fn :  KernelCompute object

        payoff_fn :  torch.Tensor(batch, timesteps, d) -> torch.Tensor(batch, 1)
            This function must be batchable i.e. F((x_i)_i) = (F(x_i))_i
        """

        ## Instantiated

        # Python options
        self.device = device
        # Finance quantities
        self.payoff_fn = payoff_fn
        self.pi_0 = price_initial
        # Kernel quantities
        self.Kernel = kernel_fn
        self.time_augment = time_augment
        self.dyadic_order = dyadic_order

        ## To instantiate later

        # DataSets
        self.train_set = None
        self.train_set_dyadic = None
        self.train_set_augmented = None
        self.test_set = None
        self.test_set_augmented = None
        # Kernel Hedge quantities
        self.eta = None
        self.eta2 = None
        self.regularisation = None
        self.alpha = None
        self.position = None
        self.pnl = None

    def pre_fit(self, train_paths: torch.Tensor):
        """
        Compute the eta_square matrix of the training batch

        Parameters
        ----------
        train_paths: (batch_train, timesteps, d)
            The batched training paths

        Returns
        -------
        None
        """

        ## Some preliminaries

        # i - Make sure everything is on the same device
        if not train_paths.device.type == self.device:
            self.train_set = train_paths.to(self.device)
        else:
            self.train_set = train_paths
        # ii - Dyadic Partition
        self.train_set_dyadic = batch_dyadic_partition(self.train_set, self.dyadic_order)
        # iii - Augment with time if self.time_augment == True
        if self.time_augment:
            self.train_set_augmented = augment_with_time(self.train_set_dyadic)
        else:
            self.train_set_augmented = self.train_set_dyadic

        ## Compute eta_square matrix

        # eta_square: (batch_train, batch_train)
        self.eta2 = self.Kernel.eta_square(self.train_set_augmented,
                                           time_augmented=self.time_augment)
        # Xi: (batch_train, batch_train)
        self.Xi = self.eta2/self.eta2.shape[0]

    def fit(self, reg_type='L2', regularisation=0.0):
        """
        Calibrate the hedging strategy.
        For calibration the sample size should be as large as possible to accurately approximate the empirical measure.
        For real data a rolling window operation could be used to artificially increase the sample size.

        Parameters
        ----------
        reg_type: str = 'RKHS' or 'L2'
            user will input which type of regularisation they want, either RKHS penalisation or L2 norm
            default is 'RKHS'

        regularisation: float > 0
            the large the regularisation, the smaller the alpha's become and more stable the strategy
            often 10**(-3) to 10**(-10) is sensible range

        Xi_precomputed: bool
            Has the SigKer Gram been already computed? Time saver

        verbose: bool
            If True prints computation time for performing operations

        Returns
        -------
        None

        """

        self.regularisation = regularisation

        # Start stopwatch if verbose is True
        start = start = time.time()

        # F: (batch, 1)
        F = self.payoff_fn(self.train_set).to(self.device)
        # Xi_final: (batch, 1)
        Xi_final = (self.Xi @ (F - self.pi_0))
        self.Xi_final = Xi_final

        ## Add regularisation
        # Penalization in RKHS
        if reg_type == 'RKHS':
            self.regulariser = self.regularisation * self.Xi
        # Penalization in L2 norm
        if reg_type == 'L2':
            self.regulariser = self.regularisation * torch.eye(self.Xi.shape[0]).to(self.device)

        ## Compute the weights
        # alpha: (batch)
        self.alpha = (torch.inverse((self.Xi @ self.Xi) + self.regulariser) @ Xi_final).squeeze(-1)

        print('Alpha Obtained: %s' % (time.time() - start))

    def pre_pnl(self, test_paths: torch.Tensor):

        ## Some preliminaries

        # i - Make sure everything is on the same device
        if not test_paths.device.type == self.device:
            self.test_set = test_paths.to(self.device)
        else:
            self.test_set = test_paths
        # ii - Dyadic Partition
        self.test_set_dyadic = batch_dyadic_partition(self.test_set, self.dyadic_order)
        # iii - Augment with time if self.time_augment == True
        if self.time_augment:
            self.test_set_augmented = augment_with_time(self.test_set_dyadic)
        else:
            self.test_set_augmented = self.test_set_dyadic

        # eta : (batch_x, batch_y, timesteps_y_dyadic, d)
        # eta[i,j,t,k] = eta_{x_i}(y_j|_{[0,t]}) = \int_0^1 K(X,Y)[i,j,s,t] dx[i,s,k]
        self.eta = self.Kernel.eta(self.train_set_augmented,
                                   self.test_set_augmented,
                                   time_augmented=self.time_augment)

    def compute_pnl(self, test_paths: torch.Tensor, eta_precomputed=True):
        """
        For a given path, we can compute the PnL with respect to the fitted strategy

        Parameters
        ----------
        test_paths: torch.Tensor(batch_y, timesteps, d)
            These are the paths to be hedged

        Returns
        -------
        None

        """

        if (self.eta is None) or (not eta_precomputed):
            self.pre_pnl(test_paths)

        ## Some preliminaries

        start = time.time()

        ## Compute position for each time t in the test path

        # position : (batch_y, timesteps_y_dyadic, d)
        # i.e. position[j,t,k] = \phi^k(y_j|_{0,t}) = (1/batch_x) * \sum_i alpha[i,*,*,*] eta[i,j,t,k]
        self.position = (self.alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3) * self.eta).mean(dim=0)
        ## Compute PnL over the whole path.

        # dy : (batch_y, timesteps_y_dyadic-1, d)
        dy = torch.diff(self.test_set_dyadic, dim=1)
        # pnl: (batch_y, timesteps_y_dyadic-1)
        # pnl[j,t] = \sum_k \int_0^t position[j,t,k] dy[j,t,k]
        self.pnl = (self.position[:, :-1] * dy).cumsum(dim=1).sum(dim=-1)

        # pnl: (batch_y, timesteps_y-1)
        self.pnl = batch_dyadic_recovery(self.pnl, self.dyadic_order)
        # position : (batch_y, timesteps_y, d)
        self.position = batch_dyadic_recovery(self.position, self.dyadic_order)

        print('Test PnL Obtained: %s' % (time.time() - start))

    def target_function(self, J_1, J_2):
        '''
        To be called after "pre_fit".

        Implements a target function to MINIMIZE of the type
        \[
            J_2(alpha, \E_{X \sim \mu}[J_1(X, < alpha, \iota^T_{\mu} \eta_X >_{L^2_{\mu}})], \mu)
        \]
        if $\mu$ is the empirical distribution of a set $\X = \{ x_1, \dots, x_N \}$ of paths then
        $alpha \in \R^N$ and we can write the target as
        \[
             J_2(alpha, \sum_{i=1}^N J_1(x_i, [\Xi@alpha]_i), \X)
        \]
        where once again $\Xi := \frac{1}{N} eta_square

        In fact
        \[
            < alpha, \iota^T_{\mu} \eta_{X_i} >_{L^2_{\mu}}
            =  < alpha, <\eta_{X_i}, \eta_{\cdot}>_{H_K} >_{L^2_{\mu}}
            =   1/N*[eta_square @ alpha]_{X_i}
            = [\Xi @ alpha]_{X_i}
        \]

        Parameters
        ----------
        J_1: Tuple(torch.Tensor(timesteps, dim), torch.Tensor(1)) ->  torch.Tensor(hidden)
            Inner function

        J_2: Tuple(torch.Tensor(batch), torch.Tensor(hidden), torch.Tensor(batch, timesteps, dim)) -> torch.Tensor(1)
            Outer function

        Return
        -------
        numpy(batch,) -> float
        '''

        def _target_function(alpha):

            # a: (batch_x, 1)
            a = torch.Tensor(alpha).unsqueeze(-1).to(self.device).type(torch.float64)
            # Xi_a: (batch_X, 1)
            Xi_a = self.Xi@a

            # Inner part
            inner_sum = 0
            for i in range(a.shape[0]):
                inner_sum += J_1(self.train_set[i], Xi_a[i])

            # Outer part
            return float(J_2(a, inner_sum, self.train_set).detach())

        return _target_function

    def quadratic_hedging(self, reg_type='L2', regularisation=0.0):
        '''
        Returns J_1 and J_2 for the quadratic hedging problem.
        Recall how we have to minimize
        \[
         \E_{X \sim \mu}[(F(S) -  \pi_0 - < \alpha, \iota_{\mu}^T \eta_X >_{L^2_{\mu}})^2]
        \]
        Hence
        \[
            J_1(x, V) = (F(x) - \pi_0 - V)^2
        \]
        and
        \[
            J_2(\alpha, z, \mu) = z
        \]

        Parameters
        ----------
        reg_type: str = 'RKHS' or 'L2'
            user will input which type of regularisation they want, either RKHS penalisation or L2 norm
            default is 'RKHS'

        regularisation: float > 0
            the large the regularisation, the smaller the alpha's become and more stable the strategy
            often 10**(-3) to 10**(-10) is sensible range


        Return
        -------
        J_1: Tuple(torch.Tensor(timesteps, dim), torch.Tensor(1)) ->  torch.Tensor(1)
            Inner function

        J_2: Tuple(torch.Tensor(batch,1), torch.Tensor(1), torch.Tensor(batch, timesteps, dim)) -> torch.Tensor(1)
            Outer function
        '''

        if reg_type == 'L2':
            J_2 = lambda a, z, m: z + 0.5*regularisation*torch.norm(a)
        if reg_type == 'RKHS':
            J_2 = lambda a, z, m: z + 0.5*regularisation*torch.norm(self.Xi@a)

        J_1 = lambda x, V: ((self.payoff_fn(x.unsqueeze(0)) - self.pi_0 - V)**2).squeeze(-1)

        return J_1, J_2

    def exponential_hedging(self, reg_type='L2', regularisation=0.0, bandwidth=0.5):
        '''
        Returns J_1 and J_2 for the quadratic hedging problem.
        Recall how we have to minimize
        \[
         \E_{X \sim \mu}[(F(S) -  \pi_0 - < \alpha, \iota_{\mu}^T \eta_X >_{L^2_{\mu}})^2]
        \]
        Hence
        \[
            J_1(x, V) = e^{bandwidth * (F(x) - \pi_0 - V)^2}
        \]
        and
        \[
            J_2(\alpha, z, \mu) = z
        \]

        Parameters
        ----------
        reg_type: str = 'RKHS' or 'L2'
            user will input which type of regularisation they want, either RKHS penalisation or L2 norm
            default is 'RKHS'

        regularisation: float > 0
            the large the regularisation, the smaller the alpha's become and more stable the strategy
            often 10**(-3) to 10**(-10) is sensible range

        Return
        -------
        J_1: Tuple(torch.Tensor(timesteps, dim), torch.Tensor(1)) ->  torch.Tensor(hidden)
            Inner function

        J_2: Tuple(torch.Tensor(batch,1), torch.Tensor(hidden), torch.Tensor(batch, timesteps, dim)) -> torch.Tensor(1)
            Outer function
        '''

        if reg_type == 'L2':
            J_2 = lambda a, z, m: z + 0.5*regularisation*torch.norm(a)
        if reg_type == 'RKHS':
            J_2 = lambda a, z, m: z + 0.5*regularisation*torch.norm(self.Xi@a)

        J_1 = lambda x, V: torch.exp(bandwidth*((self.payoff_fn(x.unsqueeze(0)) - self.pi_0 - V)**2).squeeze(-1))

        return J_1, J_2

    def fit_optimize(self, target, alpha_0=None):
        """
        Calibrate the hedging strategy.
        For calibration the sample size should be as large as possible to accurately approximate the empirical measure.
        For real data a rolling window operation could be used to artificially increase the sample size.

        Parameters
        ----------
        reg_type: str = 'RKHS' or 'L2'
            user will input which type of regularisation they want, either RKHS penalisation or L2 norm
            default is 'RKHS'

        regularisation: float > 0
            the large the regularisation, the smaller the alpha's become and more stable the strategy
            often 10**(-3) to 10**(-10) is sensible range

        K_precomputed: bool
            Has the SigKer Gram been already computed? Time saver

        verbose: bool
            If True prints computation time for performing operations

        Returns
        -------
        None

        """

        if alpha_0 is None:
            alpha_0 = torch.randn((self.Xi.shape[0],)).numpy()

        res = minimize(target, alpha_0)

        self.alpha = torch.tensor(res.x).to(self.device)

    def fit_optimize_torch(self, J1, J2, alpha_0=None, EPOCH=100, learning_rate=0.1):
        """
        Calibrate the hedging strategy.
        For calibration the sample size should be as large as possible to accurately approximate the empirical measure.
        For real data a rolling window operation could be used to artificially increase the sample size.

        Parameters
        ----------
        reg_type: str = 'RKHS' or 'L2'
            user will input which type of regularisation they want, either RKHS penalisation or L2 norm
            default is 'RKHS'

        regularisation: float > 0
            the large the regularisation, the smaller the alpha's become and more stable the strategy
            often 10**(-3) to 10**(-10) is sensible range

        K_precomputed: bool
            Has the SigKer Gram been already computed? Time saver

        verbose: bool
            If True prints computation time for performing operations

        Returns
        -------
        None

        """

        class Loss(torch.nn.Module):
            def __init__(self, alpha_0, X, Xi, J1, J2) -> None:
                super().__init__()

                self.Xi = Xi
                self.X = X

                if alpha_0 is None:
                    alpha_0 = torch.randn((self.Xi.shape[0],)).type(torch.float64)

                self.alpha = torch.nn.Parameter(alpha_0)
                self.J1 = J1
                self.J2 = J2

            def forward(self):

                Xi_alpha = self.Xi@self.alpha
                # Inner part
                inner_sum = 0
                for i in range(self.alpha.shape[0]):
                    inner_sum += self.J1(self.X[i], Xi_alpha[i])
                # Outer part
                return self.J2(self.alpha, inner_sum, self.X)

        loss = Loss(alpha_0, self.train_set, self.Xi, J1, J2)
        # opt = torch.optim.SGD(loss.parameters(), lr=learning_rate)
        opt = torch.optim.Adam(loss.parameters())

        for e in range(EPOCH):
            opt.zero_grad()
            loss_current = loss.forward()
            loss_current.backward()
            opt.step()

        return loss.alpha


class SigKernelTrader:
    def __init__(self,
                 kernel_fn,
                 risk_aversion,
                 device,
                 time_augment=True,
                 dyadic_order=0):
        """
        Parameters
        ----------

        kernel_fn :  KernelCompute object

        payoff_fn :  torch.Tensor(batch, timesteps, d) -> torch.Tensor(batch, 1)
            This function must be batchable i.e. F((x_i)_i) = (F(x_i))_i
        """

        ## Instantiated

        # Python options
        self.device = device
        # Finance Quantities
        self.risk_aversion = risk_aversion
        # Kernel quantities
        self.Kernel = kernel_fn
        self.time_augment = time_augment
        self.dyadic_order = dyadic_order

        ## To instantiate later

        # DataSets
        self.train_set = None
        self.train_set_dyadic = None
        self.train_set_augmented = None
        self.test_set = None
        self.test_set_augmented = None
        # Kernel Hedge quantities
        self.eta = None
        self.eta2 = None
        self.regularisation = None
        self.alpha = None
        self.position = None
        self.pnl = None

    def pre_fit(self, train_paths: torch.Tensor):
        """
        Compute the eta_square matrix of the training batch

        Parameters
        ----------
        train_paths: (batch_train, timesteps, d)
            The batched training paths

        Returns
        -------
        None
        """

        ## Some preliminaries

        # i - Make sure everything is on the same device
        if not train_paths.device.type == self.device:
            self.train_set = train_paths.to(self.device)
        else:
            self.train_set = train_paths
        # ii - Dyadic Partition
        self.train_set_dyadic = batch_dyadic_partition(self.train_set, self.dyadic_order)
        # iii - Augment with time if self.time_augment == True
        if self.time_augment:
            self.train_set_augmented = augment_with_time(self.train_set_dyadic)
        else:
            self.train_set_augmented = self.train_set_dyadic

        ## Compute eta_square matrix

        # eta_square: (batch_train, batch_train)
        self.eta2 = self.Kernel.eta_square(self.train_set_augmented,
                                           time_augmented=self.time_augment)
        # Xi: (batch_train, batch_train)
        self.Xi = self.eta2/self.eta2.shape[0]

    def fit(self, reg_type='L2', regularisation=0.0):
        """
        Calibrate the trading strategy.
        For calibration the sample size should be as large as possible to accurately approximate the empirical measure.
        For real data a rolling window operation could be used to artificially increase the sample size.

        Parameters
        ----------

        lambda_reg: float
            The regularization corresponding to Variance

        reg_type: str = 'RKHS' or 'L2'
            user will input which type of regularisation they want, either RKHS penalisation or L2 norm
            default is 'RKHS'

        regularisation: float > 0
            the large the regularisation, the smaller the alpha's become and more stable the strategy
            often 10**(-3) to 10**(-10) is sensible range

        Xi_precomputed: bool
            Has the SigKer Gram been already computed? Time saver

        verbose: bool
            If True prints computation time for performing operations

        Returns
        -------
        None

        """

        batch = self.train_set.shape[0]
        self.regularisation = regularisation

        # Start stopwatch if verbose is True
        start = start = time.time()

        # Xi_final: (batch, 1)
        Xi_final = (self.Xi @ torch.ones((batch, 1)).to(self.device).type(torch.float64))
        self.Xi_final = Xi_final

        ## Add regularisation
        # Penalization in RKHS
        if reg_type == 'RKHS':
            self.regulariser = self.regularisation * self.Xi
        # Penalization in L2 norm
        if reg_type == 'L2':
            self.regulariser = self.regularisation * torch.eye(batch).to(self.device)

        ## Compute the weights
        # Omega: (batch, batch)
        Omega = self.Xi @ self.Xi - (Xi_final @ Xi_final.T)/batch
        # alpha: (batch)
        self.alpha = 0.5 * (torch.inverse(self.risk_aversion*Omega + self.regulariser) @ Xi_final).squeeze(-1)

        print('Alpha Obtained: %s' % (time.time() - start))

    def pre_pnl(self, test_paths: torch.Tensor):

        ## Some preliminaries

        # i - Make sure everything is on the same device
        if not test_paths.device.type == self.device:
            self.test_set = test_paths.to(self.device)
        else:
            self.test_set = test_paths
        # ii - Dyadic Partition
        self.test_set_dyadic = batch_dyadic_partition(self.test_set, self.dyadic_order)
        # iii - Augment with time if self.time_augment == True
        if self.time_augment:
            self.test_set_augmented = augment_with_time(self.test_set_dyadic)
        else:
            self.test_set_augmented = self.test_set_dyadic

        # eta : (batch_x, batch_y, timesteps_y_dyadic, d)
        # eta[i,j,t,k] = eta_{x_i}(y_j|_{[0,t]}) = \int_0^1 K(X,Y)[i,j,s,t] dx[i,s,k]
        self.eta = self.Kernel.eta(self.train_set_augmented,
                                   self.test_set_augmented,
                                   time_augmented=self.time_augment)

    def compute_pnl(self, test_paths: torch.Tensor, eta_precomputed=True):
        """
        For a given path, we can compute the PnL with respect to the fitted strategy

        Parameters
        ----------
        test_paths: torch.Tensor(batch_y, timesteps, d)
            These are the paths to be hedged

        Returns
        -------
        None

        """

        if (self.eta is None) or (not eta_precomputed):
            self.pre_pnl(test_paths)

        ## Some preliminaries

        start = time.time()

        ## Compute position for each time t in the test path

        # position : (batch_y, timesteps_y_dyadic, d)
        # i.e. position[j,t,k] = \phi^k(y_j|_{0,t}) = (1/batch_x) * \sum_i alpha[i,*,*,*] eta[i,j,t,k]
        self.position = (self.alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3) * self.eta).mean(dim=0)
        ## Compute PnL over the whole path.

        # dy : (batch_y, timesteps_y_dyadic-1, d)
        dy = torch.diff(self.test_set_dyadic, dim=1)
        # pnl: (batch_y, timesteps_y_dyadic-1)
        # pnl[j,t] = \sum_k \int_0^t position[j,t,k] dy[j,t,k]
        self.pnl = (self.position[:, :-1] * dy).cumsum(dim=1).sum(dim=-1)

        # pnl: (batch_y, timesteps_y-1)
        self.pnl = batch_dyadic_recovery(self.pnl, self.dyadic_order)
        # position : (batch_y, timesteps_y, d)
        self.position = batch_dyadic_recovery(self.position, self.dyadic_order)

        print('Test PnL Obtained: %s' % (time.time() - start))
