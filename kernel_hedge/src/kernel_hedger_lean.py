import os.path
import torch
from src.utils import augment_with_time

base_path = os.getcwd()


class KernelHedger:
    def __init__(self,
                 kernel,
                 payoff_fn,
                 price_initial,
                 device,
                 time_augment=True):
        """
        Parameters
        ----------

        kernel :  KernelCompute object

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
        self.Kernel = kernel
        self.time_augment = time_augment

        ## To instantiate later

        # DataSets
        self.train_set = None
        self.train_set_dyadic = None
        self.train_set_augmented = None
        self.test_set = None
        self.test_set_augmented = None
        # Kernel Hedge quantities
        self.K_Phi = None
        self.Phi = None
        self.regularisation = None
        self.alpha = None
        self.position = None
        self.pnl = None

    def pre_fit(self, train_paths: torch.Tensor):
        """
        Compute the K_phi matrix of the training batch.

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
        # iu - Augment with time if self.time_augment == True
        if self.time_augment:
            self.train_set_augmented = augment_with_time(self.train_set)
        else:
            self.train_set_augmented = self.train_set

        ## Compute eta_square matrix

        # K_phi: (batch_train, batch_train)
        self.K_Phi = self.Kernel.K_Phi(self.train_set_augmented,
                                       time_augmented=self.time_augment)
        # Xi: (batch_train, batch_train)
        self.Xi = self.K_Phi/self.K_Phi.shape[0]

    def fit(self, regularisation=1e-8):
        """
        Calibrate the hedging strategy.
        For calibration the sample size should be as large as possible to accurately approximate the empirical measure.
        For real data a rolling window operation could be used to artificially increase the sample size.

        Parameters
        ----------
        regularisation: float > 0
            the large the regularisation, the smaller the alpha's become and more stable the strategy
            often 10**(-3) to 10**(-10) is sensible range

        Returns
        -------
        None

        """

        self.regularisation = regularisation

        # F: (batch, 1)
        F = self.payoff_fn(self.train_set).to(self.device)

        ## Compute the weights
        # alpha: (batch)
        temp = torch.inverse(self.Xi/self.regularisation + 0.5*torch.eye(self.Xi.shape[0]).to(self.device))
        self.alpha = (temp @ (F - self.pi_0)).squeeze(-1)

    def pre_pnl(self, test_paths: torch.Tensor):

        ## Some preliminaries

        # i - Make sure everything is on the same device
        if not test_paths.device.type == self.device:
            self.test_set = test_paths.to(self.device)
        else:
            self.test_set = test_paths
        # ii - Augment with time if self.time_augment == True
        if self.time_augment:
            self.test_set_augmented = augment_with_time(self.test_set)
        else:
            self.test_set_augmented = self.test_set

        # Phi : (batch_x, batch_y, timesteps_y_dyadic, d)
        # Phi[i,j,t,k] = Phi_{x_i}(y_j|_{[0,t]}) = \int_0^1 K(X,Y)[i,j,s,t] dx[i,s,k]
        self.Phi = self.Kernel.Phi(self.train_set_augmented,
                                   self.test_set_augmented,
                                   time_augmented=self.time_augment)

    def compute_pnl(self, test_paths: torch.Tensor, Phi_precomputed=True):
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
        if (self.Phi is None) or (not Phi_precomputed):
            self.pre_pnl(test_paths)

        ## Compute position for each time t in the test path
        # position : (batch_y, timesteps_y_dyadic, d)
        # i.e. position[j,t,k] = \phi^k(y_j|_{0,t}) = (1/batch_x) * \sum_i alpha[i,*,*,*] eta[i,j,t,k]
        self.position = (self.alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3) * self.Phi/self.regularisation).mean(dim=0)

        ## Compute PnL over the whole path.
        # dy : (batch_y, timesteps_y_dyadic-1, d)
        dy = torch.diff(self.test_set, dim=1)
        # pnl: (batch_y, timesteps_y_dyadic-1)
        # pnl[j,t] = \sum_k \int_0^t position[j,t,k] dy[j,t,k]
        self.pnl = (self.position[:, :-1] * dy).cumsum(dim=1).sum(dim=-1)
