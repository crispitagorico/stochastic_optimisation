import os.path
import torch

base_path = os.getcwd()


class KernelTrader:
    def __init__(self,
                 kernel,
                 feature_fn,
                 market_fn,
                 device):
        """
        Parameters
        ----------

        kernel :  KernelCompute
        feature_fn :  torch.Tensor(batch, ***) -> torch.Tensor(batch, timesteps, d_psi)
        market_fn :  torch.Tensor(batch, ***) -> torch.Tensor(batch, timesteps, d_xi)
        """

        ## Instantiated

        # Python options
        self.device = device
        # Finance quantities
        self.risk_aversion = None
        self.market_fn = market_fn
        # Kernel quantities
        self.Kernel = kernel
        self.feature_fn = feature_fn

        ## To instantiate later

        # DataSets
        self.train_set = None
        self.market_train = None
        self.features_train = None
        self.test_set = None
        self.market_test = None
        self.features_test = None
        # Kernel Hedge quantities
        self.K_Phi = None
        self.Phi = None
        self.regularisation = None
        self.alpha = None
        self.position = None
        self.pnl = None

    def pre_fit(self, train_set: torch.Tensor):
        """
        Compute the K_phi matrix of the training batch.

        Parameters
        ----------
        train_set: (batch_train, *)
            The batched training set

        Returns
        -------
        None
        """

        ## Some preliminaries

        # i - Make sure everything is on the same device
        if not train_set.device.type == self.device:
            train_set = train_set.to(self.device)

        self.train_set = train_set
        self.market_train = self.market_fn(train_set)
        self.features_train = self.feature_fn(train_set)

        ## Compute eta_square matrix

        # K_phi: (batch_train, batch_train)
        self.K_Phi = self.Kernel.K_Phi(self.market_train, self.features_train)

        # Xi: (batch_train, batch_train)
        self.Xi = self.K_Phi/self.K_Phi.shape[0]

    def fit(self, regularisation=1e-8, risk_aversion=2):
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
        self.risk_aversion = risk_aversion
        N = self.K_Phi.shape[0]

        # one: (batch, 1)
        one = torch.ones(N).to(self.device).double()

        ## Compute the weights
        # alpha: (batch_x)
        temp = (torch.eye(N) - torch.ones(N, N)/N).to(self.device).double()
        temp = torch.inverse(self.regularisation*torch.eye(N).to(self.device) + self.risk_aversion*temp@self.Xi)
        self.alpha = (temp @ one).squeeze(-1)

    def pre_pnl(self, test_set: torch.Tensor):

        ## Some preliminaries

        # i - Make sure everything is on the same device
        if not test_set.device.type == self.device:
            test_set = test_set.to(self.device)

        self.test_set = test_set
        self.market_test = self.market_fn(test_set)
        self.features_test = self.feature_fn(test_set)

        # Phi : (batch_x, batch_y, timesteps, d_xi)
        self.Phi = self.Kernel.Phi(self.market_train, self.features_train, self.features_test)

    def compute_pnl(self, test_set: torch.Tensor, Phi_precomputed=True):
        """
        For a given path, we can compute the PnL with respect to the fitted strategy

        Parameters
        ----------
        test_paths: torch.Tensor(batch_y, ***)
            These are the paths to be hedged

        Returns
        -------
        None

        """
        if (self.Phi is None) or (not Phi_precomputed):
            self.pre_pnl(test_set)

        ## Compute position for each time t in the test path
        # position : (batch_y, timesteps, d_xi)
        # i.e. position[j,t,:] = F^*(\psi(y_j)|_{0,t}) = (1/batch_x) * \sum_i alpha[i,*,*,*] Phi[i,j,t,:]
        self.position = (self.alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3) * self.Phi).mean(dim=0)

        ## Compute PnL over the whole path.
        # dxi_y : (batch_y, timesteps-1, d_xi)
        dxi_y = torch.diff(self.market_test, dim=1)

        # pnl: (batch_y, timesteps-1)
        # pnl[j,t] = \sum_k \int_0^t position[j,t,k] dy[j,t,k]
        self.pnl = (self.position[:, :-1] * dxi_y).cumsum(dim=1).sum(dim=-1)
