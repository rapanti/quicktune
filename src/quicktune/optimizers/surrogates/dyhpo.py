from copy import deepcopy
from typing import Dict, Optional, Tuple

import gpytorch
import torch

from .models import FeatureExtractor, CostPredictor
from .surrogate import Surrogate


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: Optional[torch.Tensor],
        train_y: Optional[torch.Tensor],
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)  # type: ignore


class DyHPO(Surrogate):
    """
    The DyHPO DeepGP model. This version of DyHPO also includes a Cost Predictor.
    """

    def __init__(
        self,
        configuration: Dict,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.configuration = configuration
        self.feature_extractor = self._get_feature_extractor()
        self.model, self.likelihood, self.mll = self._get_gp_likelihood_mll()
        self.cost_predictor = self._get_cost_predictor()

        # the number of initial points for which we will retrain fully from scratch
        # This is basically equal to the dimensionality of the search space + 1.
        self.initial_nr_points = 10
        # keeping track of the total hpo iterations. It will be used during the optimization
        # process to switch from fully training the model, to refining.
        self.iterations = 0
        # flag for when the optimization of the model should start from scratch.
        self.restart = True

        self.train_steps = 1000
        self.refine_steps = 50

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _reinit_(self):
        """
        Restart the surrogate model from scratch.
        """
        fe_config = self.configuration["feature_extractor"]
        self.feature_extractor = FeatureExtractor(**fe_config)
        self.model, self.likelihood, self.mll = self._get_gp_likelihood_mll()
        self.to(self.device)

    def _get_gp_likelihood_mll(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        gp = GPRegressionModel(None, None, likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
        return gp, likelihood, mll

    def _get_feature_extractor(self):
        return FeatureExtractor.init_from_config(
            self.configuration["feature_extractor"]
        )

    def _get_cost_predictor(self):
        return CostPredictor.init_from_config(self.configuration["cost_predictor"])

    def train_step(
        self,
        config: torch.Tensor,
        budget: torch.Tensor,
        curve: torch.Tensor,
        target: torch.Tensor,
        metafeat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        proj = self.feature_extractor(config, budget, curve, metafeat)

        self.model.set_train_data(proj, target, False)
        output = self.model(proj)

        loss = -self.mll(output, target)  # type: ignore
        return loss

    def train_pipeline(self, data: Dict[str, torch.Tensor]):
        """
        Trains the pipeline using the provided data.

        Args
        ----
        data : Dict[str, torch.Tensor]
            A dictionary containing the input data for training.
            It should contain the following keys:
                - config: The hyperparameters configurations.
                - budget: The budget values.
                - curve: The learning curves.
                - target: The target values.
                - metafeat: The metafeatures.
        """
        self.train()
        self.iterations += 1

        config = data["config"]
        if config.size(0) == 1:  # skip training if only one point is provided
            return

        optimizer = torch.optim.Adam(self.parameters(), self.configuration["lr"])

        initial_state = self._get_state()
        training_errored = False

        for key, item in data.items():
            data[key] = item.to(self.device)

        if self.restart:
            self._reinit_()
            steps = self.train_steps
            # 2 cases where the statement below is hit.
            # - We are switching from the full training phase in the beginning to refining.
            # - We are restarting because our refining diverged
            if self.initial_nr_points <= self.iterations:
                self.restart = False
        else:
            steps = self.refine_steps

        for _ in range(steps):
            try:
                loss = self.train_step(**data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(f"The following error happened while training: {e}")
                # An error has happened, trigger the restart of the optimization and restart
                # the model with default hyperparameters.
                self.restart = True
                training_errored = True
                break

        if training_errored:
            self.load_state_dict(initial_state)

    def predict_pipeline(
        self,
        train_data: Dict[str, torch.Tensor],
        test_data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts the target values for the test data.

        Args
        ----
        train_data : Dict[str, torch.Tensor]
            A dictionary containing the input data for training.
            It should contain the following
                - config: The hyperparameters configurations.
                - target: The target values.
                - budget: The budget values.
                - curve: The learning curves.
                - metafeat: The metafeatures.
        test_data : Dict[str, torch.Tensor]
            A dictionary containing the input data for testing.
            It should contain the following
                - config: The hyperparameters configurations.
                - budget: The budget values.
                - curve: The learning curves.
                - target: The target values.
                - metafeat: The metafeatures.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing the predicted means, the predicted standard deviations
            and the predicted costs.
        """
        self.eval()

        for key, item in train_data.items():
            train_data[key] = item.to(self.device)
        for key, item in test_data.items():
            test_data[key] = item.to(self.device)

        target = train_data.pop("target")
        test_data.pop("target", None)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            _train = self.feature_extractor(**train_data)
            self.model.set_train_data(_train, target, False)

            _test = self.feature_extractor(**test_data)
            pred = self.likelihood(self.model(_test))

            cost = self.cost_predictor(**test_data)

        mean = pred.mean.reshape(-1)
        std = pred.stddev.reshape(-1)

        return mean, std, cost

    def _get_state(self):
        state = deepcopy(self.state_dict())
        return state
