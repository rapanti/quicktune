from copy import deepcopy
from typing import Dict, Optional, List, Tuple

import gpytorch
import torch
import torch.nn as nn

from .surrogate import Surrogate


class ConvNet(nn.Module):
    def __init__(self, in_channels: int, output_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 8, 3, 1, padding="same")
        self.conv2 = nn.Conv1d(8, 8, 3, 1, padding="same")
        self.fc1 = nn.Linear(200, output_dim)
        self.dropout1 = nn.Dropout1d(0.25)
        self.act = nn.ReLU(inplace=True)

    @classmethod
    def init_from_config(cls, config: dict) -> "ConvNet":
        return cls(**config)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = nn.MaxPool1d(2)(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.act(x)
        x = self.fc1(x)
        x = self.act(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: list[int], out_features: int):
        super().__init__()
        self.input_size = in_features
        self.hidden_sizes = hidden_features
        self.output_size = out_features

        self.input_layer = nn.Linear(in_features, hidden_features[0])
        self.output_layer = nn.Linear(hidden_features[-1], out_features)
        self.act = nn.ReLU(inplace=True)

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_features) - 1):
            hidden_layer = nn.Linear(hidden_features[i], hidden_features[i + 1])
            self.hidden_layers.append(hidden_layer)

    @classmethod
    def init_from_config(cls, config: dict) -> "MLP":
        return cls(**config)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.act(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.act(x)
        x = self.output_layer(x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 32,
        out_features: int = 32,
        in_curve_dim: int = 1,
        out_curve_dim: int = 16,
        in_meta_features: int = 4,
        out_meta_features: int = 16,
        enc_num_layers: int = 1,
        enc_slice_ranges: Optional[List[int,]] = None,
    ):
        super().__init__()
        if enc_slice_ranges is not None:
            assert enc_slice_ranges[-1] < in_features
            _slices = [0] + enc_slice_ranges + [in_features]
            _ranges = [(_slices[i], _slices[i + 1]) for i in range(len(_slices) - 1)]
            self.enc_slice_ranges = _ranges
            self.encoder = self._build_encoder(hidden_features, enc_num_layers, _ranges)
            out_enc_features = len(self.encoder) * hidden_features
        else:
            out_enc_features = in_features
            self.encoder = None
        out_enc_features += out_curve_dim + out_meta_features + 1

        self.fc1 = nn.Linear(out_enc_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

        self.curve_embedder = ConvNet(in_curve_dim, out_curve_dim)
        self.fc_meta = nn.Linear(in_meta_features, out_meta_features)

        self.act = nn.LeakyReLU(inplace=True)

    def _build_encoder(self, hidden_feat, num_layers, ranges):
        encoder = nn.ModuleList()
        for a, b in ranges:
            encoder.append(MLP(b - a, [hidden_feat] * num_layers, hidden_feat))
        return encoder

    @classmethod
    def init_from_config(cls, config: dict) -> "FeatureExtractor":
        return cls(**config)

    def forward(self, hps, budgets, curves, metafeatures=None):
        budgets = torch.unsqueeze(budgets, dim=1)
        if curves.dim() == 2:
            curves = torch.unsqueeze(curves, dim=1)

        if self.encoder is not None:
            x = []
            for (a, b), encoder in zip(self.enc_slice_ranges, self.encoder):
                x.append(encoder(hps[:, a:b]))
            x.append(budgets)
            x = torch.cat(x, dim=1)
        else:
            x = torch.cat([hps, budgets], dim=1)

        curve_embeds = self.curve_embedder(curves)

        if metafeatures is not None:
            out = self.fc_meta(metafeatures)
            x = torch.cat([x, out], dim=1)
        x = torch.cat([x, curve_embeds], dim=1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class CostPredictor(FeatureExtractor):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        out_features = kwargs.get("out_features", 32)
        self.act = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(out_features, 1)

    @classmethod
    def init_from_config(cls, config: dict) -> "CostPredictor":
        return cls(**config)

    def forward(self, hps, budgets, curves, metafeatures=None):
        x = super().forward(hps, budgets, curves, metafeatures)
        x = self.act(x)
        x = self.fc3(x)
        return x


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
        self.gp_model, self.likelihood, self.mll = self._get_gp_likelihood_mll()
        self.cost_predictor = self._get_cost_predictor()

        # the number of initial points for which we will retrain fully from scratch
        # This is basically equal to the dimensionality of the search space + 1.
        self.initial_nr_points = 10
        # keeping track of the total hpo iterations. It will be used during the optimization
        # process to switch from fully training the model, to refining.
        self.iterations = 0
        # flag for when the optimization of the model should start from scratch.
        self.restart = True

        self.nr_epochs = 1000
        self.refine_epochs = 50

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def restart_optimization(self):
        """
        Restart the surrogate model from scratch.
        """
        fe_config = self.configuration["feature_extractor"]
        self.feature_extractor = FeatureExtractor(**fe_config)
        self.gp_model, self.likelihood, self.mll = self._get_gp_likelihood_mll()
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

    def train_step(self, x, y, budgets, curves, metafeatures=None):
        projected_x = self.feature_extractor(x, budgets, curves, metafeatures)
        self.gp_model.set_train_data(projected_x, y, strict=False)

        output = self.gp_model(projected_x)
        loss = -self.mll(output, self.gp_model.train_targets)  # type: ignore
        return loss

    def train_pipeline(self, data: Dict[str, torch.Tensor]):
        """
        Trains the pipeline using the provided data.

        Args
        ----
        data : Dict[str, torch.Tensor]
            A dictionary containing the input data for training.
            It should contain the following keys:
                - "args": Tensor containing the input arguments.
                - "budgets": Tensor containing the budgets.
                - "curves": Tensor containing the curves.
                - "targets": Tensor containing the target values.
                - "metafeatures" (optional): Tensor containing the metafeatures.
        """
        self.iterations += 1
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), self.configuration["lr"])

        x = data["args"]
        budgets = data["budgets"]
        curves = data["curves"]
        y = data["targets"]
        metafeatures = data.get("metafeatures")

        initial_state = self._get_state()
        training_errored = False

        if self.restart:
            print("Restarting the optimization of the model.")
            self.restart_optimization()
            nr_epochs = self.nr_epochs
            # 2 cases where the statement below is hit.
            # - We are switching from the full training phase in the beginning to refining.
            # - We are restarting because our refining diverged
            if self.initial_nr_points <= self.iterations:
                self.restart = False
        else:
            nr_epochs = self.refine_epochs

        for _ in range(nr_epochs):
            # skip batches with only one element
            if x.size(0) == 1:
                continue
            
            optimizer.zero_grad()

            projected_x = self.feature_extractor(x, budgets, curves, metafeatures)
            self.gp_model.set_train_data(projected_x, y, strict=False)
            output = self.gp_model(projected_x)

            try:
                # Calc Loss and Backprop
                loss = -self.mll(output, self.gp_model.train_targets)  # type: ignore
                loss.backward()
                optimizer.step()
            except Exception as training_error:
                print(f"The following error happened while training: {training_error}")
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
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            train_data: A dictionary that has the training
                examples, features, budgets and learning curves.
            test_data: Same as for the training data, but it is
                for the testing part and it does not feature labels.

        Returns:
            means, stds: The means of the predictions for the
                testing points and the standard deviations.
        """
        self.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            projected_train_x = self.feature_extractor(
                train_data["args"],
                train_data["budgets"],
                train_data["curves"],
                train_data.get("metafeatures"),
            )
            self.gp_model.set_train_data(
                inputs=projected_train_x, targets=train_data["targets"], strict=False
            )
            projected_test_x = self.feature_extractor(
                test_data["args"],
                test_data["budgets"],
                test_data["curves"],
                test_data.get("metafeatures"),
            )
            preds = self.likelihood(self.gp_model(projected_test_x))

            costs = self.cost_predictor(
                test_data["args"],
                test_data["budgets"],
                test_data["curves"],
                test_data.get("metafeatures"),
            )

        means = preds.mean.reshape(-1)
        stds = preds.stddev.reshape(-1)

        return means, stds, costs

    def _get_state(self):
        state = deepcopy(self.state_dict())
        return state
