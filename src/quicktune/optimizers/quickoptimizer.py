import json
import logging
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import norm

from .surrogates.surrogate import Surrogate
from quicktune.config_manager import ConfigManager
from quicktune.data import MetaSet
from quicktune.utils.log_utils import set_logger_verbosity
from quicktune.utils.qt_utils import QTunerResult, QTaskStatus


logger = logging.getLogger(__name__)


class QuickTuneOptimizer:
    """
    The QuickTuneOptimizer is the main element for the QuickTune optimization process.

    The QuickTuneOptimizer class provides methods for training the surrogate model,
    predicting the performances of hyperparameter configurations, suggesting the next
    hyperparameter configuration to evaluate, observing the learning curve of a hyperparameter
    configuration, and updating the information dictionary with the current HPO iteration info.

    Args
    ----
    surrogate: nn.Module
        The surrogate model to be used for the optimization process.
    config_manager: ConfigurationManager
        The configuration manager object that contains the configuration space.
    metaset: MetaSet
        The MetaSet object that contains the metafeatures of the dataset.
    num_configs: int
        The number of configurations to sample from the configuration space.
    metafeatures: Optional[torch.Tensor]
        The metafeatures of the dataset.
    max_benchmark_epochs: int, default = 50
        The maximum number of epochs to benchmark the hyperparameter configurations.
    fantasize_steps: int, default = 1
        The number of steps to fantasize the hyperparameter configurations.
    acq_func: str, default = "ei"
        The acquisition function to be used for the optimization process.
    explore_factor: float, default = 0.0
        The exploration factor for the acquisition function.
    output_path: str, default = "."
        The path to save the output files.
    """

    def __init__(
        self,
        surrogate: Surrogate,
        config_manager: ConfigManager,
        metaset: MetaSet,
        num_configs: int,
        metafeatures: Optional[torch.Tensor] = None,
        max_benchmark_epochs: int = 50,
        fantasize_steps: int = 1,
        acq_func: str = "ei",
        explore_factor: float = 0.0,
        output_path: str = ".",
        device: str = "auto",
        verbosity: int = 2,
    ):
        self.verbosity = verbosity
        set_logger_verbosity(verbosity, logger)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.cm = config_manager
        self.metafeatures = metafeatures

        self.sampled_configs = self.cm.sample_configuration(num_configs)
        self.candidate_configs = self.cm.preprocess_configurations(
            self.sampled_configs, metaset
        ).values

        self.budgets: dict[int, List[int]] = dict()
        self.scores: dict[int, List[float]] = dict()

        self.max_budget = max_benchmark_epochs
        self.fantasize_steps = fantasize_steps

        conf_individual_budget = 1
        n_initial_configs = 1
        self.init_conf_indices = random.sample(range(num_configs), n_initial_configs)
        self.init_budgets = [conf_individual_budget] * n_initial_configs

        self.surrogate = surrogate
        self.initial_random_index = 0

        self.best_value_observed = np.NINF
        self.diverged_configs = set()

        self.info_dict = dict()

        self.suggest_time_duration = 0

        self.output_path = output_path

        self.no_improvement_threshold = int(self.max_budget + 0.2 * self.max_budget)
        self.no_improvement_patience = 0
        self.converged_configs = []
        self.acq_func = acq_func
        self.explore_factor = explore_factor

    def set_metafeatures(
        self, num_samples: int, num_classes: int, image_size: int, num_channels: int
    ):
        """
        Set the metafeatures of the dataset.

        Args
        ----
        metafeatures: torch.Tensor
            The metafeatures of the dataset.
        """
        meta_scale_factor = 10000  # TODO: automate this
        t = torch.tensor(
            [num_samples, num_classes, image_size, num_channels],
            dtype=torch.float,
        ).reshape(1, -1)
        self.metafeatures = t / meta_scale_factor

    def _get_train_data(self) -> Dict[str, torch.Tensor]:
        """
        Prepare the data that will be the input to the surrogate.

        Returns
        -------
        data: Dict[str, torch.Tensor]
            The data that will be the input to the surrogate.
        """

        config, target, budget, curve = self._get_history_configs()

        config = np.array(config, dtype=np.single)
        target = np.array(target, dtype=np.single) / 100
        budget = np.array(budget, dtype=np.single)
        curve = np.array(curve, dtype=np.single) / 100

        # scale budgets to [0, 1]
        budget /= self.max_budget

        metafeat = None
        if self.metafeatures is not None:
            metafeat = self.metafeatures.repeat(len(config), 1) / 10000

        data = {
            "config": torch.tensor(config),
            "budget": torch.tensor(budget),
            "curve": torch.tensor(curve),
            "target": torch.tensor(target),
            "metafeat": metafeat,
        }

        return data

    def _fit_surrogate(self):
        """
        Train the surrogate model with the observed hyperparameter configurations.
        """
        data = self._get_train_data()
        self.surrogate.to(self.device)
        self.surrogate.train_pipeline(data)

    def _predict(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Predict the performances of the hyperparameter configurations
        as well as the standard deviations based on the surrogate model.
        Returns:
            mean_predictions, std_predictions, hp_indices, non_scaled_budgets:
                The mean predictions and the standard deviations over
                all model predictions for the given hyperparameter
                configurations with their associated indices, scaled and
                non-scaled budgets.
        """
        # config, budget, curve, indices = self._get_candidate_configs()
        config, budget, curve = self._get_candidate_configs2()

        # add fantasize steps to the budget
        _budget = torch.tensor(budget, dtype=torch.float) + self.fantasize_steps
        # scale budget to [0, 1]
        _budget /= self.max_budget
        config = torch.tensor(config, dtype=torch.float)

        curve = torch.tensor(curve, dtype=torch.float) / 100

        metafeat = None
        if self.metafeatures is not None:
            metafeat = self.metafeatures.repeat(config.size(0), 1)
            metafeat /= 10000

        test_data = {
            "config": config,
            "budget": _budget,
            "curve": curve,
            "metafeat": metafeat,
        }
        train_data = self._get_train_data()

        mean, std, cost = self.surrogate.predict_pipeline(train_data, test_data)

        mean = mean.cpu().detach().numpy()
        std = std.cpu().detach().numpy()
        cost = cost.squeeze().cpu().detach().numpy() if cost is not None else None

        return mean, std, budget, cost

    def suggest(self) -> Tuple[int, int]:
        """
        Suggest the next hyperparameter configuration to evaluate.

        Returns
        -------
        best_config_index: int
            The index of the best hyperparameter configuration.
        budget: int
            The budget of the hyperparameter configuration.
        """
        # check if we still have random configurations to evaluate
        if self.initial_random_index < len(self.init_conf_indices):
            logger.info(
                "Not enough configurations to build a model. "
                "Returning randomly sampled configuration"
            )

            index = self.init_conf_indices[self.initial_random_index]
            budget = self.init_budgets[self.initial_random_index]
            self.initial_random_index += 1

            return index, budget

        else:
            mean, std, budgets, costs = self._predict()

            best_indices = self.find_suggested_config(mean, std, budgets, costs)

            best_indices = [
                idx for idx in best_indices if idx not in self.diverged_configs
            ]
            index = best_indices[-1]

            # decide for what budget we will evaluate the most promising hyperparameter configuration next.
            if index in self.budgets:
                budget = self.budgets[index][-1]
                budget += self.fantasize_steps
                # if fantasize_step is bigger than 1
                budget = min(budget, self.max_budget)
            else:
                budget = self.fantasize_steps

        return index, budget

    def observe(
        self,
        index: int,
        budget: int,
        result: QTunerResult,
    ):
        """
        Observe the learning curve of a hyperparameter configuration.

        Args
        ----
        index: int
            The index of the hyperparameter configuration.
        budget: int
            The budget of the hyperparameter configuration.
        result:
            The performance of the hyperparameter configuration.

        Returns
        -------
        overhead_time: float
            The overhead time of the iteration.
        """
        # if y is an undefined value, append 0 as the overhead since we finish here.
        score = result.score / 100
        if result.status == QTaskStatus.ERROR:
            self.update_info_dict(index, budget, np.nan, 0)
            self.diverged_configs.add(index)
            return

        observe_time_start = time.time()

        if index in self.budgets:
            self.budgets[index].append(budget)
            self.scores[index].append(score)
        else:
            self.budgets[index] = [budget]
            self.scores[index] = [score]

        if self.best_value_observed < score:
            self.best_value_observed = score
            self.no_improvement_patience = 0
        else:
            self.no_improvement_patience += 1

        observe_time_end = time.time()
        train_time_duration = 0

        # initialization phase over. Now we can sample from the model.
        if self.initial_random_index >= len(self.init_conf_indices):
            train_time_start = time.time()
            # create the model first
            assert self.surrogate is not None

            if self.no_improvement_patience == self.no_improvement_threshold:
                self.surrogate.restart = True  # type: ignore

            self._fit_surrogate()

            train_time_end = time.time()
            train_time_duration = train_time_end - train_time_start

        observe_time_duration = observe_time_end - observe_time_start
        overhead_time = (
            observe_time_duration + self.suggest_time_duration + train_time_duration
        )
        total_duration = overhead_time
        self.update_info_dict(index, budget, score, total_duration)
        return overhead_time

    def _get_candidate_configs(
        self,
    ) -> Tuple[np.ndarray, ...]:
        """
        Generate candidate configurations that will be fantasized upon.

        Returns
        -------
        configurations: List
            The hyperparameter configurations.
        hp_indices: List
            The indices of the hyperparameter configurations.
        hp_budgets: List
            The budgets of the hyperparameter configurations.
        learning_curves: List
            The learning curves of the hyperparameter configurations.
        """
        budgets = []
        curves = []
        indices = []

        for index in range(self.candidate_configs.shape[0]):
            if index in self.converged_configs:
                continue

            if index in self.budgets:
                budget = max(self.budgets[index])
                curve = self.scores[index]
            else:  # config was not evaluated before fantasize
                budget = 0
                curve = [0.0]

            # pad the curve with zeros if it is not fully evaluated
            curve = curve + [0.0] * (self.max_budget - len(curve))

            # configuration not evaluated fully yet
            if budget < self.max_budget:
                budgets.append(budget)
                curves.append(curve)
                indices.append(index)

        configs = self.candidate_configs[indices]
        budgets = np.array(budgets)
        curves = np.array(curves)
        indices = np.array(indices)

        return configs, budgets, curves, indices

    def _get_candidate_configs2(
        self,
    ) -> Tuple[np.ndarray, ...]:
        """
        Generate candidate configurations that will be fantasized upon.

        Returns
        -------
        configurations: List
            The hyperparameter configurations.
        hp_indices: List
            The indices of the hyperparameter configurations.
        hp_budgets: List
            The budgets of the hyperparameter configurations.
        learning_curves: List
            The learning curves of the hyperparameter configurations.
        """
        budgets = []
        curves = []

        for index in range(len(self.candidate_configs)):
            if index in self.budgets:
                budget = max(self.budgets[index])
                curve = self.scores[index]
            else:  # config was not evaluated before fantasize
                budget = 0
                curve = [0.0]

            # pad the curve with zeros if it is not fully evaluated
            curve = curve + [0.0] * (self.max_budget - len(curve))

            # configuration not evaluated fully yet
            if budget < self.max_budget:
                budgets.append(budget)
                curves.append(curve)

        configs = self.candidate_configs
        budgets = np.array(budgets)
        curves = np.array(curves)

        return configs, budgets, curves

    def _get_history_configs(
        self,
    ) -> Tuple[List, List, List, List]:
        """
        Generate the configurations, labels, budgets and curves based on
        the history of evaluated configurations.

        Returns
        -------
        configs: List
            The hyperparameter configurations.
        train_labels: List
            The performances of the hyperparameter configurations.
        train_budgets: List
            The budgets of the hyperparameter configurations.
        train_curves: List
            The learning curves of the hyperparameter configurations.
        """
        configs = []
        targets = []
        budgets = []
        curves = []

        for hp_index in self.budgets:
            budget = self.budgets[hp_index]
            scores = self.scores[hp_index]
            config = self.candidate_configs[hp_index]

            for n in range(len(scores)):
                configs.append(config)
                budgets.append(budget[n])
                targets.append(scores[n])
                curve = scores[: n + 1]
                curve = curve + [0.0] * (self.max_budget - len(curve))
                curves.append(curve)

        return configs, targets, budgets, curves

    def acq(
        self,
        ymax: float | np.ndarray,
        mean: float | np.ndarray,
        std: float | np.ndarray,
        cost: Optional[float | np.ndarray] = None,
    ) -> float | np.ndarray:
        """
        Calculate the acquisition function value for a given hyperparameter configuration.

        Args
        ----
        ymax: float | np.ndarray
            The best value observed so far for the given fidelity.
        mean: float | np.ndarray
            The mean prediction of the surrogate model.
        std: float | np.ndarray
            The standard deviation of the surrogate model.
        cost: Optional[float | np.ndarray], default = None
            The cost of the hyperparameter configuration.
        xi: float, default = 0.0
            The exploration factor for the acquisition function.

        Returns
        -------
        acq_value: float | np.ndarray
            The acquisition function value for the given hyperparameter configuration.
        """
        acq_fc = self.acq_func
        xi = self.explore_factor

        if cost is None:
            cost = 1e-4
        else:
            cost += 1e-4  # to avoid division by zero

        # Expected Improvement
        if acq_fc == "ei":
            mask = std == 0
            std = std + mask * 1.0
            z = (mean - ymax - xi) / std
            acq_value = (mean - ymax - xi) * norm.cdf(z) + std * norm.pdf(z)

            if isinstance(acq_value, float):
                acq_value = acq_value if mask else 0.0
            else:
                acq_value[mask] = 0.0

        # Upper Confidence Bound
        elif acq_fc == "ucb":
            acq_value = mean + xi * std

        # Thompson Sampling
        elif acq_fc == "thompson":
            acq_value = np.random.normal(mean, std)

        elif acq_fc == "exploit":
            # Exploitation
            acq_value = mean

        else:
            msg = f"acquisition function {acq_fc} is not implemented"
            raise NotImplementedError(msg)

        return acq_value / cost

    def find_suggested_config(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        budgets: np.ndarray,
        cost: Optional[np.ndarray] = None,
    ) -> List[int]:
        if cost is None:
            cost = np.ones_like(mean)
        else:
            cost = np.array(cost)

        ymax = self._get_ymax_per_budget()
        ymax = ymax[budgets]

        acq_values = self.acq(ymax, mean, std, cost)
        return np.argsort(acq_values).tolist()

    def _get_ymax_per_budget(self) -> np.ndarray:
        """
        Calculate the maximum performance for each budget level.

        Returns
        -------
        ymax: np.ndarray
            The maximum performance for each budget level.
        """
        from itertools import zip_longest

        ymax = np.zeros(self.max_budget)
        scores = self.scores.values()
        for n, score in enumerate(zip_longest(*scores, fillvalue=0)):
            ymax[n] = max(score)

        ymax[ymax == 0] = ymax.max()
        return ymax

    def update_info_dict(
        self,
        hp_index: int,
        budget: int,
        performance: float,
        overhead: float,
    ):
        """
        Update the info dict with the current HPO iteration info. Dump a new json file
        that will update with additional information given the current HPO iteration.

        Args
        ----
        hp_index: int
            The index of the hyperparameter configuration.
        budget: int
            The budget of the hyperparameter configuration.
        performance: float
            The performance of the hyperparameter configuration.
        overhead: float
            The overhead of the iteration.
        """
        hp_index = int(hp_index)
        if "hp" in self.info_dict:
            self.info_dict["hp"].append(hp_index)
        else:
            self.info_dict["hp"] = [hp_index]

        if "scores" in self.info_dict:
            self.info_dict["scores"].append(performance)
        else:
            self.info_dict["scores"] = [performance]

        if "curve" in self.info_dict:
            self.info_dict["curve"].append(self.best_value_observed)
        else:
            self.info_dict["curve"] = [self.best_value_observed]

        if "epochs" in self.info_dict:
            self.info_dict["epochs"].append(budget)
        else:
            self.info_dict["epochs"] = [budget]

        if "overhead" in self.info_dict:
            self.info_dict["overhead"].append(overhead)
        else:
            self.info_dict["overhead"] = [overhead]

        logger.info("Dumping info_dict to json file.")
        with open(os.path.join(self.output_path, "info_dict.json"), "w") as fp:
            json.dump(self.info_dict, fp)

    def patch_curves_to_same_length(
        self,
        curves: List[List[float,]],
        max_curve_length: Optional[int] = None,
    ):
        """
        Patch the given curves to the same length. Finds the maximum curve length and
        patches all other curves that are shorter in length with zeroes.

        Args
        ----
        curves :
            The given hyperparameter curves
        max_curve_length : int, optional
            The maximum curve length. Defaults to max_benchmark_epochs.

        Returns
        -------
        curves :
            The patched hyperparameter curves
        """
        if max_curve_length is None:
            max_curve_length = self.max_budget
        for curve in curves:
            if len(curve) > max_curve_length:
                max_curve_length = len(curve)

        for curve in curves:
            difference = max_curve_length - len(curve)
            if difference > 0:
                curve.extend([0.0] * difference)

        return curves
