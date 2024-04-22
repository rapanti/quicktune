import copy
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch import Tensor
from scipy.stats import norm

from quicktune.data import MetaSet
from quicktune.configuration_manager import ConfigurationManager


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
        surrogate: nn.Module,
        config_manager: ConfigurationManager,
        metaset: MetaSet,
        num_configs: int,
        metafeatures: Optional[torch.Tensor] = None,
        max_benchmark_epochs: int = 50,
        fantasize_steps: int = 1,
        acq_func: str = "ei",
        explore_factor: float = 0.0,
        output_path: str = ".",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cm = config_manager
        self.metafeatures = metafeatures

        self.sampled_configs = self.cm.sample_configuration(num_configs)
        self.candidate_configs = self.cm.preprocess_configurations(
            self.sampled_configs, metaset
        ).values

        self.examples = dict()
        self.performances = dict()

        self.max_benchmark_epochs = max_benchmark_epochs
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

        self.no_improvement_threshold = int(
            self.max_benchmark_epochs + 0.2 * self.max_benchmark_epochs
        )
        self.no_improvement_patience = 0
        self.converged_configs = []
        self.acq_func = acq_func
        self.explore_factor = explore_factor

    def _prepare_dataset_and_budgets(self) -> Dict[str, torch.Tensor]:
        """
        Prepare the data that will be the input to the surrogate.

        Returns
        -------
        data: Dict[str, torch.Tensor]
            The data that will be the input to the surrogate.
        """

        args, targets, budgets, curves = self.history_configurations()

        args = np.array(args, dtype=np.single)
        targets = np.array(targets, dtype=np.single)
        budgets = np.array(budgets, dtype=np.single)
        curves = self.patch_curves_to_same_length(curves)
        curves = np.array(curves, dtype=np.single)

        # scale budgets to [0, 1]
        budgets = budgets / self.max_benchmark_epochs

        args = torch.tensor(args)
        targets = torch.tensor(targets)
        budgets = torch.tensor(budgets)
        curves = torch.tensor(curves)

        args = args.to(device=self.device)
        targets = targets.to(device=self.device)
        budgets = budgets.to(device=self.device)
        curves = curves.to(device=self.device)

        metafeatures = None
        if self.metafeatures is not None:
            metafeatures = self.metafeatures.repeat(args.size(0), 1).to(self.device)

        data = {
            "args": args,
            "budgets": budgets,
            "curves": curves,
            "targets": targets,
            "metafeatures": metafeatures,
        }

        return data

    def _train_surrogate(self):
        """
        Train the surrogate model with the observed hyperparameter configurations.
        """
        data = self._prepare_dataset_and_budgets()
        self.surrogate.to(self.device)
        self.surrogate.train_pipeline(data)

    def _predict(self) -> Tuple[Tensor, Tensor, Optional[Tensor], List, np.ndarray]:
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
        configurations, hp_indices, budgets, learning_curves = (
            self.generate_candidate_configurations()
        )
        budgets = np.array(budgets, dtype=np.single)
        non_scaled_budgets = copy.deepcopy(budgets)

        # scale budgets to [0, 1]
        budgets = budgets / self.max_benchmark_epochs

        configurations = np.array(configurations, dtype=np.single)
        configurations = torch.tensor(configurations)
        configurations = configurations.to(device=self.device)

        budgets = torch.tensor(budgets)
        budgets = budgets.to(device=self.device)

        learning_curves = self.patch_curves_to_same_length(learning_curves)
        learning_curves = np.array(learning_curves, dtype=np.single)
        learning_curves = torch.tensor(learning_curves)
        learning_curves = learning_curves.to(device=self.device)

        train_data = self._prepare_dataset_and_budgets()

        metafeatures = None
        if self.metafeatures is not None:
            metafeatures = self.metafeatures.repeat(configurations.size(0), 1).to(
                self.device
            )
        test_data = {
            "args": configurations,
            "budgets": budgets,
            "curves": learning_curves,
            "metafeatures": metafeatures,
        }
        self.surrogate.to(self.device)
        mean_predictions, std_predictions, costs = self.surrogate.predict_pipeline(
            train_data, test_data
        )

        return mean_predictions, std_predictions, costs, hp_indices, non_scaled_budgets

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
        # check if we still have random hyperparameters to evaluate
        if self.initial_random_index < len(self.init_conf_indices):
            print(
                "Not enough configurations to build a model. "
                "Returning randomly sampled configuration"
            )

            random_indice = self.init_conf_indices[self.initial_random_index]
            budget = self.init_budgets[self.initial_random_index]
            self.initial_random_index += 1

            return random_indice, budget

        else:
            mean_predictions, std_predictions, costs, hp_indices, non_scaled_budgets = (
                self._predict()
            )

            mean_predictions = mean_predictions.cpu().detach().numpy()
            std_predictions = std_predictions.cpu().detach().numpy()

            best_prediction_index = self.find_suggested_config(
                mean_predictions, std_predictions, non_scaled_budgets, costs
            )
            """
            the best prediction index is not always matching with the actual hp index.
            Since when evaluating the acq function, we do not consider hyperparameter
            candidates that diverged or that are evaluated fully.
            """
            best_config_index = hp_indices[best_prediction_index]

            # decide for what budget we will evaluate the most promising hyperparameter configuration next.
            if best_config_index in self.examples:
                evaluated_budgets = self.examples[best_config_index]
                max_budget = max(evaluated_budgets)
                budget = max_budget + self.fantasize_steps
                # this would only trigger if fantasize_step is bigger than 1
                if budget > self.max_benchmark_epochs:
                    budget = self.max_benchmark_epochs
            else:
                budget = self.fantasize_steps

        return best_config_index, budget

    def observe(
        self,
        hp_index: int,
        b: int,
        learning_curve: np.ndarray,
    ):
        """
        Observe the learning curve of a hyperparameter configuration.

        Args
        ----
        hp_index: int
            The index of the hyperparameter configuration.
        b: int
            The budget of the hyperparameter configuration.
        learning_curve: np.ndarray
            The learning curve of the hyperparameter configuration.

        Returns
        -------
        overhead_time: float
            The overhead time of the iteration.
        """
        score = learning_curve[-1]
        # if y is an undefined value, append 0 as the overhead since we finish here.
        if np.isnan(learning_curve).any():
            self.update_info_dict(hp_index, b, np.nan, 0)
            self.diverged_configs.add(hp_index)
            return

        observe_time_start = time.time()

        # self.examples[hp_index] = np.arange(b + 1).tolist()
        self.examples[hp_index] = np.arange(1, b + 1).tolist()
        self.performances[hp_index] = learning_curve

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

            self._train_surrogate()

            train_time_end = time.time()
            train_time_duration = train_time_end - train_time_start

        observe_time_duration = observe_time_end - observe_time_start
        overhead_time = (
            observe_time_duration + self.suggest_time_duration + train_time_duration
        )
        total_duration = overhead_time
        self.update_info_dict(hp_index, b, score, total_duration)
        return overhead_time

    def prepare_examples(self, hp_indices: List) -> List[np.ndarray]:
        """
        Prepare the examples to be given to the surrogate model.

        Args
        ----
        hp_indices: List
            The indices of the hyperparameter configurations.

        Returns
        -------
        examples : List
            A list of the hyperparameter configurations.
        """
        examples = []
        for hp_index in hp_indices:
            examples.append(self.candidate_configs[hp_index])

        return examples

    def generate_candidate_configurations(self) -> Tuple[List, List, List, List]:
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
        hp_indices = []
        hp_budgets = []
        learning_curves = []

        for hp_index in range(0, self.candidate_configs.shape[0]):
            if hp_index in self.converged_configs:
                continue

            if hp_index in self.examples:
                budgets = self.examples[hp_index]
                # Take the max budget evaluated for a certain hpc
                max_budget = max(budgets)
                next_budget = max_budget + self.fantasize_steps
                # take the learning curve until the point we have evaluated so far
                # curve = self.performances[hp_index][:max_budget - 1] if max_budget > 1 else [0.0]
                curve = self.performances[hp_index][:max_budget]
                # if the curve is shorter than the length of the kernel size, pad it with zeros
                difference_curve_length = self.max_benchmark_epochs - len(curve)
                if difference_curve_length > 0:
                    curve.extend([0.0] * difference_curve_length)
            else:
                # The hpc was not evaluated before, so fantasize its performance
                next_budget = self.fantasize_steps
                curve = [0, 0, 0]

            # this hyperparameter configuration is not evaluated fully
            if next_budget <= self.max_benchmark_epochs:
                hp_indices.append(hp_index)
                hp_budgets.append(next_budget)
                learning_curves.append(curve)

        configurations = self.prepare_examples(hp_indices)
        return configurations, hp_indices, hp_budgets, learning_curves

    def history_configurations(
        self,
    ) -> Tuple[List, List, List, List]:
        """
        Generate the configurations, labels, budgets and curves based on
        the history of evaluated configurations.

        Returns
        -------
        train_examples: List
            The hyperparameter configurations.
        train_labels: List
            The performances of the hyperparameter configurations.
        train_budgets: List
            The budgets of the hyperparameter configurations.
        train_curves: List
            The learning curves of the hyperparameter configurations.
        """
        train_examples = []
        train_labels = []
        train_budgets = []
        train_curves = []

        for hp_index in self.examples:
            budgets = self.examples[hp_index]
            performances = self.performances[hp_index]
            example = self.candidate_configs[hp_index]

            for budget, performance in zip(budgets, performances):
                train_examples.append(example)
                train_budgets.append(budget)
                train_labels.append(performance)
                train_curve = performances[: budget - 1] if budget > 1 else [0.0]
                # difference_curve_length = self.surrogate_config['cnn_kernel_size']- len(train_curve)
                difference_curve_length = self.max_benchmark_epochs - len(train_curve)
                if difference_curve_length > 0:
                    train_curve.extend([0.0] * difference_curve_length)

                train_curves.append(train_curve)

        return train_examples, train_labels, train_budgets, train_curves

    def acq(
        self,
        best_value: float,
        mean: float,
        std: float,
        explore_factor: float = 0.,
        acq_fc: str = "ei",
        cost: float = 1,
    ) -> np.ndarray | float:
        """
        Calculate the acquisition function value for a given hyperparameter configuration.

        Args
        ----
        best_value: float
            The best value observed so far for the given fidelity.
        mean: float
            The mean prediction of the surrogate model.
        std: float
            The standard deviation of the surrogate model.
        explore_factor: float, default = 0.
            The exploration factor for the acquisition function.
        acq_fc: str, default = "ei"
            The acquisition function to be used.
        cost: float, default = 1
            The cost of the hyperparameter configuration.

        Returns
        -------
        acq_value: np.ndarray | float
            The acquisition function value for the given hyperparameter configuration.
        """
        if acq_fc == "ei":
            if std == 0:
                return 0
            z = (mean - best_value - explore_factor) / std
            acq_value = (mean - best_value - explore_factor) * norm.cdf(
                z
            ) + std * norm.pdf(z)
        elif acq_fc == "ucb":
            acq_value = mean + explore_factor * std
        elif acq_fc == "thompson":
            acq_value = np.random.normal(mean, std)
        elif acq_fc == "exploit":
            acq_value = mean
        else:
            raise NotImplementedError(
                f"Acquisition function {acq_fc} has not been implemented"
            )

        if cost != 0:
            return acq_value / cost
        else:
            return acq_value / (1e-4)

    def find_suggested_config(
        self,
        mean_predictions: torch.Tensor,
        mean_stds: torch.Tensor,
        budgets: np.ndarray,
        costs: Optional[torch.Tensor] = None,
    ) -> int:
        """
        Find the hyperparameter configuration that has the highest score.

        Args
        ----
        mean_predictions: torch.Tensor
            The mean predictions of the surrogate model.
        mean_stds: torch.Tensor
            The standard deviations of the surrogate model.
        budgets: np.ndarray
            The budgets of the hyperparameter configurations.
        costs: Optional[torch.Tensor]
            The costs of the hyperparameter configurations.

        Returns
        -------
        best_index: int
            The index of the best hyperparameter configuration.
        """
        highest_acq_value = np.NINF
        best_index = -1

        index = 0
        for mean_value, std in zip(mean_predictions, mean_stds):
            budget = int(budgets[index])
            cost = costs[index] if costs is not None else 1
            best_value = self.calculate_fidelity_ymax(budget)
            acq_value = self.acq(
                best_value,
                mean_value,
                std,
                acq_fc=self.acq_func,
                explore_factor=self.explore_factor,
                cost=cost,
            )
            if acq_value > highest_acq_value:
                print(f"Acq: {acq_value}, index: {index}")
                highest_acq_value = acq_value
                best_index = index

            index += 1

        return best_index

    def calculate_fidelity_ymax(self, fidelity: int) -> float:
        """
        Find ymax for a given fidelity level. If there are hyperparameters evaluated
        for that fidelity take the maximum from their values. Otherwise,
        take the maximum from all previous fidelity levels for the hyperparameters
        that we have evaluated.

        Args
        ----
        fidelity: int
            The fidelity of the hyperparameter configuration.

        Returns
        -------
        best_value: float
        The best value seen so far for the given fidelity.
        """
        exact_fidelity_config_values = []
        lower_fidelity_config_values = []

        for example_index in self.examples.keys():
            try:
                performance = self.performances[example_index][fidelity - 1]
                exact_fidelity_config_values.append(performance)
            except IndexError:
                learning_curve = self.performances[example_index]
                # The hyperparameter was not evaluated until fidelity, or more.
                # Take the maximum value from the curve.
                lower_fidelity_config_values.append(max(learning_curve))

        if len(exact_fidelity_config_values) > 0:
            # lowest error corresponds to best value
            best_value = max(exact_fidelity_config_values)
        else:
            best_value = max(lower_fidelity_config_values)

        return best_value

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
            max_curve_length = self.max_benchmark_epochs
        for curve in curves:
            if len(curve) > max_curve_length:
                max_curve_length = len(curve)

        for curve in curves:
            difference = max_curve_length - len(curve)
            if difference > 0:
                curve.extend([0.0] * difference)

        return curves
