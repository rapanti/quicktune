import copy
import json
import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

from quicktune.optimizers.dyhpo import DyHPO


class QuickTuneOptimizer:
    def __init__(
        self,
        model: DyHPO,
        hp_candidates: np.ndarray,
        log_indicator: List[bool],
        hp_names: List[str],
        init_conf_indices: list = [],
        seed: int = 11,
        max_benchmark_epochs: int = 50,
        fantasize_step: int = 1,
        minimization: bool = True,
        total_budget: int = 10000,  # budget in epochs (not time)
        device: str = '',
        dataset_name: str = "dataset",
        output_path: str = ".",
        acqf_fc: str = "ucb",
        explore_factor=1.0,
        learning_rate: float = 0.001,
        apply_preprocessing: bool = False,
        verbose: bool = True,
    ):
        """
        Args:
            hp_candidates: np.ndarray
                The full list of hyperparameter candidates for
                a given dataset.
            log_indicator: List
                A list with boolean values indicating if a
                hyperparameter has been log sampled or not.
            seed: int
                The seed that will be used for the surrogate.
            max_benchmark_epochs: int
                The maximal budget that a hyperparameter configuration
                has been evaluated in the benchmark for.
            fantasize_step: int
                The number of steps for which we are looking ahead to
                evaluate the performance of a hpc.
            minimization: bool
                If the objective should be maximized or minimized.
            total_budget: int
                The total budget (epochs) given for hyperparameter optimization.
            device: str
                The device where the experiment will be run on.
            dataset_name: str
                The name of the dataset that the experiment will be run on.
            output_path: str
                The path where all the output will be stored.
            surrogate_config: dict
                The model configurations for the surrogate.
            verbose: boolean
                If detailed information is preferred in the log file.
        """
        cudnn.deterministic = True
        cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")            

        self.hp_candidates = hp_candidates
        self.log_indicator = log_indicator
        self.hp_names = hp_names

        if apply_preprocessing:
            self.scaler = MinMaxScaler()
            self.hp_candidates = self.preprocess_hp_candidates()
        else:
            self.scaler = None
            self.hp_candidates = np.array(self.hp_candidates)

        self.minimization = minimization
        self.seed = seed
        self.learning_rate = learning_rate

        if verbose:
            logging_level = logging.DEBUG
        else:
            logging_level = logging.INFO

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

        # the keys will be hyperparameter indices while the value
        # will be a list with all the budgets evaluated for examples
        # and with all performances for the performances
        self.examples = dict()
        self.performances = dict()

        # set a seed already, so that it is deterministic when
        # generating the seeds of the ensemble
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.max_benchmark_epochs = max_benchmark_epochs
        self.total_budget = total_budget
        self.fantasize_step = fantasize_step
        self.nr_features = self.hp_candidates.shape[1]
        self.rng = np.random.default_rng(seed)

        conf_individual_budget = 1
        if not init_conf_indices:
            initial_configurations_nr = 1
            self.init_conf_indices = self.rng.choice(
                self.hp_candidates.shape[0], initial_configurations_nr, replace=False
            )
        else:
            initial_configurations_nr = len(init_conf_indices)
            self.init_conf_indices = np.array(init_conf_indices)

        self.init_budgets = [conf_individual_budget] * initial_configurations_nr
        # with what percentage configurations will be taken randomly instead of being sampled from the model
        self.fraction_random_configs = 0.1

        self.model = model
        # An index keeping track of where we are in the init_conf_indices
        # list of hyperparmeters that are not sampled from the model.
        self.initial_random_index = 0

        # the incumbent value observed during the hpo process.
        self.best_value_observed = np.NINF
        # a set which will keep track of the hyperparameter configurations that diverge.
        self.diverged_configs = set()

        # info dict to drop every surrogate iteration
        self.info_dict = dict()

        # the start time for the overhead of every surrogate optimization iteration
        # will be recorded here
        self.suggest_time_duration = 0
        # the total budget consumed so far
        self.budget_spent = 0

        self.output_path = output_path
        self.dataset_name = dataset_name

        self.no_improvement_threshold = int(self.max_benchmark_epochs + 0.2 * self.max_benchmark_epochs)
        self.no_improvement_patience = 0
        self.converged_configs = []
        self.acq_fc = acqf_fc
        self.explore_factor = explore_factor
        self.cost_trainer = None

    def _prepare_dataset_and_budgets(self) -> Dict[str, torch.Tensor]:
        """
        Prepare the data that will be the input to the surrogate.
        Returns:
            data: A Dictionary that contains inside the training examples,
            the budgets, the curves and lastly the labels.
        """

        train_examples, train_labels, train_budgets, train_curves = self.history_configurations()

        train_examples = np.array(train_examples, dtype=np.single)
        train_labels = np.array(train_labels, dtype=np.single)
        train_budgets = np.array(train_budgets, dtype=np.single)
        train_curves = self.patch_curves_to_same_length(train_curves)
        train_curves = np.array(train_curves, dtype=np.single)

        # scale budgets to [0, 1]
        train_budgets = train_budgets / self.max_benchmark_epochs

        train_examples = torch.tensor(train_examples)
        train_labels = torch.tensor(train_labels)
        train_budgets = torch.tensor(train_budgets)
        train_curves = torch.tensor(train_curves)

        train_examples = train_examples.to(device=self.device)
        train_labels = train_labels.to(device=self.device)
        train_budgets = train_budgets.to(device=self.device)
        train_curves = train_curves.to(device=self.device)

        data = {
            "X_train": train_examples,
            "train_budgets": train_budgets,
            "train_curves": train_curves,
            "y_train": train_labels,
        }

        return data

    def _train_surrogate(self):
        """
        Train the surrogate model.
        """
        data = self._prepare_dataset_and_budgets()
        self.logger.info("Started training the model")

        self.model.train_pipeline(
            data,
            load_checkpoint=False,
        )

    def _predict(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List, List]:
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
        configurations, hp_indices, budgets, learning_curves = self.generate_candidate_configurations()
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
        test_data = {
            "X_test": configurations,
            "test_budgets": budgets,
            "test_curves": learning_curves,
        }
        mean_predictions, std_predictions, costs = self.model.predict_pipeline(train_data, test_data)

        return mean_predictions, std_predictions, costs, hp_indices, non_scaled_budgets

    def set_hyperparameter_candidates(self, hyperparameter_candidates, init_conf_indices=None):
        """
        Set the hyperparameter candidates that will be used to build the surrogate model.
        Args:
            hyperparameter_candidates: A list of dictionaries that contain the hyperparameter
                candidates.
        """
        self.hp_candidates = hyperparameter_candidates
        self.nr_features = self.hp_candidates.shape[1]
        conf_individual_budget = 1
        if init_conf_indices is None:
            initial_configurations_nr = 1
            self.init_conf_indices = self.rng.choice(
                self.hp_candidates.shape[0], initial_configurations_nr, replace=False
            )
        else:
            initial_configurations_nr = len(init_conf_indices)
            self.init_conf_indices = np.array(init_conf_indices)

        self.init_budgets = [conf_individual_budget] * initial_configurations_nr

    def suggest(self) -> Tuple[int, int]:
        """
        Suggest a hyperparameter configuration to be evaluated next.
        Returns:
            best_config_index, budget: The index of the hyperparamter
                configuration to be evaluated and the budget for
                what it is going to be evaluated for.
        """
        suggest_time_start = time.time()
        # check if we still have random hyperparameters to evaluate
        if self.initial_random_index < len(self.init_conf_indices):
            self.logger.info("Not enough configurations to build a model. Returning randomly sampled configuration")

            random_indice = self.init_conf_indices[self.initial_random_index]
            budget = self.init_budgets[self.initial_random_index]
            self.initial_random_index += 1

            return random_indice, budget
        else:
            mean_predictions, std_predictions, costs, hp_indices, non_scaled_budgets = self._predict()

            best_prediction_index = self.find_suggested_config(
                mean_predictions, std_predictions, non_scaled_budgets, costs
            )
            """
            the best prediction index is not always matching with the actual hp index.
            Since when evaluating the acq function, we do not consider hyperparameter
            candidates that diverged or that are evaluated fully.
            """
            best_config_index = hp_indices[best_prediction_index]

            # decide for what budget we will evaluate the most
            # promising hyperparameter configuration next.
            if best_config_index in self.examples:
                evaluated_budgets = self.examples[best_config_index]
                max_budget = max(evaluated_budgets)
                budget = max_budget + self.fantasize_step
                # this would only trigger if fantasize_step is bigger than 1
                if budget > self.max_benchmark_epochs:
                    budget = self.max_benchmark_epochs
            else:
                budget = self.fantasize_step

        suggest_time_end = time.time()
        self.suggest_time_duration = suggest_time_end - suggest_time_start

        self.budget_spent += self.fantasize_step

        # exhausted hpo budget, finish.
        if self.budget_spent > self.total_budget:
            exit(0)

        return best_config_index, budget

    def observe(
        self,
        hp_index: int,
        b: int,
        learning_curve: np.ndarray,
        alg_time: Optional[float] = None,
    ):
        """
        Args:
            hp_index: The index of the evaluated hyperparameter configuration.
            b: The budget for which the hyperparameter configuration was evaluated.
            learning_curve: The learning curve of the hyperparameter configuration.
            alg_time: The time taken from the algorithm to evaluate the hp configuration.
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

        if alg_time is not None and self.cost_trainer is not None and len(self.info_dict) > 0:
            self.cost_trainer.info_dict = self.info_dict
            self.cost_trainer.train()

        observe_time_end = time.time()
        train_time_duration = 0

        # initialization phase over. Now we can sample from the model.
        if self.initial_random_index >= len(self.init_conf_indices):
            train_time_start = time.time()
            # create the model for the first time
            assert self.model is not None

            if self.no_improvement_patience == self.no_improvement_threshold:
                self.model.restart = True

            self._train_surrogate()

            train_time_end = time.time()
            train_time_duration = train_time_end - train_time_start

        observe_time_duration = observe_time_end - observe_time_start
        overhead_time = observe_time_duration + self.suggest_time_duration + train_time_duration
        if alg_time is not None:
            total_duration = overhead_time + alg_time
        else:
            total_duration = overhead_time
        self.update_info_dict(hp_index, b, score, total_duration)
        return overhead_time

    def prepare_examples(self, hp_indices: List) -> List[np.ndarray]:
        """
        Prepare the examples to be given to the surrogate model.
        Args:
            hp_indices: The list of hp indices that are already evaluated.
        Returns:
            examples: A list of the hyperparameter configurations.
        """
        examples = []
        for hp_index in hp_indices:
            examples.append(self.hp_candidates[hp_index])

        return examples

    def generate_candidate_configurations(
        self,
    ) -> Tuple[List, List, List, List]:
        """
        Generate candidate configurations that will be
        fantasized upon.
        Returns:
            (configurations, hp_indices, hp_budgets, learning_curves): Tuple
                A tuple of configurations, their indices in the hp list
                and the budgets that they should be fantasized upon.
        """
        hp_indices = []
        hp_budgets = []
        learning_curves = []

        for hp_index in range(0, self.hp_candidates.shape[0]):
            if hp_index in self.converged_configs:
                continue

            if hp_index in self.examples:
                budgets = self.examples[hp_index]
                # Take the max budget evaluated for a certain hpc
                max_budget = max(budgets)
                next_budget = max_budget + self.fantasize_step
                # take the learning curve until the point we have evaluated so far
                # curve = self.performances[hp_index][:max_budget - 1] if max_budget > 1 else [0.0]
                curve = self.performances[hp_index][:max_budget]
                # if the curve is shorter than the length of the kernel size,
                # pad it with zeros
                difference_curve_length = self.max_benchmark_epochs - len(curve)
                if difference_curve_length > 0:
                    curve.extend([0.0] * difference_curve_length)
            else:
                # The hpc was not evaluated before, so fantasize its
                # performance
                next_budget = self.fantasize_step
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
        Returns:
            (train_examples, train_labels, train_budgets, train_curves):
                A tuple of examples, labels, budgets and curves for the
                configurations evaluated so far.
        """
        train_examples = []
        train_labels = []
        train_budgets = []
        train_curves = []

        for hp_index in self.examples:
            budgets = self.examples[hp_index]
            performances = self.performances[hp_index]
            example = self.hp_candidates[hp_index]

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
        explore_factor: Optional[float] = 0.25,
        acq_fc: str = "ei",
        cost: Optional[float] = 1,
    ) -> float:
        """
        The acquisition function that will be called
        to evaluate the score of a hyperparameter configuration.
        Parameters
        ----------
        best_value: float
            Best observed function evaluation. Individual per fidelity.
        mean: float
            Point mean of the posterior process.
        std: float
            Point std of the posterior process.
        explore_factor: float
            The exploration factor for when ucb is used as the
            acquisition function.
        ei_calibration_factor: float
            The factor used to calibrate expected improvement.
        acq_fc: str
            The type of acquisition function to use.
        Returns
        -------
        acq_value: float
            The value of the acquisition function.
        """
        if acq_fc == "ei":
            if std == 0:
                return 0
            z = (mean - best_value - explore_factor) / std
            acq_value = (mean - best_value - explore_factor) * norm.cdf(z) + std * norm.pdf(z)
        elif acq_fc == "ucb":
            acq_value = mean + explore_factor * std
        elif acq_fc == "thompson":
            acq_value = np.random.normal(mean, std)
        elif acq_fc == "exploit":
            acq_value = mean
        else:
            raise NotImplementedError(
                f"Acquisition function {acq_fc} has not been" f"implemented",
            )

        if cost != 0:
            return acq_value / cost
        else:
            return acq_value / (1e-4)

    def find_suggested_config(
        self,
        mean_predictions: np.ndarray,
        mean_stds: np.ndarray,
        budgets: List,
        costs: Optional[np.array] = None,
        return_highest_acq_value: bool = False,
    ):
        """
        Find the hyperparameter configuration that has the highest score
        with the acquisition function.
        Args:
            mean_predictions: The mean predictions of the posterior.
            mean_stds: The mean standard deviations of the posterior.
            budgets: The next budgets that the hyperparameter configurations
                will be evaluated for.
        Returns:
            best_index: The index of the hyperparameter configuration with the
                highest score.
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
                acq_fc=self.acq_fc,
                explore_factor=self.explore_factor,
                cost=cost,
            )
            if acq_value > highest_acq_value:
                highest_acq_value = acq_value
                best_index = index

            index += 1

        if return_highest_acq_value:
            return best_index, highest_acq_value
        else:
            return best_index

    def calculate_fidelity_ymax(self, fidelity: int):
        """
        Find ymax for a given fidelity level.
        If there are hyperparameters evaluated for that fidelity
        take the maximum from their values. Otherwise, take
        the maximum from all previous fidelity levels for the
        hyperparameters that we have evaluated.
        Args:
            fidelity: The fidelity of the hyperparameter
                configuration.
        Returns:
            best_value: The best value seen so far for the
                given fidelity.
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
        Update the info dict with the current HPO iteration info.
        Dump a new json file that will update with additional information
        given the current HPO iteration.
        Args:
            hp_index: The index of the hyperparameter configuration.
            budget: The budget of the hyperparameter configuration.
            performance:  The performance of the hyperparameter configuration.
            overhead: The total overhead (in seconds) of the iteration.
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

        with open(os.path.join(self.output_path, f"{self.dataset_name}_{self.seed}.json"), "w") as fp:
            json.dump(self.info_dict, fp)

    def preprocess_hp_candidates(self) -> List:
        """
        Preprocess the list of all hyperparameter candidates
        by  performing a log transform for the hyperparameters that
        were log sampled.
        Returns:
            log_hp_candidates: The list of all hyperparameter configurations
                where hyperparameters that were log sampled are log transformed.
        """
        log_hp_candidates = []

        for hp_candidate in self.hp_candidates:
            new_hp_candidate = []
            for index, hp_value in enumerate(hp_candidate):
                new_hp_candidate.append(math.log(hp_value) if self.log_indicator[index] else hp_value)

            log_hp_candidates.append(new_hp_candidate)

        log_hp_candidates = np.array(log_hp_candidates)
        # scaler for the hp configurations

        log_hp_candidates = self.scaler.fit_transform(log_hp_candidates)

        return log_hp_candidates

    @staticmethod
    def patch_curves_to_same_length(curves, max_curve_length=50):
        """
        Patch the given curves to the same length.
        Finds the maximum curve length and patches all
        other curves that are shorter in length with zeroes.
        Args:
            curves: The given hyperparameter curves.
        Returns:
            curves: The updated array where the learning
                curves are of the same length.
        """
        for curve in curves:
            if len(curve) > max_curve_length:
                max_curve_length = len(curve)

        for curve in curves:
            difference = max_curve_length - len(curve)
            if difference > 0:
                curve.extend([0.0] * difference)

        return curves

    def project_to_valid_range(self, hp_name, hp_value):
        valid_values = None
        if hp_name in ["pct_to_freeze", "lr", "warmp_lr", "weight_decay"]:
            # clap value to range
            hp_value = max(0.0, min(1.0, hp_value))

        elif hp_name in [
            "trivial_augment",
            "random_augment",
            "auto_augment",
            "linear_probing",
            "stoch_norm",
        ]:
            valid_values = [0, 1]

        elif hp_name in [
            "patience_epochs",
            "ra_num_ops",
            "ra_magnitude",
            "decay_epochs",
            "epochs",
            "batch_size",
            "num_classes",
            "warmup_epochs",
        ]:
            hp_value = int(hp_value)

        elif hp_name == "clip_grad":
            valid_values = [-1] + np.arange(1, 10).tolist()

        elif hp_name == "layer_decay":
            valid_values = [-1] + [0.65, 0.75]

        if valid_values is not None:
            hp_value = valid_values[np.digitize(hp_value, valid_values) - 1]

        return hp_value

    def to_qt_config(self, values, mean=None, std=None):
        """
        Convert the hyperparameter configuration to the qt format.
        Returns:
            qt_config: The hyperparameter configuration in the qt format.
        """
        assert len(values) == len(self.hp_names), "The number of hyperparameters does not match the number of values."
        categorical_hp = {
            "auto_augment": ["", -1000],
            "model": ["", -1000],
            "opt_betas": ["", -1000],
            "opt": ["", -1000],
            "sched": ["", -1000],
        }
        qt_config = {}

        for hp_name, hp_value in zip(self.hp_names, values):
            hp_value = self.project_to_valid_range(hp_name, hp_value)

            if hp_name.startswith("cat__"):
                for hp in categorical_hp:
                    if hp_name.startswith(f"cat__{hp}_"):
                        hp_category = hp_name.split(f"{hp}_")[-1]
                        if hp_category.startswith("betas"):  # fails with opt_betas
                            continue
                        if categorical_hp[hp][1] < hp_value:
                            categorical_hp[hp] = [hp_category, hp_value]
            else:
                if mean is not None and std is not None:
                    hp_value = hp_value * std[hp_name] + mean[hp_name]

                if hp_value == -1:
                    pass
                else:
                    qt_config[hp_name] = hp_value

        for hp in categorical_hp:
            if categorical_hp[hp][0] != "":
                if hp == "sched" and categorical_hp[hp][1] == 0:
                    qt_config[hp] = "None"
                else:
                    qt_config[hp] = categorical_hp[hp][0]
        return qt_config
