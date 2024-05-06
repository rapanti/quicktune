import json
import logging
import os
import time
from typing import Callable, Optional

from quicktune.optimizers.quick import QuickOptimizer
from quicktune.utils.log_utils import add_log_to_file, set_logger_verbosity
from quicktune.utils.qt_utils import QTunerResult, get_dataset_metafeatures
from quicktune.utils.utils import setup_outputdir

logger = logging.getLogger("QuickTuner")


class QuickTuner:
    """
    QuickTuner class for performing hyperparameter tuning using the QuickTuneTool.

    Parameters
    ----------
        config : dict
            The configuration dictionary for the experiment.
        objective_function : Callable[[dict], QTunerResult]
            The objective function which is used to evaluate the performance of the suggested configurations.
            The function should take a dictionary as input and return a dictionary with the following keys [TODO]
        path : Optional[str, default = None
            Path to directory where models and intermediate outputs should be saved.
        verbosity : int, default = 2
            Verbosity levels range from 0 to 4 and control how much information is printed.
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
            If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
            where `L` ranges from 0 to 50 (Note: higher values of `L` correspond to fewer print statements, opposite of verbosity levels).
            Verbosity levels:
                0: Only log exceptions
                1: Only log warnings + exceptions
                2: Standard logging
                3: Verbose logging (ex: log validation score every 50 iterations)
                4: Maximally verbose logging (ex: log validation score every iteration)
        **kwargs :
            _log_to_file : bool, default = True
                Whether to log output to a file. Defaults to True.
            _log_file_path : str, default = "auto"
                The path to the log file. Defaults to "auto".

    """

    _log_to_file: bool = True
    _log_file_name: str = "quicktuner_log.txt"
    _log_file_path: str = "auto"

    def __init__(
        self,
        optimizer: QuickOptimizer,
        objective_function: Callable[..., QTunerResult],
        path: Optional[str] = None,
        verbosity: int = 4,
        **kwargs,
    ):
        self.verbosity = verbosity
        set_logger_verbosity(verbosity, logger)
        self.path = setup_outputdir(path)
        self.exp_dir = os.path.join(self.path, "exp")
        self._setup_log_to_file(self._log_to_file, self._log_file_path)

        self._validate_init_kwargs(kwargs)

        self.optimizer = optimizer
        self.objective_function = objective_function

    def _setup_log_to_file(self, log_to_file: bool, log_file_path: str) -> None:
        """
        Setup logging to a file if enabled.

        Args:
            log_to_file (bool): Whether to log output to a file.
            log_file_path (str): The path to the log file.

        Returns:
            None
        """
        if log_to_file:
            if log_file_path == "auto":
                log_file_path = os.path.join(self.path, "logs", self._log_file_name)
            log_file_path = os.path.abspath(os.path.normpath(log_file_path))
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            add_log_to_file(log_file_path, logger)

    def fit(
        self,
        data_path: str,
        time_limit: Optional[float] = None,
        train_split: str = "train",
        val_split: str = "val",
    ):
        """
        Fit the QuickTuner model.

        Parameters
            time_limit (Optional[float]): The time limit for the tuning process. Defaults to None.

        Returns:
            None
        """
        logger.info("Starting QuickTuner fit.")
        logger.info(f"QuickTuneTool will save results to {self.path}")

        if time_limit is None:
            time_limit = float("inf")

        # if self.config.include_metafeatures:
        metafeat = get_dataset_metafeatures(data_path)
        self.optimizer.set_metafeatures(**metafeat)

        data_info = {
            "train-split": train_split,
            "val-split": val_split,
            "num_classes": metafeat["num_classes"],
        }

        orig_configs = self.optimizer.sample_configs
        configs = {i: config.get_dictionary() for i, config in enumerate(orig_configs)}
        with open(os.path.join(self.path, "configs.json"), "w") as f:
            json.dump(configs, f, indent=2, sort_keys=True)

        history: dict = {"score": [], "cost": [], "configs": dict()}

        start_time = time.time()
        while True:
            config_id, budget = self.optimizer.suggest()
            logger.info(f"Optimizer suggests: {config_id} with budget {budget}")

            config = self.optimizer.sample_configs[config_id].get_dictionary()

            result = self.objective_function(
                budget=budget,
                config=config,
                config_id=config_id,
                data_path=data_path,
                data_info=data_info,
                output=self.exp_dir,
                verbosity=self.verbosity,
            )

            logger.info("Evaluation complete.")
            logger.info(f"Score: {result.score:.3f}% | Time: {result.time:.1f}s")

            self.optimizer.observe(config_id, budget, result)

            # save results
            score = result.score
            if config_id not in history["configs"].keys():
                history["configs"][config_id] = list()
            history["configs"][config_id].append(score)
            history["score"].append(score)
            history["cost"].append(time.time() - start_time)
            # save curves
            with open(os.path.join(self.path, "history.json"), "w") as f:
                logger.info(f"Saving history to {self.path}")
                json.dump(history, f, indent=2, sort_keys=True)

            if (time.time() - start_time) > time_limit:
                logger.info("Time limit reached.")
                break

        logger.info("QuickTuner fit complete.")

        return self

    def _validate_init_kwargs(self, kwargs: dict) -> None:
        """
        Validate and set additional keyword arguments passed to the constructor.

        Parameters
        ----------
        kwargs : dict
            The keyword arguments passed to the constructor.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown argument: {key}")
