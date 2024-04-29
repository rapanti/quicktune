import json
import logging
import os
import shutil
import time
from typing import Callable, Optional

from quicktune.factory import setup_quicktune
from quicktune.optimizers.quickoptimizer import QuickTuneOptimizer
from quicktune.utils.log_utils import add_log_to_file, set_logger_verbosity
from quicktune.utils.qt_utils import get_dataset_metafeatures, QTunerResult
from quicktune.utils.utils import setup_outputdir

logger = logging.getLogger(__name__)


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
        config: dict,
        objective_function: Callable[[dict], QTunerResult],
        path: Optional[str] = None,
        verbosity: int = 4,
        **kwargs,
    ):
        self.verbosity = verbosity
        set_logger_verbosity(verbosity, logger)
        self.path = setup_outputdir(path)
        self._setup_log_to_file(self._log_to_file, self._log_file_path)

        self._validate_init_kwargs(kwargs)

        config.update(
            verbosity=verbosity,
            output_path=self.path,
        )
        print(f"config: {config}")

        self.optimizer: QuickTuneOptimizer = setup_quicktune(config)
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

        metafeat = get_dataset_metafeatures(data_path)
        self.optimizer.set_metafeatures(**metafeat)

        task_info = {
            "train-split": train_split,
            "val-split": val_split,
            "num_classes": metafeat["num_classes"],
        }

        perf_curves: dict = {"all_perf": [], "all_cost": []}
        output_dir = os.path.join(self.path, "temp")
        os.makedirs(self.path, exist_ok=True)

        orig_configs = self.optimizer.sampled_configs

        info_dict = {
            "all_configs": [dict(config) for config in orig_configs],
            "observed_ids": [],
            "query_config": [],
            "status": [],
        }

        incumbent = 0
        start_time = time.time()
        done = False
        while not done:
            hp_index, budget = self.optimizer.suggest()
            logger.info(f"hp_index: {hp_index}, budget: {budget}")

            if str(hp_index) not in perf_curves.keys():
                perf_curves[str(hp_index)] = []

            suggested_config = orig_configs[hp_index]
            ft_config = suggested_config.get_dictionary()
            # print(f"ft_config: {ft_config}")

            func_config = {
                "hp_config": ft_config,
                "output": output_dir,
                "experiment": str(hp_index),
                "data_path": data_path,
                "budget": budget,
                "task_info": task_info,
            }
            result = self.objective_function(func_config)
            print(result)

            score = result.score
            status = str(result.status)
            self.optimizer.observe(hp_index, budget, result)

            perf_curves[str(hp_index)].append(score)
            perf_curves["all_perf"].append(score)
            perf_curves["all_cost"].append(time.time() - start_time)
            info_dict["observed_ids"].append(int(hp_index))
            info_dict["query_config"].append(str(ft_config))
            info_dict["status"].append(status)

            if score > incumbent:
                incumbent = score
                # save best config
                # move file to output folder
                shutil.copy(
                    os.path.join(output_dir, str(hp_index), "last.pth.tar"),
                    os.path.join(self.path, "best_model.pt"),
                )

            # save info dict
            with open(os.path.join(self.path, "info_dict.json"), "w") as f:
                json.dump(info_dict, f)

            # save curves in output folder
            with open(os.path.join(self.path, "perf_curves.json"), "w") as f:
                json.dump(perf_curves, f)

            done = (time.time() - start_time) > time_limit
        
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
