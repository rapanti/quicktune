import json
import os
import shutil
import time
from typing import Callable, Dict, Optional

import torch
from ConfigSpace import Configuration

from quicktune.config_manager import ConfigManager
from quicktune.data.metaset import MetaSet
from quicktune.optimizers.quickoptimizer import QuickTuneOptimizer
from quicktune.utils import get_dataset_metafeatures


class QuickTuner:
    output_dir: str = "exp"
    task_info: Optional[dict]

    def __init__(
        self,
        config: dict,
        optimizer: QuickTuneOptimizer,
        searchspace: ConfigManager,
        metadataset: MetaSet,
        objective_function: Callable[[dict], dict],
        **kwargs,
    ) -> None:
        self.config = config["experiment"]
        self.optimizer = optimizer
        self.searchspace = searchspace
        self.metadataset = metadataset
        self.task_info = {}
        self.data_path = config["experiment"]["data_path"]

        self.objective_function = objective_function

        self.output_dir = kwargs.pop("output_dir", self.output_dir)
        self.task_info = kwargs.pop("task_info", None)

    def run(
        self,
    ) -> None:
        total_budget = self.config["budget"]

        task_info = self.config

        data_path = self.config["data_path"]
        self.optimizer.metafeatures = get_dataset_metafeatures(data_path)

        perf_curves: dict = {"all_perf": [], "all_cost": []}
        output = os.path.join(self.output_dir, "temp")
        os.makedirs(self.output_dir, exist_ok=True)

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
            print(f"hp_index: {hp_index}, budget: {budget}")

            if str(hp_index) not in perf_curves.keys():
                perf_curves[str(hp_index)] = []

            suggested_config = orig_configs[hp_index]
            ft_config = suggested_config.get_dictionary()

            torch.cuda.empty_cache()
            func_config = {
                "hp_config": ft_config,
                "output": output,
                "experiment": str(hp_index),
                "data_path": self.data_path,
                "budget": budget,
                "task_info": task_info,
            }
            result = self.objective_function(func_config)

            score = result["score"]
            status = result["status"]

            if score == 1 or budget >= 50:
                self.optimizer.converged_configs.append(hp_index)

            perf_curves[str(hp_index)].append(score / 100)
            self.optimizer.observe(hp_index, budget, perf_curves[str(hp_index)])

            perf_curves["all_perf"].append(score / 100)
            perf_curves["all_cost"].append(time.time() - start_time)
            info_dict["observed_ids"].append(int(hp_index))
            info_dict["query_config"].append(str(ft_config))
            info_dict["status"].append(status)

            if score > incumbent:
                incumbent = score
                # save best config
                # move file to output folder
                shutil.copy(
                    os.path.join(output, str(hp_index), "last.pth.tar"),
                    os.path.join(self.output_dir, "best_model.pt"),
                )

            # save info dict
            with open(os.path.join(self.output_dir, "info_dict.json"), "w") as f:
                json.dump(info_dict, f)

            # save curves in output folder
            with open(os.path.join(self.output_dir, "perf_curves.json"), "w") as f:
                json.dump(perf_curves, f)

            done = (time.time() - start_time) > total_budget


def from_cs_to_ft_config(config: Configuration) -> Dict:
    return config.get_dictionary()
