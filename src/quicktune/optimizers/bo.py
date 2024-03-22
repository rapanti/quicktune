import datetime
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch

from quicktune.finetune.utils.eval_autofinetune import eval_finetune_conf
from quicktune.optimizers.quicktune import QuickTuneOptimizer
from quicktune.tools.metadataset import MetaDataset
from quicktune.tools.searchspace import SearchSpace


class BO:
    def __init__(
        self,
        task_info: dict,
        optimizer: QuickTuneOptimizer,
        searchspace: SearchSpace,
        metadataset: MetaDataset,
        data_path: str,
        output_dir: str,
    ) -> None:
        self.task_info = task_info
        self.optimizer = optimizer
        self.searchspace = searchspace
        self.metadataset = metadataset
        self.data_path = data_path
        self.output_dir = output_dir

    def run(self, total_budget: int, num_configurations: int = 400, verbose: bool = False) -> None:
        self.optimizer.model.metafeatures = _get_metafeatures(self.task_info)

        hp_names = self.metadataset.get_hyperparameters_names()

        original_configurations = self.searchspace.sample_configuration(num_configurations)
        configurations = _preprocess_configurations(
            original_configurations, hp_names, self.searchspace, self.metadataset
        )
        self.optimizer.set_hyperparameter_candidates(configurations)

        perf_curves: dict = {"all_perf": [], "all_cost": []}
        temp_output = os.path.join(self.output_dir, "temp")
        os.makedirs(self.output_dir, exist_ok=True)

        info_dict = {
            "all_configs": [dict(config) for config in original_configurations],  # type: ignore
            "observed_ids": [],
            "query_config": [],
            "status": [],
        }

        highest_perf = 0
        start_time = time.time()
        done = False

        while not done:
            hp_index, budget = self.optimizer.suggest()

            if str(hp_index) not in perf_curves.keys():
                perf_curves[str(hp_index)] = []

            selected_hp = configurations[hp_index]
            selected_hp = _postprocess_configurations(selected_hp, self.metadataset, hp_names)
            aft_config = self.optimizer.to_qt_config(selected_hp)

            temp_config_id = f"{hp_index}"
            aft_config = _validate_configuration(aft_config, self.searchspace)

            torch.cuda.empty_cache()
            perf, _, status = eval_finetune_conf(
                aft_config,
                self.task_info,
                budget=budget,
                experiment=temp_config_id,
                data_path=self.data_path,
                output=temp_output,
                verbose=verbose,
                dataset_download=False,
            )

            # if perf == 1:
            #     self.optimizer.converged_configs.append(hp_index)

            perf_curves[str(hp_index)].append(perf / 100)
            self.optimizer.observe(hp_index, budget, perf_curves[str(hp_index)])

            perf_curves["all_perf"].append(perf / 100)
            perf_curves["all_cost"].append(time.time() - start_time)
            info_dict["observed_ids"].append(int(hp_index))
            info_dict["query_config"].append(str(aft_config))
            info_dict["status"].append(status)

            if perf > highest_perf:
                highest_perf = perf
                # save best config
                # move file to output folder
                shutil.copy(
                    os.path.join(temp_output, temp_config_id, "last.pth.tar"),
                    os.path.join(self.output_dir, "best_model.pt"),
                )

            # save info dict
            with open(os.path.join(self.output_dir, "info_dict.json"), "w") as f:
                json.dump(info_dict, f)

            # save curves in output folder
            with open(os.path.join(self.output_dir, "perf_curves.json"), "w") as f:
                json.dump(perf_curves, f)

            done = time.time() - start_time > total_budget


def _preprocess_configurations(configurations, hp_names, ss, metadataset):
    """
    Preprocess the configurations.

    Args:
        configurations: The configurations to preprocess.
        hp_names: The names of the hyperparameters.
        ss: The search space.
        metadataset: The QuickTuneMetaDataset object.

    Returns:
        The preprocessed configurations.
    """
    conf_df = pd.DataFrame(configurations)

    conf_df["batch_size"] = conf_df["batch_size"].apply(lambda x: min(x - x % 2, 256))

    # convert to the hp names type
    data_augmentations = ["trivial_augment", "random_augment"]
    categorical_groups, mapping_to_group = _get_categorical_groups(ss)
    hp_values = []
    default_values = {
        "patience_epochs": 10.0,
        "decay_epochs": 20.0,
        "decay_rate": 0.1,
        "momentum": 0.9,
        "ra_magnitude": 8.0,
        "ra_num_ops": 2.0,
    }

    try:
        for hp in hp_names:
            if hp.startswith("cat__"):
                if hp in mapping_to_group.keys():
                    hp_name, hp_option = mapping_to_group[hp]
                    if hp_name in conf_df.columns:
                        hp_values.append(conf_df[hp_name].apply(lambda x: 1 if x == hp_option else 0).values)
                    else:
                        hp_values.append(np.zeros(conf_df.shape[0]))
                else:
                    hp_values.append(np.zeros(conf_df.shape[0]))
            elif hp in data_augmentations:
                if "data_augmentation" in conf_df.columns:
                    hp_values.append(conf_df["data_augmentation"].apply(lambda x: 1 if x == hp else 0).values)
                # else:
                #    hp_values.append(np.zeros(conf_df.shape[0]))
            else:
                if hp in conf_df.columns:
                    x = conf_df[hp].values
                    x[x == "None"] = -1.0
                    if hp in default_values.keys():
                        x = [default_values[hp] if np.isnan(v) else v for v in x]
                # elif hp in default_values.keys():
                #     x = [default_values[hp] for v in x]
                else:
                    x = np.zeros(conf_df.shape[0])
                x = np.array(x, dtype=np.float32)
                if np.isnan(x).any():
                    print("hp {} is nan".format(hp))
                hp_values.append(x)
        # convert to the hp names type
        # hp_values = np.concatenate(hp_values, axis=1)
    except Exception as e:
        print(e)
        raise ValueError("The configuration is not valid")
    hp_values = [np.array(x, dtype=np.float32).reshape(1, -1) for x in hp_values]
    hp_values = np.vstack(hp_values).T.round(6)
    # standardize the hp for input to optimizer

    mean_values = metadataset.args_mean[hp_names].values
    std_values = metadataset.args_std[hp_names].values

    for i in range(hp_values.shape[1]):
        if not hp_names[i].startswith("cat") and std_values[i] != 0:
            hp_values[:, i] = (hp_values[:, i] - mean_values[i]) / std_values[i]

    return hp_values


def _postprocess_configurations(x, metadataset, hp_names):
    """
    Postprocess the configurations.

    Args:
        x: The configurations to postprocess.
        metadataset: The QuickTuneMetaDataset object.
        hp_names: The names of the hyperparameters.

    Returns:
        The postprocessed configurations.
    """
    # only one configuration is valid
    mean_values = metadataset.args_mean[hp_names].values.round(6)
    std_values = metadataset.args_std[hp_names].values.round(6)
    x = x.round(6)
    for i in range(x.shape[0]):
        if not hp_names[i].startswith("cat") and std_values[i] != 0:
            x[i] = x[i] * std_values[i] + mean_values[i]

    return x.round(6)


def _get_categorical_groups(search_space):
    """
    Get the categorical groups from the search space.

    Args:
        search_space: The search space.

    Returns:
        The categorical groups and the mapping to group.
    """
    categorical_vars = ["model", "sched", "auto_augment", "opt", "opt_betas"]
    categorical_groups = {}
    mapping_to_group = {}
    for var in categorical_vars:
        if isinstance(search_space.data[var], list):
            names = [f"cat__{var}_{i}" for i in search_space.data[var]]
            categorical_groups[var] = names
            for i, k in enumerate(names):
                mapping_to_group[k] = (var, search_space.data[var][i])
        else:
            if var == "opt_betas":
                names = [f"cat__{var}_[{i}]" for i in search_space.data[var]["options"]]
                names = [name.replace(" ", ", ") for name in names]
                names = [name.replace("0,", "0.0,") for name in names]
            else:
                names = [f"cat__{var}_{i}" for i in search_space.data[var]["options"]]
            categorical_groups[var] = names
            for i, k in enumerate(names):
                mapping_to_group[k] = (var, search_space.data[var]["options"][i])

    return categorical_groups, mapping_to_group


def _get_metafeatures(task_info: dict) -> torch.Tensor:
    num_samples = task_info["num_samples"]
    num_classes = task_info["num_classes"]
    image_size = task_info["image_size"]
    num_channels = task_info["num_channels"]
    return torch.FloatTensor([num_samples, num_classes, image_size, num_channels]).reshape(1, -1) / 10000


def _validate_configuration(configuration, search_space):
    """
    Validate the configuration.

    Args:
        configuration: The configuration to validate.
        search_space: The search space.

    Returns:
        The validated configuration.
    """
    configuration["batch_size"] = min(configuration["batch_size"] + configuration["batch_size"] % 2, 256)

    for hp in configuration.keys():
        data = search_space.__dict__["data"]
        if hp in data.keys():
            values = data[hp].copy()
            current_value = configuration[hp]

            if hp == "clip_grad":
                values.remove("None")

            if isinstance(values, dict):
                values = values["options"]

            if isinstance(current_value, (int, float)):
                if "None" in values:
                    values.remove("None")
                min_hp = min(values)
                max_hp = max(values)
                configuration[hp] = min(max(current_value, min_hp), max_hp)

            elif current_value == "None":
                pass

            elif isinstance(current_value, str):
                if hp != "opt_betas":
                    assert configuration[hp] in values, "The value {} is not in the search space".format(
                        configuration[hp]
                    )

    return configuration
