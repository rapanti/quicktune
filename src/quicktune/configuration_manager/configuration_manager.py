from __future__ import annotations

import os
from typing import List

import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant
from ConfigSpace.read_and_write import json as cs_json

from quicktune.data import metaset


class ConfigurationManager:
    """
    A class for managing configurations and performing preprocessing operations.

    The ConfigurationManager class provides methods for loading configurations from a JSON file,
    sampling configurations from a configuration space, preprocessing configurations by encoding
    categorical and numerical hyperparameters, generating unique identifiers for configurations,
    and more.

    Args:
        configspace (ConfigurationSpace): The configuration space object.

    Attributes:
        cs (ConfigurationSpace): The configuration space object.

    Methods:
        from_json_file(json_path: str) -> ConfigurationManager:
            Loads a configuration space from a JSON file and returns a ConfigurationManager object.

        sample_configuration(n: int):
            Samples n configurations from the configuration space and returns them.

        get_one_hot_enc_names() -> List[str]:
            Returns a list of names for one-hot encoded hyperparameters.

        preprocess_configurations(configurations: List[Configuration], ms: metaset.MetaSet, standardize: bool = False) -> pd.DataFrame:
            Preprocesses a list of configurations by encoding categorical and numerical hyperparameters,
            and optionally standardizing numerical hyperparameters.

        config_id(config: Configuration) -> str:
            Generates a unique identifier for a configuration object.
    """

    def __init__(self, configspace: ConfigurationSpace):
        self.cs = configspace

    @classmethod
    def from_json_file(cls, json_path: str) -> ConfigurationManager:
        """
        Creates a ConfigurationManager object from a JSON file.

        Args:
            json_path (str): The path to the JSON file.

        Returns:
            ConfigurationManager: The created ConfigurationManager object.

        Raises:
            AssertionError: If the specified JSON file does not exist.

        """
        assert os.path.isfile(json_path), f"File not found: {json_path}"
        with open(json_path, "r") as f:
            configspace = cs_json.read(f.read())
        return cls(configspace)

    def sample_configuration(self, n: int):
        """
        Samples n configurations from the configuration space.

        Args:
            n (int): The number of configurations to sample.

        Returns:
            List[Configuration]: A list of sampled configurations.
        """
        return self.cs.sample_configuration(n)

    def get_one_hot_enc_names(self) -> List[str]:
        """
        Returns a list of names for one-hot encoded hyperparameters.

        This method iterates over the hyperparameters in the configuration space
        and generates names for one-hot encoded hyperparameters. For categorical
        hyperparameters, it adds a one-hot encoding for each choice. For conditional
        hyperparameters, it adds a not-active flag. For numerical, boolean, and
        ordinal hyperparameters, it adds the hyperparameter name as is.

        Returns:
            A list of names for one-hot encoded hyperparameters.
        """
        hp_names = []
        for hp, val in self.cs.items():
            # skip constant hyperparameters
            if isinstance(val, Constant):
                continue
            elif isinstance(val, CategoricalHyperparameter):
                # add a one-hot encoding for categorical hyperparameters
                for choice in val.choices:
                    enc_hp = f"cat:{hp}:{choice}"
                    hp_names.append(enc_hp)
                # add a not-active flag for conditional hyperparameters
                if hp in self.cs.get_all_conditional_hyperparameters():
                    enc_hp = f"cat:{hp}:None"
                    hp_names.append(enc_hp)
            else:
                # numerical, boolean and ordinal hyperparameters
                hp_names.append(hp)
        return hp_names

    def preprocess_configurations(
        self,
        configurations: List[Configuration],
        ms: metaset.MetaSet,
        standardize: bool = False,
    ):
        """
        Preprocesses a list of configurations by encoding categorical and numerical hyperparameters,
        and optionally standardizing numerical hyperparameters.

        Args:
            configurations (List[Configuration]): A list of configurations to preprocess.
            ms (metaset.MetaSet): The metaset containing the configuration metadata.
            standardize (bool, optional): Whether to standardize numerical hyperparameters. Defaults to False.

        Returns:
            pd.DataFrame: The preprocessed configurations as a pandas DataFrame.
        """
        hp_names = self.get_one_hot_enc_names()

        encoded_configs = []
        for config in configurations:
            encoded_dict = {}
            for hp in hp_names:
                # categorical hyperparameters
                if hp.startswith("cat:"):
                    _, key, value = hp.split(":")
                    val = 1 if config.get(key) == value else 0
                else:
                    # numerical hyperparameters
                    val = config.get(hp, -1)
                    # boolean hyperparameters
                    if isinstance(val, bool):
                        val = int(val)
                    # not-active (conditional or numerical) hyperparameters
                    elif val == "None":
                        val = -1
                encoded_dict[hp] = val
            encoded_configs.append(encoded_dict)
        df = pd.DataFrame(encoded_configs)

        # reorder columns to match the order of the metaset
        df = df[ms.configs_df.columns]

        # standardize numerical hyperparameters
        if standardize:
            cols = ms.num_args_mean.index
            df[cols] = (df[cols] - ms.num_args_mean) / ms.num_args_std

        return df

    def config_id(self, config: Configuration) -> str:
        """
        Generates a unique identifier for a configuration object.

        Args:
            config (Configuration): The configuration object.

        Returns:
            str: The unique identifier for the configuration object.
        """
        return str(hash(frozenset(config.get_dictionary().items())))

    def _get_num_hp_to_std(self) -> List[str]:
        """
        Returns the list of (numerical) hyperparameters that can be standardized.

        Returns:
            List[str]: The list of hyperparameters to standardize.

        num_hp = []
        for hp, val in self.cs.items():
            # TODO ConfigSpace does not save meta information to json
            if val.meta is not None and val.meta.get("standardize", False):
                num_hp.append(hp)
        return num_hp
        """
        raise NotImplementedError
            
