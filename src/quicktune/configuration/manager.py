from typing import List

import pandas as pd
from ConfigSpace import Configuration

from quicktune.configuration.space import MetaSpace
from quicktune.data.metaset import MetaSet


class ConfigManager:
    """
    A class for defining the configuration space, managing configurations and
    performing preprocessing operations for the meta-learning framework.

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

    _one_hot_enc_names: list[str] = []

    def __init__(
        self,
        space: MetaSpace,
        metaset: MetaSet,
    ):
        self.space = space
        self.ms = metaset

    def sample_configuration(self, n: int):
        """
        Samples n configurations from the configuration space.

        Args:
            n (int): The number of configurations to sample.

        Returns:
            List[Configuration]: A list of sampled configurations.
        """
        return self.space.sample_configuration(n)

    def preprocess_configurations(
        self,
        configurations: List[Configuration],
        standardize: bool = True,
    ) -> pd.DataFrame:
        """
        Preprocesses a list of configurations by encoding categorical and numerical hyperparameters,
        and optionally standardizing numerical hyperparameters.

        Args:
            configurations (List[Configuration]): A list of configurations to preprocess.
            standardize (bool, optional): Whether to standardize numerical hyperparameters. Defaults to False.

        Returns:
            pd.DataFrame: The preprocessed configurations as a pandas DataFrame.
        """
        encoded_configs = self.space.get_hot_encoded_config(configurations)

        df = pd.DataFrame(encoded_configs)

        # reorder columns to match the order of the metaset
        df = df[self.ms.configs_df.columns]

        # standardize numerical hyperparameters
        if standardize:
            cols = self.ms.num_hp_mean.index
            df[cols] = (df[cols] - self.ms.num_hp_mean) / self.ms.num_hp_std

        return df

    @staticmethod
    def config_id(config: Configuration) -> str:
        """
        Generates a unique identifier for a configuration object.

        Args:
            config (Configuration): The configuration object.
        Returns:
            str: The unique identifier for the configuration object.
        """
        return str(hash(frozenset(config.get_dictionary().items())))
