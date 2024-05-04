from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Constant,
    CategoricalHyperparameter,
)


class MetaSpace:
    hot_encoding: list[str]
    n_cat_hps: int
    n_oth_hps: int

    def __init__(self, configspace: ConfigurationSpace):
        self.space = configspace

        self._set_hot_encoding()

    def sample_configuration(self, n: int):
        """
        Samples n configurations from the configuration space.

        Args:
            n (int): The number of configurations to sample.

        Returns:
            List[Configuration]: A list of sampled configurations.
        """
        return self.space.sample_configuration(n)

    def get_hot_encoding(self):
        return self.hot_encoding

    def get_hot_encoded_config(self, configs: list[Configuration]):
        """
        One-hot encodes a list of configurations.

        Args:
            configs (List[Configuration]): A list of configurations.

        Returns:
            List[List[int]]: A list of one-hot encoded configurations.
        """
        enc_configs = []
        for config in configs:
            enc_config = dict()
            for hp in self.hot_encoding:
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
                enc_config[hp] = val
            enc_configs.append(enc_config)
        return enc_configs

    def _set_hot_encoding(self):
        """
        Returns a list of names for one-hot encoded hyperparameters.
        """
        cat, other = [], []
        for hp, val in self.space.items():
            # skip constant hyperparameters
            if isinstance(val, Constant):
                continue
            elif isinstance(val, CategoricalHyperparameter):
                # add a one-hot encoding for categorical hyperparameters
                for choice in val.choices:
                    enc_hp = f"cat:{hp}:{choice}"
                    cat.append(enc_hp)
                # add a not-active flag for conditional hyperparameters
                if hp in self.space.get_all_conditional_hyperparameters():
                    enc_hp = f"cat:{hp}:None"
                    cat.append(enc_hp)
            else:
                # numerical, boolean and ordinal hyperparameters
                other.append(hp)
        cat.sort()
        other.sort()
            
        self.n_cat_hps = len(cat)
        self.n_oth_hps = len(other)

        self.hot_encoding = cat + other

    def get_hyperparameter_names(self):
        return list(self.space.keys())
