from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class FEConfig:
    """
    Configuration for the feature extractor

    Args:
        in_dim (Optional[int]): The input dimension. If None, the input dimension is inferred from the data.
        hidden_dim (int): The hidden dimension of the feature extractor.
        in_curves_dim (int): The input dimension of the curves.
        out_curves_dim (int): The output dimension of the curves.
        in_metafeatures_dim (Optional[int]): The input dimension of the metafeatures. If None, the input dimension is inferred from the data.
        out_metafeatures_dim (int): The output dimension of the metafeatures.
        encoder_num_layers (int): The number of layers in the encoder.
        encoder_dim_ranges (Optional[List[Tuple[int, ]]): The ranges of the dimensions of the encoder.
    """

    in_dim: Optional[int] = None
    hidden_dim: int = 32
    in_curves_dim: int = 1
    out_curves_dim: int = 16
    in_metafeatures_dim: Optional[int] = None
    out_metafeatures_dim: int = 16
    encoder_num_layers: int = 1
    encoder_dim_ranges: Optional[List[Tuple[int,]]] = None


@dataclass
class CPConfig:
    """
    Configuration for the feature extractor

    Args:
        in_dim (Optional[int]): The input dimension. If None, the input dimension is inferred from the data.
        hidden_dim (int): The hidden dimension of the feature extractor.
        in_curves_dim (int): The input dimension of the curves.
        out_curves_dim (int): The output dimension of the curves.
        in_metafeatures_dim (Optional[int]): The input dimension of the metafeatures. If None, the input dimension is inferred from the data.
        out_metafeatures_dim (int): The output dimension of the metafeatures.
        encoder_num_layers (int): The number of layers in the encoder.
        encoder_dim_ranges (Optional[List[Tuple[int, ]]): The ranges of the dimensions of the encoder.
    """

    in_dim: Optional[int] = None
    hidden_dim: int = 32
    in_curves_dim: int = 1
    out_curves_dim: int = 16
    in_metafeatures_dim: Optional[int] = None
    out_metafeatures_dim: int = 16
    encoder_num_layers: int = 1
    encoder_dim_ranges: Optional[List[Tuple[int,]]] = None


@dataclass
class SurrogateConfig:
    """
    Configuration for the surrogate model
    """

    num_configs: int = 100
    fantasize_steps: int = 1
    acq_func: str = "ei"
    explore_factor: float = 0.0
    fe: FEConfig = FEConfig()
    cp: CPConfig = CPConfig()


@dataclass
class MTLBM_SurrogateConfig(SurrogateConfig):
    """
    Configuration for the surrogate model
    """

    num_configs: int = 100
    fantasize_steps: int = 1
    acq_func: str = "ei"
    explore_factor: float = 0.0
    fe: FEConfig = FEConfig()
    cp: CPConfig = CPConfig()
