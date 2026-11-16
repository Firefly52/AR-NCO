"""The same Full AR architecture with the historical identity output path."""
from torch import nn
from TSPModelFullAR import TSPModel as CurrentTSPModel


class TSPModel(CurrentTSPModel):
    def __init__(self, **model_params):
        super().__init__(**model_params)
        self.encoder.final_norm = nn.Identity()
