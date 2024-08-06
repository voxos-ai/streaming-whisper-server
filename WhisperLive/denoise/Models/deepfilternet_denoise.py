import torch as t
from df.enhance import enhance, init_df
from torch._tensor import Tensor
from .base import Denoise

class DeepFilterNetDenoise(Denoise):
    def __init__(self) -> None:
        #TODO: fix the input sample rate, i fix to 16000
        super().__init__("DeepFilterNet", 16000, 48000)
        self.model, self.df_state, _ = init_df()
        self._init()
    def infrence(self, chunk: t.tensor) -> Tensor:
        return enhance(self.model, self.df_state, chunk)