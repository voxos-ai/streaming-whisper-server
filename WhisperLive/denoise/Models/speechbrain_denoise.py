import torch as t
from speechbrain.inference.separation import SepformerSeparation as separator
from .base import Denoise
class SpeechBrainDenoise(Denoise):
    def __init__(self) -> None:
        super().__init__("SpeechBrain", 16000, 16000)
        self.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
        self.model = separator.from_hparams(source="speechbrain/sepformer-dns4-16k-enhancement")
        self.model.device = self.device
        self.model.to(self.device)
        self._init(**{"device":self.device})
    def infrence(self, chunk: t.Tensor) -> t.Tensor:
        chunk = chunk.to(self.device)
        chunk = self.model.separate_batch(chunk)
        return chunk.cpu()[:, :, 0]
