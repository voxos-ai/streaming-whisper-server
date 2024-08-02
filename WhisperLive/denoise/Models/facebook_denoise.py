from .base import Denoise
from WhisperLive.denoise.demucs import Demucs, LoadModel
import torch as t

class FaceBookDenoise(Denoise):
    def __init__(self) -> None:
        "this model is input and output were same"
        super().__init__("FaceBook", 16000, 16000)
        self.model = LoadModel()
        self.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
        self.model.to(self.device)
    def infrence(self, chunk: t.Tensor) -> t.Tensor:
        chunk = chunk.to(self.device)
        with t.no_grad():
            # output shape: 1, None
            return self.model(chunk[None])[0]
