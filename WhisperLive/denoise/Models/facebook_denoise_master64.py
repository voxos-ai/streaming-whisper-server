from .base import Denoise
from WhisperLive.denoise.demucs import Demucs, LoadModel
import torch as t
import os




class FaceBookDenoiseM64(Denoise):
    def __init__(self) -> None:
        "this model is input and output were same"
        super().__init__("FaceBook", 16000, 16000)
        self.path = "./NoiseWeights/master64.th"
        self.url = "https://dl.fbaipublicfiles.com/adiyoss/denoiser/master64-8a5dfb4bb92753dd.th"
        if not os.path.isfile(self.path):
            #download logic
            self.download_model(self.url,self.path)
            
        self.model = LoadModel(self.path,hidden=64)
        self.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
        self.model.to(self.device)
        self._init(**{"device":self.device})
    def infrence(self, chunk: t.Tensor) -> t.Tensor:
        chunk = chunk.to(self.device)
        with t.no_grad():
            # output shape: 1, None
            return self.model(chunk[None])[0]
