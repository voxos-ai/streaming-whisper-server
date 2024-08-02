from WhisperLive.logger_config import configure_logger
import torch as t
import numpy as np
import torchaudio



logger = configure_logger(__name__)

class Denoise:
    def __init__(self,model:str,inp_rate:int,model_rate:int) -> None:
        self.model = model
        self.inp_rate = inp_rate
        self.out_rate = 16000
        self.model_rate = model_rate
    
    def convert_sample_rate_IA2MD(self,chunk:np.ndarray) -> t.Tensor:
        "this function were use to convert input audio to req. sample rate audio need by denoise model"
        chunk = chunk.reshape((1,chunk.shape[0])).copy()
        chunk:t.Tensor = t.from_numpy(chunk)
        if self.inp_rate == self.model_rate:
            return chunk
        else:
            return torchaudio.functional.resample(chunk,self.inp_rate,self.model_rate)
    def convert_sample_rate_MD2OA(self,chunk:t.Tensor) -> np.ndarray:
        "this function were use to convert denoise audio to req. sample rate which is 16000 "
        if self.model_rate != self.out_rate:
            chunk = torchaudio.functional.resample(chunk,self.model_rate,self.out_rate)
        logger.info(f"DISPATCH AUDIO SIZE: {chunk.shape}")
        chunk = chunk.reshape(chunk.shape[1])
        return chunk.cpu().detach().numpy()
    def infrence(self,chunk:t.Tensor) -> t.Tensor:
        raise "NOT IMPLEMENTED"
    
    def __call__(self, audio:np.ndarray) -> np.ndarray:
        audio = self.convert_sample_rate_IA2MD(audio)
        audio = self.infrence(audio)
        return self.convert_sample_rate_MD2OA(audio)