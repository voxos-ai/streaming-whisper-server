from WhisperLive.logger_config import configure_logger
import torch as t
import numpy as np
import torchaudio
import time
import requests
from tqdm import tqdm


logger = configure_logger(__name__)

class Denoise:
    def __init__(self,model:str,inp_rate:int,model_rate:int) -> None:
        self.model = model
        self.inp_rate = inp_rate
        self.out_rate = 16000
        self.model_rate = model_rate
        self._time = time.time()
    
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
        "have to return have [1,None]"
        raise "NOT IMPLEMENTED"
    
    def __call__(self, audio:np.ndarray) -> np.ndarray:
        
        audio = self.convert_sample_rate_IA2MD(audio)
        __ = time.time()
        audio = self.infrence(audio)
        logger.info(f"TRNSCRIBE TIME: {time.time() - __}")
        ret = self.convert_sample_rate_MD2OA(audio)
        
        return ret
    def _init(self,**kwags):
        logger.info(f"MODEL LOADING TIME: {time.time() - self._time}")
        logger.info(f"INFO: {kwags}")
    
    def download_model(self,url:str,path:str):
        "NOTE: location context inside the server Folder"
        #download logic
        response = requests.get(url, stream=True)

        # Sizes in bytes.
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024

        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            with open(path, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError("Could not download file")