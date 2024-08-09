from threading import Lock
from WhisperLive.whisper_live.transcriber import WhisperModel
import torch


class ModelStore:
    def __init__(self) -> None:
        self.__memory = dict()
        self.__lock = Lock()
    def get(self,key:str)->WhisperModel:
        with self.__lock:
            return self.__memory.get(key)
    def pop(self,key:str)->bool:
        with self.__lock:
            if key in self.__memory.keys():
                model =  self.__memory.pop(key)
                del model
                # we use ctranslate2 so that we didn't need this
                # torch.cuda.empty_cache()
                return True
            return False
    def add(self,key:str,model:WhisperModel)->bool:
        with self.__lock:
            if key not in self.__memory.keys():
                self.__memory[key] = model
                return True
            return False
