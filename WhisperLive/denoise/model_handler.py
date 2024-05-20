from queue import Queue
from typing import Dict, Any, List
import torch
from .demucs import LoadModel, Demucs
class ModelHandler:
    def __init__(self,load_no_model:int) -> None:
        self.load_no_model:int = load_no_model
        self.__modelQueue:Dict[Any:Queue] = dict() # websocket : queue
        self.__model:List[Demucs] = [LoadModel() for _ in range(self.load_no_model)]
    def register(self,websocket):
        self.__modelQueue[websocket] = Queue()
        return True
    def push(self,websocket,audio_array):
        pass
    
    