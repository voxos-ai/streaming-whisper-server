from flask import Flask, request, jsonify
import threading
from WhisperLive.whisper_live.transcriber import WhisperModel
import ctypes
import time
import asyncio
import torch
import uuid
from .model_load_map import ModelStore


class WarmUPService(threading.Thread):
    def __init__(self,model_hash_table:ModelStore,model_list:list,port=6700,host="0.0.0.0") -> None:
        threading.Thread.__init__(self)
        # THREAD NAME
        self.name = "web server"

        # model list in ASR
        # todo cvt into hashset
        self.model_list = model_list
        self.model_hash_table = model_hash_table

        # SERVER VAR's
        self.port = port
        self.host = host
        
        # flask application
        self.app = Flask(__name__)
        
        # device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # rotuter
        self.app.route("/",methods=['GET'])(self.heartbeat)
        self.app.route("/load-model/",methods=['POST'])(self.load_model)
    

    
    def get_id(self):
        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id
            


    # this function is use to close running thread
    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
              ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')
    # utils
    def load_model_memory(self,model_name:str):
        model_id = str(uuid.uuid4())
        try:
            model = WhisperModel(
                    model_size_or_path=f"./ASR/{model_name}",
                    device=self.device,
                    compute_type="int8" if self.device == "cpu" else "float16",
                    local_files_only=False,
                )
            self.model_hash_table.add(model_id,model)
            return model_id
        except Exception as e:
            return None
    # routes
    async def heartbeat(self):
        # for checking server is online or not
        return f"{time.time()}"
    
    async def load_model(self):
        """
        json
        
        model-name: <>
        mode: <cpu,gpu,auto>
        """
        payload = request.get_json()
        try:
            # fix all the mess, below
            model_name = payload["model-name"]
            if model_name in self.model_list:
                
                model_id = await asyncio.to_thread(self.load_model_memory,model_name)
                if model_id:
                    return jsonify({
                        "model-id":model_id,
                        "status": "succeful"
                    })
            return jsonify({
                "model-id":None,
                "status": "error"
            })
        except KeyError:
            return jsonify({
                "model-id":None,
                "status": "error"
            })
    
    # main run method
    def run(self):
        self.app.run(host=self.host,port=self.port)