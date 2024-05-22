from uuid import uuid4
import numpy as np
import threading
import json
import websocket
import uuid
from queue import Queue
from websockets.exceptions import *


class BasicWhisperClient:
    def __init__(self,host:str, port:int, model:str) -> None:
        self.ws_url =  f"ws://{host}:{port}"
        self.ws_connection:websocket.WebSocket = websocket.WebSocket()
        self.ws_connection.connect(self.ws_url)
        self.client_id:str = str(uuid.uuid4())
        self.retrive_token= None
        self.recever_task = None
        self.model = model


        self.commited_list:list[str] = []



        self.prev_segment = None
        self.curr_segment = None
        self.seg_ptr = 0
        self.same_data_count = 0


        self.segments_collection_thread:threading.Thread = threading.Thread(target=self.get_segment) 

        self.segments:Queue = Queue()
    def MakeConnectionToServer(self):
        self.ws_connection.send(json.dumps(
            {
                "uid": str(uuid.uuid4()),
                "language": "en",
                "task": "transcribe",
                "model": self.model,
                "use_vad": True
            }
        ))
        self.retrive_token = json.loads(self.ws_connection.recv())
        self.segments_collection_thread.start()

    def __check_server_status(self):
        if self.retrive_token == None:
            return False
        elif self.retrive_token["message"] == "SERVER_READY":
            return True
        return False
    
    def send_data_chunk(self,chunk:bytes):
        print("send the chunk")
        self.ws_connection.send(chunk,websocket.ABNF.OPCODE_BINARY)
    

    def CloseConnectionToServer(self):
        self.ws_connection.close()
    
    def SendEOS(self):
        self.ws_connection.send(b'END_OF_AUDIO',websocket.ABNF.OPCODE_BINARY)
        return self.ws_connection.recv()
    
    def SendEnd(self):
        self.SendEOS()
        self.CloseConnectionToServer()
    
    def AddComited(self, segments):
        if len(segments) > 1 and len(segments) - self.seg_ptr >= 2:
            self.commited_list.append(segments[self.seg_ptr]['text'])
            segments[self.seg_ptr]["is_final"] = True
            self.onTranscript(segments[self.seg_ptr])
            self.seg_ptr += 1
        return segments

            

    def AddAttributes(self,segments:dict):
        segments_list = [seg for seg in segments['segments']]

        for i,seg in enumerate(segments_list):
            if seg['text'] in self.commited_list:
                seg["is_final"] = True
            else:
                seg["is_final"] = False
        return segments_list


        
    
    def get_segment(self):
        while True:
            try:
                print("receverd some thing")
                __data = self.ws_connection.recv()
                print(__data)
                data:dict = json.loads(__data)
                if "message" not in data:
                    # self.segments.put(data)
                    data = self.AddAttributes(data)
                    
                    data = self.AddComited(data)

                    if self.curr_segment == None:
                        self.curr_segment = data
                    else:
                        self.prev_segment = self.curr_segment
                        self.curr_segment = data
                    print(data)
                else:
                    print(data)
                    if data['message'] == 'DISCONNECT':
                        self.ws_connection.close()
                        # self.onDisconnect()
                        break
                    elif data['message'] == "UTTERANCE_END":
                        if self.prev_segment != None:
                            if len(self.prev_segment) > 0:
                                self.prev_segment[-1]['is_final'] = True
                        # note make this changes
                    elif data['message'] == 'SERVER_READY':
                        print("server id ready")
                    

            except Exception as e:
                import traceback
                print(traceback.format_exc())
                print(f"rcever stoped {e}")
                break
    
         
    

    def onTranscript(self,segment:dict):
        pass
