from WhisperLive import BasicWhisperClient
import numpy as np
import pyaudio
import logging
import time
import requests




res = requests.post("http://127.0.0.1:6700/load-model/",json={
    "model-name":"tiny",
    "mode":"auto"
})

print(res.status_code)
model = res.json()["model-id"]
print(model)
class Client(BasicWhisperClient):
    def __init__(self, host: str, port: int) -> None:
        super().__init__(host, port, model)
    def onTranscript(self, segment: dict):
        super().onTranscript(segment)
        print(segment)
__ = time.time()
client = Client("127.0.0.1",9001)
client.MakeConnectionToServer()
print(client.retrive_token)
print(f"TIME FOR WEBSOCKET CONNECTION: {time.time()- __}")


def bytes_to_float_array(audio_bytes):
    raw_data = np.frombuffer(buffer=audio_bytes, dtype=np.int16)
    return raw_data.astype(np.float32) / 32768.0

chunk = 8192
format = pyaudio.paInt16
channels = 1
rate = 16000
record_seconds = 60000
frames = b""
p = pyaudio.PyAudio()

stream = p.open(
            format=format,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=chunk
        )
try:
    for _ in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk, exception_on_overflow=False)
        audio_array = bytes_to_float_array(data)
        try:
            client.send_data_chunk(audio_array.tobytes())
        except Exception as e:
            print(e)
            break

except KeyboardInterrupt:
    print(client.SendEOS())