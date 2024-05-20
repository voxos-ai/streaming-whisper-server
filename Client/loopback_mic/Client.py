import soundcard as sc
import soundfile as sf
from WhisperLive import BasicWhisperClient
import numpy as np
from utils import write_bytesIO
from scipy.io.wavfile import write,read
import uuid
import io
import torchaudio


OUTPUT_FILE = "out.srt"
class Client(BasicWhisperClient):
    def __init__(self, host: str, port: int) -> None:
        super().__init__(host, port, "tiny")
        self.__segments:list = []
        self.flush_threshold = 3
    def flush_on_file(self,segment):
        print(segment)
        self.__segments.append(segment)
        if len(self.__segments) >  self.flush_threshold:
            with open(OUTPUT_FILE,'a') as file:
                segments = [f"{seg['start']}~{seg['end']}: {seg['text']}" for seg in self.__segments]
                file.write("\n".join(segments))

    def onTranscript(self, segment: dict):
        super().onTranscript(segment)
        print(segment)
        # print(f"{segment['start']}~{segment['end']}: {segment['text']}")
        self.flush_on_file(segment)

client = Client("127.0.0.1",9000)
client.MakeConnectionToServer()
print(client.retrive_token)


def bytes_to_float_array(audio_bytes):
    # raw_data = np.frombuffer(buffer=audio_bytes, dtype=np.int16)
    return audio_bytes.astype(np.float32) / 32768.0


SAMPLE_RATE = 16000              
RECORD_SEC_CHUNK = 3
RECORD_SEC = 8000                

for _ in range(int(RECORD_SEC * RECORD_SEC_CHUNK)):
    with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE,channels=1) as mic:
        # record audio with loopback from default speaker.
        data:np.ndarray = mic.record(numframes=SAMPLE_RATE*RECORD_SEC_CHUNK)

        # audio_array = bytes_to_float_array(data[0])
        # print(data)
        file = write_bytesIO(16000,data)
        sr, wav = read(file)
        try:
            client.send_data_chunk(wav.tobytes())
        except Exception as e:
            print(e)
            break
    