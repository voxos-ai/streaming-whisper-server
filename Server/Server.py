from WhisperLive import TranscriptionServer
from WhisperLive.denoise.Models import Denoisers
import argparse
import os

if __name__ == "__main__":
    
    ASRs = [i.name for i in os.scandir("./ASR") if i.is_dir()]
    with open("./hotwords",'r') as file:
        hotwords = [i for i in file.readlines() if (i.strip()) != 0]
    if len(hotwords) <= 0:
        hotwords = None
    denoise_model = [name for name in Denoisers.keys()]
    parser = argparse.ArgumentParser()
    parser.add_argument('--denoise', '-deno',
                        type=str,
                        default="FaceBookDenoise",
                        help=f"denoise models {denoise_model}")
    parser.add_argument('--port', '-p',
                        type=int,
                        default=9090,
                        help="Websocket port to run the server on.")
    args = parser.parse_args()
    if args.denoise in denoise_model:
        server = TranscriptionServer(use_vad=True,denoise=True,denoise_model=args.denoise,hotwords=hotwords,model_list=ASRs,no_speech_prob=0.45)
        server.run(
            "0.0.0.0",
            port=args.port)
    else:
        print(f"models for denoise {denoise_model}")