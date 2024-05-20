from WhisperLive import TranscriptionServer
import argparse
import os

if __name__ == "__main__":
    
    ASRs = [i.name for i in os.scandir("./ASR") if i.is_dir]
    with open("./hotwords",'r') as file:
        hotwords = [i for i in file.readlines() if (i.strip()) != 0]
    if len(hotwords) <= 0:
        hotwords = None
    server = TranscriptionServer(use_vad=True,denoise=True,hotwords=hotwords,model_list=ASRs)
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p',
                        type=int,
                        default=9090,
                        help="Websocket port to run the server on.")
    args = parser.parse_args()
    server.run(
        "0.0.0.0",
        port=args.port)