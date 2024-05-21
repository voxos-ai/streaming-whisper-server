import argparse
import os

simple_server = """
from WhisperLive import TranscriptionServer
import argparse
import os

if __name__ == "__main__":
    
    ASRs = [i.name for i in os.scandir("./ASR") if i.is_dir()]
    with open("./hotwords",'r') as file:
        hotwords = [i for i in file.readlines() if (i.strip()) != 0]
    if len(hotwords) <= 0:
        hotwords = None
    server = TranscriptionServer(use_vad=True,denoise=False,hotwords=hotwords,model_list=ASRs)
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p',
                        type=int,
                        default=9090,
                        help="Websocket port to run the server on.")
    args = parser.parse_args()
    server.run(
        "0.0.0.0",
        port=args.port)
"""
denoise_server = """
from WhisperLive import TranscriptionServer
import argparse
import os

if __name__ == "__main__":
    
    ASRs = [i.name for i in os.scandir("./ASR") if i.is_dir()]
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
"""
parser = argparse.ArgumentParser()
parser.add_argument('--create_project', '-c',
                    type=str,
                    help="enter the name of project")
parser.add_argument('--denoise', '-dn',
                    type=bool, default=True,
                    help="Custom Faster Whisper Model")
args = parser.parse_args()

if __name__ == "__main__":
    os.mkdir(args.create_project)
    os.mkdir(f"{args.create_project}/ASR")
    os.mkdir(f"{args.create_project}/NoiseWeights")
    os.mkdir(f"{args.create_project}/VAD")
    with open(f"{args.create_project}/hotwords",'w') as file:
        pass
    with open(f"{args.create_project}/Server.py",'w') as file:
        if args.denoise:
            file.write(denoise_server)
        else:
            file.write(denoise_server)    
    # os.system(f"ct2-transformers-converter --model {args.whisper_model_name} --copy_files preprocessor_config.json --output_dir {out_dir} --quantization float16")
    os.system(f"wget https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns48-11decc9d8e3f0998.th -O {args.create_project}/NoiseWeights/model.th")
    os.system(f"wget https://github.com/SYSTRAN/faster-whisper/raw/master/faster_whisper/assets/silero_vad.onnx -O {args.create_project}/VAD/silero_vad.onnx")
    print("HERE IS BASIC SETUP OF FOLDER TO RUN THIS MODULE IN SERVER SIDE")
    print(f"use this command to load ASR: ct2-transformers-converter --model openai/whisper-tiny --copy_files preprocessor_config.json --output_dir /ASR/whisper-tiny --quantization float16")