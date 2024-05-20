import argparse
import os

simple_server = """
from CustomWhisper import TranscriptionServer
import argparse

if __name__ == "__main__":
    server = TranscriptionServer(use_vad=True,denoise=False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p',
                        type=int,
                        default=9090,
                        help="Websocket port to run the server on.")
    parser.add_argument('--faster_whisper_custom_model_path', '-fw',
                        type=str, default=None,
                        help="Custom Faster Whisper Model")
    args = parser.parse_args()
    server.run(
        "0.0.0.0",
        port=args.port,
        backend="faster_whisper",
        faster_whisper_custom_model_path=args.faster_whisper_custom_model_path
    )
"""
denoise_server = """
from CustomWhisper import TranscriptionServer
import argparse

if __name__ == "__main__":
    server = TranscriptionServer(use_vad=True,denoise=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p',
                        type=int,
                        default=9090,
                        help="Websocket port to run the server on.")
    parser.add_argument('--faster_whisper_custom_model_path', '-fw',
                        type=str, default=None,
                        help="Custom Faster Whisper Model")
    args = parser.parse_args()
    server.run(
        "0.0.0.0",
        port=args.port,
        backend="faster_whisper",
        faster_whisper_custom_model_path=args.faster_whisper_custom_model_path
    )
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
    with open(f"{args.create_project}/Server.py",'w') as file:
        if args.denoise:
            file.write(denoise_server)
        else:
            file.write(denoise_server)    
    # os.system(f"ct2-transformers-converter --model {args.whisper_model_name} --copy_files preprocessor_config.json --output_dir {out_dir} --quantization float16")
    os.system(f"wget https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns48-11decc9d8e3f0998.th -O {args.create_project}/NoiseWeights/model.th")
    os.system(f"wget https://github.com/anshjoseph/WhisperCustom/raw/master/example/SimpleExample/VAD/silero_vad.onnx -O {args.create_project}/VAD/silero_vad.onnx")
    print("HERE IS BASIC SETUP OF FOLDER TO RUN THIS MODULE IN SERVER SIDE")
    print(f"use this command to load ASR: ct2-transformers-converter --model openai/whisper-tiny --copy_files preprocessor_config.json --output_dir /ASR/whisper-tiny --quantization float16")