import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--create_project', '-c',
                    type=str,
                    help="enter the name of project")
parser.add_argument('--whisper_model_name', '-wmn',
                    type=str, default="openai/whisper-tiny",
                    help="Custom Faster Whisper Model")
args = parser.parse_args()

if __name__ == "__main__":
    os.mkdir(args.create_project)
    os.mkdir(f"{args.create_project}/LLM")
    os.mkdir(f"{args.create_project}/NoiseWeights")
    os.mkdir(f"{args.create_project}/VAD")
    model_name = args.whisper_model_name.split("/")[-1]
    out_dir = f"{args.create_project}/LLM/{model_name}"
    os.system(f"ct2-transformers-converter --model {args.whisper_model_name} --copy_files preprocessor_config.json --output_dir {out_dir} --quantization float16")
    os.system(f"wget https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns48-11decc9d8e3f0998.th -O {args.create_project}/NoiseWeights/model.th")
    os.system(f"wget https://github.com/anshjoseph/WhisperCustom/raw/master/example/SimpleExample/VAD/silero_vad.onnx -O {args.create_project}/VAD/silero_vad.onnx")
    print("HERE IS BASIC SETUP OF FOLDER TO RUN THIS MODULE IN SERVER SIDE")