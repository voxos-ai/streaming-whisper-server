# CustomWhisper
this package is combination whisper live and denoise packages, 

## Install Instruction
```shell
pip install git+https://github.com/bolna-ai/streaming-whisper-server.git

pip install transformers
```
note: dont worry about version conflict

## Create project
```shell
python3 -m CustomWhisper -c <project name> -dn <True,False (for denoise active or deactive)>
```
## Add ASR models
```shell
ct2-transformers-converter --model <model: openai/whisper-tiny> --copy_files preprocessor_config.json --output_dir <output_dir: ASR/whisper_tiny_ct> --quantization float16
```
## Run server
```
> cd my-project
> ls
ASR  NoiseWeights  Server.py  VAD
> python3 Server.py -p 9000
```