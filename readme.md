# WhisperLive
This module integrates Facebook's denoising technology with OpenAI's Whisper ASR (Automatic Speech Recognition) system to reduce noise in input audio streams, improving transcription accuracy and reducing hallucinations. The module supports multi-threading and can load various Whisper models, including language-specific models from Hugging Face, making it versatile for different transcription needs. The server-client architecture allows users to leverage Whisper's capabilities in their daily tasks.
Features

- Noise Reduction: Utilizes Facebook's denoising algorithm to clean input audio.
- Whisper Integration: Leverages Whisper's robust ASR capabilities for accurate  transcription.
- Multi-Model Support: Easily load different Whisper models, including language-specific models from Hugging Face.
- Server-Client Architecture: Runs Whisper on a server, with a client program for interfacing.
- Real-Time Processing: Supports real-time audio processing for immediate feedback.
- Multi-Threading: Handles multiple requests simultaneously for efficient processing.
- Hot-word: you can specify the hot word in `hotword` file inside the server folder by which it can focus on particular words accoring to you needs
  

## Install Instruction
```shell
pip install git+https://github.com/SYSTRAN/faster-whisper.git
pip install git+https://github.com/bolna-ai/streaming-whisper-server.git

pip install transformers
```
note: dont worry about version conflict

## Create project
```shell
python3 -m CustomWhisper -c <project name> -dn <True,False (for denoise active or deactive)>
```
project have three folder ASR, NoiseWeights, VAD and two file hotwords and Server.py you have to make sure you put model in ASR folder, the server code automatically detect model from ASR folder so please make sure of that, if client send wrong model name then , first model is selected automatically
## Add ASR models
```shell
ct2-transformers-converter --model <model: openai/whisper-tiny> --copy_files preprocessor_config.json --output_dir <output_dir: ASR/whisper_tiny_ct> --quantization float16
```
note: add it on ASR folder
## Run server
```
> cd my-project
> ls
ASR  NoiseWeights hotwords Server.py  VAD
> python3 Server.py -p 9000
```