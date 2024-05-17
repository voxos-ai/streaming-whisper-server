from CustomWhisper import TranscriptionServer
import argparse

if __name__ == "__main__":
    server = TranscriptionServer()
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