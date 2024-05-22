import os
import time
import threading
import json
import functools
import torch
import numpy as np
from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosed
from WhisperLive.whisper_live.vad import VoiceActivityDetector
from WhisperLive.whisper_live.transcriber import WhisperModel
from WhisperLive.whisper_live.server import ServeClientBase, ClientManager
from WhisperLive.whisper_live.HypothesisBuffer import HypothesisBufferPrefix
from .denoise import LoadModel, Demucs, BasicInferenceMechanism
from .logger_config import configure_logger


logger = configure_logger(__name__)

class TranscriptionServer:
    RATE = 16000
    @staticmethod
    def bytes_to_float_array(audio:np.ndarray):
        return audio.astype(np.float32) / 32768.0
    
    def __init__(self,use_vad=True,denoise=False,hotwords=None,model_list=[]):
        self.client_manager = ClientManager()
        self.no_voice_activity_chunks = 0
        self.use_vad = use_vad
        self.denoise = denoise
        self.hotwords = hotwords
        self.model_list = model_list
        self.default_model_index = 0
        if model_list == None or len(model_list) <= 0:
            raise("without model list we can't start server")
        
        if self.denoise:
            self.noise_deduction_model:Demucs = LoadModel()
            self.infrence_mech:BasicInferenceMechanism = BasicInferenceMechanism(self.noise_deduction_model)
            
        else:
            self.noise_deduction_model = None
            self.infrence_mech = None
        logger.info("TranscriptionServer is created")

    def initialize_client(self, websocket, options):
        logger.info(options)
        __hotwords = options.get('keywords')
        if __hotwords == None: __hotwords = []
        if options["model"] not in self.model_list:
            logger.info("model name is not  in model list so getting revert to default model")
            model = f"./ASR/{self.model_list[self.default_model_index]}"
        else:
            logger.info("model is detected")
            model =  f"./ASR/{options['model']}"
        
        logger.info(f"loaded model {model}")
        # making the FasterWhisper server
        client:ServeClientFasterWhisper = ServeClientFasterWhisper(
            websocket,
            language=options["language"],
            task=options["task"],
            client_uid=options["uid"],
            model=model,
            initial_prompt=options.get("initial_prompt"),
            vad_parameters=options.get("vad_parameters"),
            use_vad=self.use_vad,
            hotwords=list(set(__hotwords + self.hotwords))
        )
        logger.info("Running faster_whisper backend.")

        self.client_manager.add_client(websocket, client)

    def get_audio_from_websocket(self, websocket):
        """
        Receives audio buffer from websocket and creates a numpy array out of it.

        Args:
            websocket: The websocket to receive audio from.

        Returns:
            A numpy array containing the audio.
        """
        frame_data = websocket.recv()
        if frame_data == b"END_OF_AUDIO":
            return False

        audio = np.frombuffer(frame_data, dtype=np.float32)
        if self.denoise:
            logger.info("denoising voice")
            out = self.infrence_mech(audio)[0]
            logger.info(f"denoising voice {out.shape}")
            return out
        else:
            logger.info(audio)
            return audio

    def handle_new_connection(self, websocket):
        try:
            logger.info("New client connected")
            options = websocket.recv()
            options = json.loads(options)
            self.use_vad = options.get('use_vad')
            if self.client_manager.is_server_full(websocket, options):
                websocket.close()
                return False  # Indicates that the connection should not continue

            # if self.backend == "tensorrt":
            self.vad_detector = VoiceActivityDetector(frame_rate=self.RATE)
            self.initialize_client(websocket, options)
            return True
        
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from client")
            return False
        
        except ConnectionClosed:
            logger.info("Connection closed by client")
            return False
        
        except Exception as e:
            logger.error(f"Error during new connection initialization: {str(e)}")
            return False

    def process_audio_frames(self, websocket):
        frame_np = self.get_audio_from_websocket(websocket)
        client:ServeClientFasterWhisper = self.client_manager.get_client(websocket)
        if frame_np is False:
            client.disconnect()
            return False


        client.add_frames(frame_np)
        return True

    def recv_audio(self,
                   websocket,
                   backend="faster_whisper"):
        """
        Receive audio chunks from a client in an infinite loop.

        Continuously receives audio frames from a connected client
        over a WebSocket connection. It processes the audio frames using a
        voice activity detection (VAD) model to determine if they contain speech
        or not. If the audio frame contains speech, it is added to the client's
        audio data for ASR.
        If the maximum number of clients is reached, the method sends a
        "WAIT" status to the client, indicating that they should wait
        until a slot is available.
        If a client's connection exceeds the maximum allowed time, it will
        be disconnected, and the client's resources will be cleaned up.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.
            backend (str): The backend to run the server with.
            faster_whisper_custom_model_path (str): path to custom faster whisper model.
            whisper_tensorrt_path (str): Required for tensorrt backend.
            trt_multilingual(bool): Only used for tensorrt, True if multilingual model.

        Raises:
            Exception: If there is an error during the audio frame processing.
        """
        logger.info("start receving audio")
        self.backend = backend
        if not self.handle_new_connection(websocket):
            return

        try:
            while not self.client_manager.is_client_timeout(websocket):
                if not self.process_audio_frames(websocket):
                    break
        except ConnectionClosed:
            logger.info("Connection closed by client")
        except Exception as e:
            logger.error(e.with_traceback())
            logger.error(f"Unexpected error: {str(e)}")
        finally:
            if self.client_manager.get_client(websocket):
                self.cleanup(websocket)
                websocket.close()
            del websocket

    def run(self,
            host,
            port=9090):
        """
        Run the transcription server.

        Args:
            host (str): The host address to bind the server.
            port (int): The port number to bind the server.
        """
        with serve(
            functools.partial(
                self.recv_audio,
                backend="faster_whisper",
            ),
            host,
            port
        ) as server:
            logger.info("runing the server")
            logger.info(f"with port no: {port} and host {host}")
            server.serve_forever()

    def voice_activity(self, websocket, frame_np):
        """
        Evaluates the voice activity in a given audio frame and manages the state of voice activity detection.

        This method uses the configured voice activity detection (VAD) model to assess whether the given audio frame
        contains speech. If the VAD model detects no voice activity for more than three consecutive frames,
        it sets an end-of-speech (EOS) flag for the associated client. This method aims to efficiently manage
        speech detection to improve subsequent processing steps.

        Args:
            websocket: The websocket associated with the current client. Used to retrieve the client object
                    from the client manager for state management.
            frame_np (numpy.ndarray): The audio frame to be analyzed. This should be a NumPy array containing
                                    the audio data for the current frame.

        Returns:
            bool: True if voice activity is detected in the current frame, False otherwise. When returning False
                after detecting no voice activity for more than three consecutive frames, it also triggers the
                end-of-speech (EOS) flag for the client.
        """
        if not self.vad_detector(frame_np):
            self.no_voice_activity_chunks += 1
            if self.no_voice_activity_chunks > 3:
                client = self.client_manager.get_client(websocket)
                if not client.eos:
                    client.set_eos(True)
                time.sleep(0.1)    # Sleep 100m; wait some voice activity.
            return False
        return True

    def cleanup(self, websocket):
        """
        Cleans up resources associated with a given client's websocket.

        Args:
            websocket: The websocket associated with the client to be cleaned up.
        """
        if self.client_manager.get_client(websocket):
            self.client_manager.remove_client(websocket)



class ServeClientFasterWhisper(ServeClientBase):
    def __init__(self, websocket, hotwords=None, task="transcribe", device=None, language=None, client_uid=None, model="./LLM/whisper_tiny_ct",
                 initial_prompt=None, vad_parameters=None, use_vad=True):
        """
        Initialize a ServeClient instance.
        The Whisper model is initialized based on the client's language and device availability.
        The transcription thread is started upon initialization. A "SERVER_READY" message is sent
        to the client to indicate that the server is ready.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.
            task (str, optional): The task type, e.g., "transcribe." Defaults to "transcribe".
            device (str, optional): The device type for Whisper, "cuda" or "cpu". Defaults to None.
            language (str, optional): The language for transcription. Defaults to None.
            client_uid (str, optional): A unique identifier for the client. Defaults to None.
            model (str, optional): The whisper model size. Defaults to 'small.en'
            initial_prompt (str, optional): Prompt for whisper inference. Defaults to None.
        """
        super().__init__(client_uid, websocket)
        self.hotwords = hotwords
        self.model_sizes = [
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large-v2", "large-v3",
        ]
        if not os.path.exists(model):
            self.model_size_or_path = self.check_valid_model(model)
        else:
            self.model_size_or_path = model
        self.language = "en" if self.model_size_or_path.endswith("en") else language
        self.task = task
        self.initial_prompt = initial_prompt
        self.vad_parameters = vad_parameters or {"threshold": 0.5}
        self.no_speech_thresh = 0.45

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.model_size_or_path is None:
            return

        self.transcriber = WhisperModel(
            self.model_size_or_path,
            device=device,
            compute_type="int8" if device == "cpu" else "float16",
            local_files_only=False,
        )
        self.use_vad = use_vad


        # UTTRENCE END COUNT
        self.uttrence_end_count = 0
        self.uttrence_bool = False
        self.start_speeking = False



        # exp
        self.prev_timestamp_offset = 0
        self.prev_timestamp_offset_set:bool = False


        # HYPOTHESIS BUFFER
        self.hypothesis_buffer:HypothesisBufferPrefix = HypothesisBufferPrefix()
        # self.buffer_time_offset:int = 0
        self.commited:list = []
        # self.last_chunked_at:int = 0

        # threading
        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()
        self.websocket.send(
            json.dumps(
                {
                    "uid": self.client_uid,
                    "message": self.SERVER_READY,
                    "backend": "faster_whisper"
                }
            )
        )

    def check_valid_model(self, model_size):
        """
        Check if it's a valid whisper model size.

        Args:
            model_size (str): The name of the model size to check.

        Returns:
            str: The model size if valid, None otherwise.
        """
        if model_size not in self.model_sizes:
            self.websocket.send(
                json.dumps(
                    {
                        "uid": self.client_uid,
                        "status": "ERROR",
                        "message": f"Invalid model size {model_size}. Available choices: {self.model_sizes}"
                    }
                )
            )
            return None
        return model_size

    def set_language(self, info):
        """
        Updates the language attribute based on the detected language information.

        Args:
            info (object): An object containing the detected language and its probability. This object
                        must have at least two attributes: `language`, a string indicating the detected
                        language, and `language_probability`, a float representing the confidence level
                        of the language detection.
        """
        if info.language_probability > 0.5:
            self.language = info.language
            logger.info(f"Detected language {self.language} with probability {info.language_probability}")
            self.websocket.send(json.dumps(
                {"uid": self.client_uid, "language": self.language, "language_prob": info.language_probability}))

    def transcribe_audio(self, input_sample):
        """
        Transcribes the provided audio sample using the configured transcriber instance.

        If the language has not been set, it updates the session's language based on the transcription
        information.

        Args:
            input_sample (np.array): The audio chunk to be transcribed. This should be a NumPy
                                    array representing the audio data.

        Returns:
            The transcription result from the transcriber. The exact format of this result
            depends on the implementation of the `transcriber.transcribe` method but typically
            includes the transcribed text.
        """
        result, info = self.transcriber.transcribe(
            input_sample,
            initial_prompt=self.initial_prompt,
            language=self.language,
            task=self.task,
            vad_filter=self.use_vad,
            vad_parameters=self.vad_parameters if self.use_vad else None,
            hotwords=self.hotwords)
        logger.info(result)

        if self.language is None and info is not None:
            self.set_language(info)
        return result

    def get_previous_output(self):
        """
        Retrieves previously generated transcription outputs if no new transcription is available
        from the current audio chunks.

        Checks the time since the last transcription output and, if it is within a specified
        threshold, returns the most recent segments of transcribed text. It also manages
        adding a pause (blank segment) to indicate a significant gap in speech based on a defined
        threshold.

        Returns:
            segments (list): A list of transcription segments. This may include the most recent
                            transcribed text segments or a blank segment to indicate a pause
                            in speech.
        """
        segments = []
        if self.t_start is None:
            self.t_start = time.time()
        if time.time() - self.t_start < self.show_prev_out_thresh:
            segments = self.prepare_segments()

        # add a blank if there is no speech for 3 seconds
        if len(self.text) and self.text[-1] != '':
            if time.time() - self.t_start > self.add_pause_thresh:
                self.text.append('')
            
        return segments

    def handle_transcription_output(self, result, duration):
        """
        Handle the transcription output, updating the transcript and sending data to the client.

        Args:
            result (str): The result from whisper inference i.e. the list of segments.
            duration (float): Duration of the transcribed audio chunk.
        """
        segments = []
        if len(result):
            self.t_start = None
            last_segment = self.update_segments(result, duration)
            segments = self.prepare_segments(last_segment)
        else:
            # show previous output if there is pause i.e. no output from whisper
            segments = self.get_previous_output()

        if len(segments):
            self.send_transcription_to_client(segments)

    def speech_to_text(self):
        """
        Process an audio stream in an infinite loop, continuously transcribing the speech.

        This method continuously receives audio frames, performs real-time transcription, and sends
        transcribed segments to the client via a WebSocket connection.

        If the client's language is not detected, it waits for 30 seconds of audio input to make a language prediction.
        It utilizes the Whisper ASR model to transcribe the audio, continuously processing and streaming results. Segments
        are sent to the client in real-time, and a history of segments is maintained to provide context.Pauses in speech
        (no output from Whisper) are handled by showing the previous output for a set duration. A blank segment is added if
        there is no speech for a specified duration to indicate a pause.

        Raises:
            Exception: If there is an issue with audio processing or WebSocket communication.

        """
        while True:
            if self.exit:
                logger.info("Exiting speech to text thread")
                self.websocket.close()
                break

            if self.frames_np is None:
                # logger.info("frame np is not set yet")
                continue

            self.clip_audio_if_no_valid_segment()

            input_bytes, duration = self.get_audio_chunk_for_processing()
            if duration < 1.0:
                continue
            try:
                input_sample = input_bytes.copy()
                result = self.transcribe_audio(input_sample)

                if result is None or self.language is None:
                    if self.prev_timestamp_offset_set == False:
                        self.prev_timestamp_offset = self.timestamp_offset 
                        self.prev_timestamp_offset_set = True
                    self.timestamp_offset += duration
                    time.sleep(0.25)    # wait for voice activity, result is None when no voice activity
                    logger.info(f"{self.timestamp_offset,self.prev_timestamp_offset}")
                    logger.info(f"{self.timestamp_offset - self.prev_timestamp_offset}")

                    if self.start_speeking:
                        if self.timestamp_offset - self.prev_timestamp_offset > 2:
                            if self.uttrence_bool == False:
                                self.uttrence_end()
                                self.uttrence_bool = True
                                # if self.timestamp_offset - self.prev_timestamp_offset > 5:
                                #     self.disconnect()
                        if self.timestamp_offset - self.prev_timestamp_offset > 10:
                            if self.uttrence_bool == False:
                                self.uttrence_end()
                                self.uttrence_bool = True
                    continue
                else:
                    self.start_speeking = True
                    self.prev_timestamp_offset_set = False
                    self.uttrence_bool = False
                self.handle_transcription_output(result, duration)

            except Exception as e:
                logger.error(f"[ERROR]: Failed to transcribe audio chunk: {e}")
                time.sleep(0.01)

    def format_segment(self, start:float, end:float, text:str):
        """
        Formats a transcription segment with precise start and end times alongside the transcribed text.

        Args:
            start (float): The start time of the transcription segment in seconds.
            end (float): The end time of the transcription segment in seconds.
            text (str): The transcribed text corresponding to the segment.

        Returns:
            dict: A dictionary representing the formatted transcription segment, including
                'start' and 'end' times as strings with three decimal places and the 'text'
                of the transcription.
        """
        return {
            'start': "{:.3f}".format(start),
            'end': "{:.3f}".format(end),
            'text': text
        }

    def update_segments(self, segments, duration):
        """
        Processes the segments from whisper. Appends all the segments to the list
        except for the last segment assuming that it is incomplete.

        Updates the ongoing transcript with transcribed segments, including their start and end times.
        Complete segments are appended to the transcript in chronological order. Incomplete segments
        (assumed to be the last one) are processed to identify repeated content. If the same incomplete
        segment is seen multiple times, it updates the offset and appends the segment to the transcript.
        A threshold is used to detect repeated content and ensure it is only included once in the transcript.
        The timestamp offset is updated based on the duration of processed segments. The method returns the
        last processed segment, allowing it to be sent to the client for real-time updates.

        Args:
            segments(dict) : dictionary of segments as returned by whisper
            duration(float): duration of the current chunk

        Returns:
            dict or None: The last processed segment with its start time, end time, and transcribed text.
                     Returns None if there are no valid segments to process.
        """
        offset = None
        self.current_out = ''
        # process complete segments
        if len(segments) > 1:
            for i, s in enumerate(segments[:-1]):
                text_ = s.text
                self.text.append(text_)
                start, end = self.timestamp_offset + s.start, self.timestamp_offset + min(duration, s.end)

                if start >= end:
                    continue
                if s.no_speech_prob > self.no_speech_thresh:
                    continue

                self.transcript.append(self.format_segment(start, end, text_))
                offset = min(duration, s.end)

        self.current_out += segments[-1].text
        last_segment = self.format_segment(
            self.timestamp_offset + segments[-1].start,
            self.timestamp_offset + min(duration, segments[-1].end),
            self.current_out
        )

        # if same incomplete segment is seen multiple times then update the offset
        # and append the segment to the list
        if self.current_out.strip() == self.prev_out.strip() and self.current_out != '':
            self.same_output_threshold += 1
        else:
            self.same_output_threshold = 0

        if self.same_output_threshold > 5:
            if not len(self.text) or self.text[-1].strip().lower() != self.current_out.strip().lower():
                self.text.append(self.current_out)
                self.transcript.append(self.format_segment(
                    self.timestamp_offset,
                    self.timestamp_offset + duration,
                    self.current_out
                ))
            self.current_out = ''
            offset = duration
            self.same_output_threshold = 0
            last_segment = None
        else:
            self.prev_out = self.current_out

        # update offset
        if offset is not None:
            self.timestamp_offset += offset

        return last_segment



    