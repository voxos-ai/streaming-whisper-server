from whisper_live.client import TranscriptionClient


client = TranscriptionClient(
  # "44.221.66.152",
  "127.0.0.1",
  9090,
  translate=False,
  model="tiny.en",
  use_vad=True,
)

# client("./audio/audio.wav")
# client.client.get_avg_timetake()
# client.client.time_at_receving[0] - client.client.time_at_sending[0]
# client.close_all_clients()
