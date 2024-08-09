from .base import Denoise
from .deepfilternet_denoise import DeepFilterNetDenoise
from .facebook_denoise import FaceBookDenoise
from .speechbrain_denoise import SpeechBrainDenoise
# from typing import Dict


Denoisers ={
    "FaceBookDenoise": FaceBookDenoise,
    "DeepFilterNetDenoise": DeepFilterNetDenoise,
    "SpeechBrainDenoise": SpeechBrainDenoise
}
