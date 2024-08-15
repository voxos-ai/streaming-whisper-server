from .base import Denoise
from .deepfilternet_denoise import DeepFilterNetDenoise
from .facebook_denoise import FaceBookDenoise
from .facebook_denoise_master64 import FaceBookDenoiseM64
# from typing import Dict


Denoisers ={
    "FaceBookDenoise": FaceBookDenoise,
    "DeepFilterNetDenoise": DeepFilterNetDenoise,
    "FaceBookDenoiseM64": FaceBookDenoiseM64
}
