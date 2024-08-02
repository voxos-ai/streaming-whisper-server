from .base import Denoise
from .deepfilternet_denoise import DeepFilterNetDenoise
from .facebook_denoise import FaceBookDenoise
# from typing import Dict


Denoisers ={
    "FaceBookDenoise": FaceBookDenoise,
    "DeepFilterNetDenoise": DeepFilterNetDenoise
}
