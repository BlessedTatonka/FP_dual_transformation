import nemo
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
from omegaconf import DictConfig
import pathlib
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
import re

import copy
import json
import os
import tempfile
from math import ceil
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from torch.utils.data import ChainDataset
from tqdm import tqdm

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.parts.mixins import ASRModuleMixin
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, LogprobsType, NeuralType, SpectrogramType
from nemo.utils import logging
from nemo.collections.tts.models import FastPitchHifiGanE2EModel

import IPython.display as ipd
import librosa
import librosa.display
import numpy as np
import torch
from matplotlib import pyplot as plt

# Reduce logging messages for this notebook
from nemo.utils import logging
logging.setLevel(logging.ERROR)

from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models import HifiGanModel
from nemo.collections.tts.helpers.helpers import regulate_len
from nemo.collections.tts.models import MelGanModel

from torch.nn import MSELoss
import torch.nn.functional as F
from matplotlib.pyplot import imshow

from nemo.collections.tts.models import FastPitchModel
import pytorch_lightning as pl

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import FastPitchModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
import noisereduce as nr


try:
    from ruamel.yaml import YAML
except ModuleNotFoundError:
    from ruamel_yaml import YAML

from dataset import UnpairedAudioToBPEDataset