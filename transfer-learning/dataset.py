import io
import math
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import braceexpand
import librosa
import numpy as np
import torch
import webdataset as wd
from scipy.stats import betabinom
from torch.nn import functional as F

from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.common.parts.preprocessing import collections, parsers
from nemo.core.classes import Dataset, IterableDataset
from nemo.core.neural_types import *
from nemo.core.neural_types.elements import ProbsType
from nemo.utils import logging

from exp_collections import ASRSignal, ASRText


class ExtendedASRManifestProcessor:
    """
    Class that processes a manifest json file containing paths to audio files, transcripts, and durations (in seconds).
    Each new line is a different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath": "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    Args:
        manifest_filepath: Path to manifest json as described above. Can be comma-separated paths.
        parser: Str for a language specific preprocessor or a callable.
        max_duration: If audio exceeds this length, do not include in dataset.
        min_duration: If audio is less than this length, do not include in dataset.
        max_utts: Limit number of utterances.
        bos_id: Id of beginning of sequence symbol to append if not None.
        eos_id: Id of end of sequence symbol to append if not None.
        pad_id: Id of pad symbol. Defaults to 0.
    """

    def __init__(
        self,
        manifest_filepath: str,
        unpaired_text_manifest_filepath: str,
        unpaired_signal_manifest_filepath: str,
        parser: Union[str, Callable],
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
    ):
        self.parser = parser

        self.collection_paired = collections.ASRAudioText(
            manifests_files=manifest_filepath,
            parser=parser,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
        )
        
        self.collection_signal = ASRSignal(
            manifests_files=unpaired_signal_manifest_filepath,
            parser=parser,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
        )
        self.collection_text = ASRText(
            manifests_files=unpaired_text_manifest_filepath,
            parser=parser,
        )

        while len(self.collection_paired) < min(len(self.collection_signal), len(self.collection_text)):
            for i in range(len(self.collection_paired)):
                self.collection_paired.append(self.collection_paired[i])
                
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id

    def process_text(self, index, i) -> (List[int], int):
        if i == 0:
            sample = self.collection_paired[index]
        else:
            sample = self.collection_text[index]
            
        t, tl = sample.text_tokens, len(sample.text_tokens)

        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        return t, tl

class _UnpairedAudioTextDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio files, transcripts, and durations (in seconds).
    Each new line is a different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath": "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    Args:
        manifest_filepath: Path to manifest json as described above. Can be comma-separated paths.
        labels: String containing all the possible characters to map to
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor object used to augment loaded
            audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include in dataset
        max_utts: Limit number of utterances
        blank_index: blank character index, default = -1
        unk_index: unk_character index, default = -1
        normalize: whether to normalize transcript text (default): True
        bos_id: Id of beginning of sequence symbol to append if not None
        eos_id: Id of end of sequence symbol to append if not None
        return_sample_id (bool): whether to return the sample_id as a part of each sample
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'paired_audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'paired_a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'paired_transcripts': NeuralType(('B', 'T'), LabelsType()),
            'paired_transcript_length': NeuralType(tuple('B'), LengthsType()),
            'unpaired_audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'unpaired_a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'unpaired_transcripts': NeuralType(('B', 'T'), LabelsType()),
            'unpaired_transcript_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(
        self,
        manifest_filepath: str,
        unpaired_text_manifest_filepath: str,
        unpaired_signal_manifest_filepath: str,
        parser: Union[str, Callable],
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        return_sample_id: bool = False,
    ):
        if type(manifest_filepath) == str:
            manifest_filepath = manifest_filepath.split(",")

        self.manifest_processor = ExtendedASRManifestProcessor(
            manifest_filepath=manifest_filepath,
            unpaired_text_manifest_filepath=unpaired_text_manifest_filepath,
            unpaired_signal_manifest_filepath=unpaired_signal_manifest_filepath,
            parser=parser,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
        )
        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim
        self.return_sample_id = return_sample_id

    def get_manifest_sample(self, sample_id):
        return [self.manifest_processor.collection_paired[sample_id], 
                self.manifest_processor.collection_signal[sample_id],
                self.manifest_processor.collection_text[sample_id]]

    def __getitem__(self, index):
        sample_paired = self.manifest_processor.collection_paired[index]
        sample_signal = self.manifest_processor.collection_signal[index]
        sample_text = self.manifest_processor.collection_text[index]
        offset = sample_paired.offset

        if offset is None:
            offset = 0

        features = self.featurizer.process(
            sample_paired.audio_file, offset=offset, duration=sample_paired.duration, trim=self.trim,
            orig_sr=sample_paired.orig_sr
        )
        f, fl = features, torch.tensor(features.shape[0]).long()

        t, tl = self.manifest_processor.process_text(index, 0)
        
        u_features = self.featurizer.process(
            sample_signal.audio_file, offset=offset, duration=sample_signal.duration, trim=self.trim,
            orig_sr=sample_signal.orig_sr
        )
        
        u_f, u_fl = u_features, torch.tensor(u_features.shape[0]).long()
        
        
        u_t, u_tl = self.manifest_processor.process_text(index, 1)
                       
        if self.return_sample_id:
            output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), \
            u_f, u_fl,  torch.tensor(u_t).long(), torch.tensor(u_tl).long(), index
        else:
            output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), \
            u_f, u_fl,  torch.tensor(u_t).long(), torch.tensor(u_tl).long(),

        return output

    def __len__(self):
        return len(self.manifest_processor.collection_paired)

    def len_signal(self):
        return len(self.manifest_processor.collection_signal)
    
    def len_text(self):
        return len(self.manifest_processor.collection_text)
    
    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, pad_id=self.manifest_processor.pad_id)

class UnpairedAudioToBPEDataset(_UnpairedAudioTextDataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, transcripts, and durations (in seconds). Each new line is a
    different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath":
    "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the
    transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    In practice, the dataset and manifest used for character encoding and byte pair encoding
    are exactly the same. The only difference lies in how the dataset tokenizes the text in
    the manifest.
    Args:
        manifest_filepath: Path to manifest json as described above. Can
            be comma-separated paths.
        tokenizer: A subclass of the Tokenizer wrapper found in the common collection,
            nemo.collections.common.tokenizers.TokenizerSpec. ASR Models support a subset of
            all available tokenizers.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include
            in dataset
        max_utts: Limit number of utterances
        trim: Whether to trim silence segments
        use_start_end_token: Boolean which dictates whether to add [BOS] and [EOS]
            tokens to beginning and ending of speech respectively.
        return_sample_id (bool): whether to return the sample_id as a part of each sample
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'paired_audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'paired_a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'paired_transcripts': NeuralType(('B', 'T'), LabelsType()),
            'paired_transcript_length': NeuralType(tuple('B'), LengthsType()),
            'unpaired_audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'unpaired_a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'unpaired_transcripts': NeuralType(('B', 'T'), LabelsType()),
            'unpaired_transcript_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(
        self,
        manifest_filepath: str,
        unpaired_text_manifest_filepath: str,
        unpaired_signal_manifest_filepath: str,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        trim: bool = False,
        use_start_end_token: bool = True,
        return_sample_id: bool = False,
    ):
        if use_start_end_token and hasattr(tokenizer, 'bos_token'):
            bos_id = tokenizer.bos_id
        else:
            bos_id = None

        if use_start_end_token and hasattr(tokenizer, 'eos_token'):
            eos_id = tokenizer.eos_id
        else:
            eos_id = None

        if hasattr(tokenizer, 'pad_token'):
            pad_id = tokenizer.pad_id
        else:
            pad_id = 0

        class TokenizerWrapper:
            def __init__(self, tokenizer):
                self._tokenizer = tokenizer

            def __call__(self, text):
                t = self._tokenizer.text_to_ids(text)
                return t

        super().__init__(
            manifest_filepath=manifest_filepath,
            unpaired_text_manifest_filepath=unpaired_text_manifest_filepath,
            unpaired_signal_manifest_filepath=unpaired_signal_manifest_filepath,
            parser=TokenizerWrapper(tokenizer),
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            trim=trim,
            return_sample_id=return_sample_id,
        )