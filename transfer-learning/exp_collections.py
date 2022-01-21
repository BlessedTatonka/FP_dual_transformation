import collections
import json
import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from nemo.collections.common.parts.preprocessing import manifest, parsers
from nemo.utils import logging

from os.path import expanduser

  
class _Collection(collections.UserList):
    """List of parsed and preprocessed data."""

    OUTPUT_TYPE = None  # Single element output type.
    
class Text(_Collection):
    """List of text correspondence with preprocessing."""

    OUTPUT_TYPE = collections.namedtuple(typename='TextEntity', field_names='text_tokens')

    def __init__(
        self,
        ids: List[int],
        texts: List[str],
        parser: parsers.CharParser,
    ):
        """Instantiates text manifest with filters and preprocessing.
        Args:
            texts: List of raw text transcripts.
            parser: Instance of `CharParser` to convert string to tokens.
        """

        data, output_type = [], self.OUTPUT_TYPE
        for text in texts:
            tokens = parser(text)

            if tokens is None:
                logging.warning("Fail to parse '%s' text line.", text)
                continue

            data.append(output_type(tokens))

        super().__init__(data)
        
class ASRText(Text):
    """`Text` collector from asr structured json files."""

    def __init__(self, manifests_files: Union[str, List[str]], *args, **kwargs):
        """Parse lists of audio files, durations and transcripts texts.
        Args:
            manifests_files: Either single string file or list of such -
                manifests to yield items from.
            *args: Args to pass to `Text` constructor.
            **kwargs: Kwargs to pass to `Text` constructor.
        """

        ids, texts = [], []
        for item in manifest.item_iter(manifests_files, parse_func=self.parse_text_item):
            ids.append(item['id'])
            texts.append(item['text'])

        super().__init__(ids=ids, texts=texts, *args, **kwargs)

    def parse_text_item(self, line: str, manifest_file: str) -> Dict[str, Any]:
        item = json.loads(line)

        if 'text' in item:
            pass
        elif 'text_filepath' in item:
            with open(item.pop('text_filepath'), 'r') as f:
                item['text'] = f.read().replace('\n', '')
        elif 'normalized_text' in item:
            item['text'] = item['normalized_text']

        item = dict(
            text=item.get('text', ""),
        )

        return item

class Signal(_Collection):
    """List of audio-transcript text correspondence with preprocessing."""

    OUTPUT_TYPE = collections.namedtuple(
        typename='SignalEntity', field_names='id audio_file duration offset speaker orig_sr',
    )

    def __init__(
        self,
        ids: List[int],
        audio_files: List[str],
        durations: List[float],
        offsets: List[str],
        speakers: List[Optional[int]],
        orig_sampling_rates: List[Optional[int]],
        parser: parsers.CharParser,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_number: Optional[int] = None,
        do_sort_by_duration: bool = False,
        index_by_file_id: bool = False,
    ):
        """Instantiates audio manifest with filters and preprocessing.
        Args:
            ids: List of examples positions.
            audio_files: List of audio files.
            durations: List of float durations.
            offsets: List of duration offsets or None.
            speakers: List of optional speakers ids.
            orig_sampling_rates: List of original sampling rates of audio files.
            parser: Instance of `CharParser` to convert string to tokens.
            min_duration: Minimum duration to keep entry with (default: None).
            max_duration: Maximum duration to keep entry with (default: None).
            max_number: Maximum number of samples to collect.
            do_sort_by_duration: True if sort samples list by duration. Not compatible with index_by_file_id.
            index_by_file_id: If True, saves a mapping from filename base (ID) to index in data.
        """

        output_type = self.OUTPUT_TYPE
        data, duration_filtered, num_filtered, total_duration = [], 0.0, 0, 0.0
        if index_by_file_id:
            self.mapping = {}

        for id_, audio_file, duration, offset, speaker, orig_sr in zip(
            ids, audio_files, durations, offsets, speakers, orig_sampling_rates
        ):
            # Duration filters.
            if min_duration is not None and duration < min_duration:
                duration_filtered += duration
                num_filtered += 1
                continue

            if max_duration is not None and duration > max_duration:
                duration_filtered += duration
                num_filtered += 1
                continue

            total_duration += duration

            data.append(output_type(id_, audio_file, duration, offset, speaker, orig_sr))
            if index_by_file_id:
                file_id, _ = os.path.splitext(os.path.basename(audio_file))
                self.mapping[file_id] = len(data) - 1

            # Max number of entities filter.
            if len(data) == max_number:
                break

        if do_sort_by_duration:
            if index_by_file_id:
                logging.warning("Tried to sort dataset by duration, but cannot since index_by_file_id is set.")
            else:
                data.sort(key=lambda entity: entity.duration)

        logging.info("Dataset loaded with %d files totalling %.2f hours", len(data), total_duration / 3600)
        logging.info("%d files were filtered totalling %.2f hours", num_filtered, duration_filtered / 3600)

        super().__init__(data)


class ASRSignal(Signal):
    """`Signal` collector from asr structured json files."""

    def __init__(self, manifests_files: Union[str, List[str]], *args, **kwargs):
        """Parse lists of audio files, durations.
        Args:
            manifests_files: Either single string file or list of such -
                manifests to yield items from.
            *args: Args to pass to `Signal` constructor.
            **kwargs: Kwargs to pass to `Signal` constructor.
        """

        ids, audio_files, durations, offsets, speakers, orig_srs = [], [], [], [], [], []
        for item in manifest.item_iter(manifests_files, parse_func=self.parse_signal_item):
            ids.append(item['id'])
            audio_files.append(item['audio_file'])
            durations.append(item['duration'])
            offsets.append(item['offset'])
            speakers.append(item['speaker'])
            orig_srs.append(item['orig_sr'])

        super().__init__(ids, audio_files, durations, offsets, speakers, orig_srs, *args, **kwargs)
        
    def parse_signal_item(self, line: str, manifest_file: str) -> Dict[str, Any]:
        item = json.loads(line)

        # Audio file
        if 'audio_filename' in item:
            item['audio_file'] = item.pop('audio_filename')
        elif 'audio_filepath' in item:
            item['audio_file'] = item.pop('audio_filepath')
        else:
            raise ValueError(
                f"Manifest file {manifest_file} has invalid json line structure: {line} without proper audio file key."
            )
        item['audio_file'] = expanduser(item['audio_file'])

        if 'duration' not in item:
            raise ValueError(
                f"Manifest file {manifest_file} has invalid json line structure: {line} without proper duration key."
            )

        item = dict(
            audio_file=item['audio_file'],
            duration=item['duration'],
            offset=item.get('offset', None),
            speaker=item.get('speaker', None),
            orig_sr=item.get('orig_sample_rate', None),
        )

        return item