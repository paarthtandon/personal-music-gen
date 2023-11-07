import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)
from transformers.file_utils import ModelOutput

from dataclasses import dataclass
from typing import Optional, Tuple
import random
import os
import re

import demucs.api

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForAudioClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode \
                      must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class Postprocessor:

    def __init__(
            self
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_sample_rate = 32_000

        genre_model_name = 'm3hrdadfi/wav2vec2-base-100k-gtzan-music-genres'
        self.genre_config = AutoConfig.from_pretrained(genre_model_name)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(genre_model_name)
        self.sampling_rate = self.feature_extractor.sampling_rate
        self.genre_model = Wav2Vec2ForAudioClassification \
                        .from_pretrained(genre_model_name).to(self.device)
        
        self.separator = demucs.api.Separator()
    
    def predict_genre(self, path):
        waveform, sample_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sample_rate)
        waveform = resampler(waveform)

        clip_length = 5  # seconds
        num_samples_in_clip = sample_rate * clip_length

        start_sample = random.randint(0, waveform.size(1) - num_samples_in_clip)
        random_clip = waveform[:, start_sample:start_sample + num_samples_in_clip]
        random_clip = random_clip.squeeze().numpy()

        inputs = self.feature_extractor(
            random_clip, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True
        )
        inputs = {key: inputs[key].to(self.device) for key in inputs}

        with torch.no_grad():
            logits = self.genre_model(**inputs).logits

        scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        preds = [
            {"Label": self.genre_config.id2label[i], "Score": score} \
                for i, score in enumerate(scores)
        ]
        return preds
    
    def save_chunks(self, track_name, chunk, folder_path, genre, signal_type):
        filename = f"{track_name}_{signal_type}.wav"
        text_filename = f"{track_name}_{signal_type}.txt"
        
        # Save the chunk as a wav file
        torchaudio.save(os.path.join(folder_path, filename), chunk, self.target_sample_rate)
        
        # Save the genre in a text file
        with open(os.path.join(folder_path, text_filename), 'w') as txt_file:
            txt_file.write(f"personal, {genre}")

    def remove_voice_and_save(self, track_name, chunk, folder_path, genre):
        # Apply the voice removal to the provided chunk
        _, separated = self.separator.separate_tensor(chunk)
        no_voice = separated['bass'] + separated['drums'] + separated['other']
        
        # Ensure no_voice is in the correct shape (channels, samples)
        no_voice = no_voice if no_voice.dim() == 2 else no_voice.unsqueeze(0)
        
        # Resample the no_voice part to the target sample rate
        # resampler = torchaudio.transforms.Resample(
        #     orig_freq=self.separator.samplerate, new_freq=self.target_sample_rate
        # )
        no_voice_resampled = no_voice #resampler(no_voice)
        
        # Save the no_voice chunk and the associated text file
        self.save_chunks(track_name, no_voice_resampled, folder_path, genre, 'no_voice')

    def postprocess(self, path, original_folder, no_voice_folder, max_chunks=None):
        genre_preds = self.predict_genre(path)
        top_genre = sorted(genre_preds, key=lambda x: x['Score'], reverse=True)[0]['Label']

        waveform, sample_rate = torchaudio.load(path)
        
        # Resample the full track before chunking
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=self.target_sample_rate
        )
        waveform = resampler(waveform)
        
        chunk_size = self.target_sample_rate * 30
        num_possible_chunks = waveform.shape[1] // chunk_size

        # Create a list of chunk indices to select them randomly if max_chunks is specified
        chunk_indices = list(range(num_possible_chunks))
        if max_chunks is not None and max_chunks < num_possible_chunks:
            chunk_indices = random.sample(chunk_indices, max_chunks)

        for i in chunk_indices:
            start_sample = i * chunk_size
            end_sample = start_sample + chunk_size
            chunk = waveform[:, start_sample:end_sample]

            if chunk.shape[1] == chunk_size:
                # Process the name
                _, filename = os.path.split(path)
                file_base, _ = os.path.splitext(filename)
                pattern = re.compile(r'[\W\d]+', re.UNICODE)
                track_base = re.sub(pattern, '', file_base)
                track_base = re.sub(r'[\d_]', '', track_base).lower()
                track_chunk_name = f"{track_base}_chunk_{i}"

                self.save_chunks(track_chunk_name, chunk, original_folder, top_genre, 'original')
                self.remove_voice_and_save(track_chunk_name, chunk, no_voice_folder, top_genre)
