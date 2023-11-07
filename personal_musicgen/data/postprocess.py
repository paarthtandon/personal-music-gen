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
            self,
            remove_voice: bool = True,
            detect_genre: bool = True
    ):
        self.remove_voice = remove_voice
        self.detect_genre = detect_genre
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_sample_rate = 32_000

        if self.detect_genre:
            genre_model_name = 'm3hrdadfi/wav2vec2-base-100k-gtzan-music-genres'
            self.genre_config = AutoConfig.from_pretrained(genre_model_name)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(genre_model_name)
            self.sampling_rate = self.feature_extractor.sampling_rate
            self.genre_model = Wav2Vec2ForAudioClassification \
                            .from_pretrained(genre_model_name).to(self.device)
        
        if self.remove_voice:
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
            {"Label": self.genre_config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} \
                for i, score in enumerate(scores)
        ]
        return preds
    
    def separate_voice(self, path):
        original, separated = self.separator.separate_audio_file(path)
        no_voice = separated['bass'] + separated['drums'] + separated['other']
        return original, no_voice, self.separator.samplerate
