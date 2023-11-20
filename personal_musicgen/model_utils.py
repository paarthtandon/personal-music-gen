import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import torchaudio

from audiocraft.models import MusicGen
from audiocraft.models.lm import LMOutput
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout

import wandb
from tqdm import tqdm


def get_contitional_vector(model: MusicGen, attributes: torch.Tensor, use_cfg:bool = False) -> dict:
    if use_cfg:
        null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(attributes)
        attributes = attributes + null_conditions
    tokenized = model.lm.condition_provider.tokenize(attributes)
    cfg_conditions = model.lm.condition_provider(tokenized)
    
    return cfg_conditions

def encode_audio(model: MusicGen, wav_file: str) -> torch.Tensor:
    wav, _ = torchaudio.load(wav_file)
    wav = wav.to(model.device).mean(dim=0, keepdim=True)
    wav = wav.unsqueeze(1)

    assert wav.shape[0] == 1

    with torch.no_grad():
        codes, _ = model.compression_model.encode(wav)
    return codes

def compute_masked_loss(lm_output: LMOutput, codes: torch.Tensor) -> torch.Tensor:
    codes = codes[0]
    logits = lm_output.logits[0]
    mask = lm_output.mask[0]

    codes = torch.nn.functional.one_hot(codes, 2048).type(logits.dtype)
    mask = mask.view(-1)
    masked_logits = logits.view(-1, 2048)[mask]
    masked_codes = codes.view(-1, 2048)[mask]
    
    criterion = nn.CrossEntropyLoss()
    loss = criterion(masked_logits, masked_codes)

    return loss

def train_step(
        model: MusicGen,
        optimizer: AdamW,
        scaler: GradScaler,
        dataloader: DataLoader,
        grad_acc_steps: int,
        use_cfg: bool = False
) -> dict:
    model.lm.train()
    device = model.device
    
    total_loss = 0

    for i, (audio_fns, label_fns) in tqdm(
        enumerate(dataloader), total=len(dataloader), desc='Train'
    ):
        codes_l = []
        text_l = []

        for audio_fn, label_fn in zip(audio_fns, label_fns):
            codes = encode_audio(model, audio_fn)
            codes_l.append(codes)
            with open(label_fn, 'r') as label_f:
                text_l.append(label_f.read().strip())
        
        codes = torch.cat(codes_l, dim=0).to(device)
        if use_cfg:
            codes = torch.cat([codes, codes], dim=0)

        attributes, _ = model._prepare_tokens_and_attributes(text_l, None)
        conditional_vector = get_contitional_vector(model, attributes, use_cfg=False)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            lm_output = model.lm.compute_predictions(
                codes=codes,
                conditions=[],
                condition_tensors=conditional_vector
            )

            loss = compute_masked_loss(lm_output, codes)
            scaler.scale(loss).backward()
            
            total_loss += loss.item()
            wandb.log({
                'ins_train_loss': loss.item()
            })

            if (i + 1) % grad_acc_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.lm.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

    return {
        'train_loss': total_loss / len(dataloader)
    }

def eval_step(
        model: MusicGen,
        dataloader: DataLoader
) -> dict:
    model.lm.eval()
    device = model.device
    
    total_loss = 0

    with torch.no_grad():
        for i, (audio_fns, label_fns) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc='Eval'
        ):
            codes_l = []
            text_l = []

            for audio_fn, label_fn in zip(audio_fns, label_fns):
                codes = encode_audio(model, audio_fn)
                codes_l.append(codes)
                with open(label_fn, 'r') as label_f:
                    text_l.append(label_f.read().strip())
            
            codes = torch.cat(codes_l, dim=0).to(device)

            attributes, _ = model._prepare_tokens_and_attributes(text_l, None)
            conditional_vector = get_contitional_vector(model, attributes)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                lm_output = model.lm.compute_predictions(
                    codes=codes,
                    conditions=[],
                    condition_tensors=conditional_vector
                )

                loss = compute_masked_loss(lm_output, codes)
                total_loss += loss.item()

    return {
        'eval_loss': total_loss / len(dataloader)
    }

def generate(
        model: MusicGen,
        text_prompt: str,
        temp: float = 1.0,
        top_k: int= 250,
        top_p: float = 0.0
):
    model.generation_params = {
        'max_gen_len': int(30 * model.frame_rate),
        'use_sampling': True,
        'temp': temp,
        'top_k': top_k,
        'top_p': top_p,
        'cfg_coef': 3,
        'two_step_cfg': 0,
    }

    attributes, prompt_tokens = model._prepare_tokens_and_attributes([text_prompt], None)
    with model.autocast:
        gen_tokens = model.lm.generate(prompt_tokens, attributes, callback=None, **model.generation_params)

    with torch.no_grad():
        gen_audio = model.compression_model.decode(gen_tokens, None)
    gen_audio = gen_audio.cpu()

    return gen_audio
