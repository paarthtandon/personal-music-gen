import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
import os

from audiocraft.models import MusicGen
from personal_musicgen.data.datasets import AudioDataset
from personal_musicgen.model_utils import train_step, eval_step

import wandb
wandb.login()

PROJECT_NAME = 'always'
RUN_NAME = 'voice-encodec'
DATA_DIR = './data/always/chunks_original'
CHECKPOINT_DIR = './checkpoints'
MODEL = 'small'
START_WEIGHTS = None
TOTAL_DATA_RATIO = 1.0
EVAL_DATA_RATIO = 0
EPOCHS = 1000
BATCH_SIZE = 16
GRAD_ACC_STEPS = 1
LR = 1e-4

run = wandb.init(
    project = PROJECT_NAME,
    name = RUN_NAME,
    config = {
        'dataset': DATA_DIR,
        'model': MODEL,
        'start_weights': START_WEIGHTS,
        'total_data_ratio': TOTAL_DATA_RATIO,
        'eval_data_ratio': EVAL_DATA_RATIO,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'grad_acc_steps': GRAD_ACC_STEPS,
        'lr': LR
    }
)

if not os.path.exists(CHECKPOINT_DIR + f'/{run.id}'):
    os.makedirs(CHECKPOINT_DIR + f'/{run.id}')

torch.manual_seed(222)

########## Data Setup ##########

dataset = AudioDataset(DATA_DIR)
indices = torch.randperm(len(dataset)).tolist()
shuffled_dataset = Subset(dataset, indices[:int(len(indices) * TOTAL_DATA_RATIO)])

train_len = int(len(shuffled_dataset) * (1 - EVAL_DATA_RATIO))
train_dataset = Subset(shuffled_dataset, range(train_len))
if EVAL_DATA_RATIO > 0:
    eval_dataset = Subset(shuffled_dataset, range(train_len, len(shuffled_dataset)))

if EVAL_DATA_RATIO > 0:
    print(f'{len(train_dataset)=}, {len(eval_dataset)=}')
else:
    print(f'{len(train_dataset)=}')

train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=False)
if EVAL_DATA_RATIO > 0:
    eval_dataloader = DataLoader(eval_dataset, BATCH_SIZE, shuffle=False)

########## Model Setup ##########

torch.cuda.empty_cache()

model = MusicGen.get_pretrained(MODEL)
model.lm = model.lm.to(torch.float32)
device = model.device

if START_WEIGHTS != None:
    model.lm.load_state_dict(torch.load(START_WEIGHTS)['model_state_dict'])

print(f'{device=}')

optimizer = AdamW(
    list(model.lm.condition_provider.parameters()) + \
    list(model.compression_model.encoder.parameters()) + \
    list(model.compression_model.decoder.parameters()),
    lr=LR,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

if START_WEIGHTS != None:
    optimizer.load_state_dict(torch.load(START_WEIGHTS)['optimizer_state_dict'])

scaler = GradScaler()
scheduler = CosineAnnealingLR(optimizer, T_max=10)

########## Training ##########

if START_WEIGHTS != None:
    start_epoch = torch.load(START_WEIGHTS)['epoch']
else:
    start_epoch = 0

for epoch in range(start_epoch, EPOCHS):
    print(f'epoch {epoch}/{EPOCHS}')

    train_loss = train_step(
        model,
        optimizer,
        scaler,
        train_dataloader,
        GRAD_ACC_STEPS,
        scheduler = scheduler
    )['train_loss']

    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss
    })

    if EVAL_DATA_RATIO > 0:
        eval_loss = eval_step(
            model,
            eval_dataloader
        )['eval_loss']

        wandb.log({
            'epoch': epoch,
            'eval_loss': eval_loss
        })
    
    if (epoch + 1) % 25 == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR + f'/{run.id}', f'checkpoint_{epoch}.pth')
        print(f'Saving to {checkpoint_path}...')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.lm.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
