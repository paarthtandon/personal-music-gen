# Personal MusicGen Trainer

[MusicGen](https://ai.honu.io/papers/musicgen/) is a music generation model developed by Meta.
This project sets up a pipeline for fine-tuning MusicGen on a personal dataset.

## Setup

Before starting the setup, I would recommend using a virtual environment for this project.

First, clone the repository with submodules.

```{bash}
git clone --recurse-submodules https://github.com/paarthtandon/personal-music-gen
```

After that, install PyTorch.

Next, install both `libs/audiocraft` and `libs/demucs`.
To install them, enter the respective directory and run:

```{bash}
python -m pip install -e .
```

Finally, install the required packages.

```{bash}
python -m pip install -r requirements.txt
```

## Data Utils

To create the dataset, I wrote some scripts that scrape song titles from Spotify and then download them using `yt-dlp`.
To use them, create a [developer account](https://developer.spotify.com/) and save your credentials in an environment file (`.env`).
Currently, I have three download scripts:

* `download_album.py`
* `download_playlist.py`
* `download_liked.py`

I also have scripts that prepare the files before training.

* `mp3_to_wav.py`: Converts downloaded mp3 files to wav files at 32k sample rate.
* `preprocess_wav.py`: Chops up the wav files into 30 second chunks. This script has options to strip the voice from the audio, and also predict the genre for the audio. It creates a text file associated with each chunk, which functions as the text to condition on when predicting the audio.

## Training

`train.py` is an example training script that uses `wandb` for experiment organization.

## Experimental Results

`experimental_results/best_sample/eyedazzler` has some of the best samples generated after training on the Alison's Halo album "Eyedazzler". These samples were generated after training the "small" model until it was fully optimized, reaching a loss of near zero.

Using an Nvidia L40 GPU, I was able to train the "small" version of the model with a batch size of 16. I trained it using a cosine annealing schedule with a max learning rate of 1e-5 and a min of 0.

## Credits

I got help from these previous attempts:

* [Beinabih/Unconditional-MusicGen-Trainer](https://github.com/Beinabih/Unconditional-MusicGen-Trainer)
* [chavinlo/musicgen_trainer](https://github.com/chavinlo/musicgen_trainer)
