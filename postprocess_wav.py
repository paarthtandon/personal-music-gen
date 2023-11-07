from pathlib import Path

from personal_musicgen.data.postprocess import Postprocessor
from tqdm import tqdm

source_dir = 'data/wav_32k/'
original_dir = 'data/chunks_original'
no_voice_dir = 'data/chunks_no_voice'

Path(original_dir).mkdir(exist_ok=True)
Path(no_voice_dir).mkdir(exist_ok=True)

pp = Postprocessor()
wav_files = list(Path(source_dir).glob('*.wav'))

for wav_file in tqdm(wav_files):
    wav_fp = source_dir + wav_file.name
    try:
        pp.postprocess(wav_fp, original_dir, no_voice_dir, max_chunks=2)
    except:
        print(f'Failed to process {wav_fp}')
        continue
