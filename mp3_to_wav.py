import subprocess
from pathlib import Path

from tqdm import tqdm

ffmpeg_location = 'ffmpeg'
source_dir = Path('data/always/mp3')
target_dir = Path('data/always/wav_32k')
target_dir.mkdir(parents=True, exist_ok=True)

mp3_files = list(source_dir.glob('*.mp3'))
for mp3_file in tqdm(mp3_files):
    wav_file = target_dir / mp3_file.with_suffix('.wav').name
    command = [ffmpeg_location, '-i', str(mp3_file), '-ar', '32000', str(wav_file)]
    
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
