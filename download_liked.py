import os

from dotenv import load_dotenv
from personal_musicgen.data import connectors

DATA_DIR = './data/mp3/'

load_dotenv()

client_id = os.getenv('spotify_client_id')
client_secret = os.getenv('spotify_client_secret')

print(f'Collecting liked songs from Spotify...')

track_strings = connectors.get_saved_tracks_spotify(
    client_id,
    client_secret,
    'https://paarthtandon.com/'
)

print(f'Downloading {len(track_strings)} tracks from YouTube...')

connectors.download_songs_from_youtube(
    track_strings,
    DATA_DIR,
    # ffmpeg_loc='./ffmpeg-6/bin/ffmpeg.exe'
)
