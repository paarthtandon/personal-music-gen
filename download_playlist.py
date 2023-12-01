import os

from dotenv import load_dotenv
from personal_musicgen.data import connectors

DATA_DIR = './data/always/mp3/'
PLAYLIST_ID = '496ea8f7yhFss7p12HDKAL'

load_dotenv()

client_id = os.getenv('spotify_client_id')
client_secret = os.getenv('spotify_client_secret')

print(f'Collecting liked songs from Spotify...')

track_strings = connectors.get_playlist_tracks_spotify(
    client_id,
    client_secret,
    'https://paarthtandon.com/',
    PLAYLIST_ID
)

print(f'Downloading {len(track_strings)} tracks from YouTube...')

connectors.download_songs_from_youtube(
    track_strings,
    DATA_DIR,
    # ffmpeg_loc='./ffmpeg-6/bin/ffmpeg.exe'
)
