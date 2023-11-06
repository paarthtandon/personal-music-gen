import os

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import yt_dlp

def get_saved_tracks_spotify(
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        limit: int = None
) -> list:
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope='user-library-read'
        ))
    
    offset = 0
    songs = []
    while True:
        if limit is not None and len(songs) >= limit:
            break
        results = sp.current_user_saved_tracks(limit=50, offset=offset)
        if not results['items']:
            break

        for item in results['items']:
            song_name = item['track']['name']
            artist_name = item['track']['artists'][0]['name']
            songs.append(f'{artist_name} - {song_name}')

        offset += 50

    return songs

def download_songs_from_youtube(
        songs: list,
        download_dir: str,
        ffmpeg_loc: str
) -> None:
    if not os.path.isdir(download_dir):
        os.makedirs(download_dir)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '32', 
        }],
        'outtmpl': os.path.join(download_dir, '%(title)s.%(ext)s'),
        'default_search': 'ytsearch1:',
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
        'ffmpeg_location': ffmpeg_loc
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for track_string in songs:
            print(f"Downloading: {track_string}")
            ydl.download([track_string])
            print(f"Finished downloading: {track_string}")
