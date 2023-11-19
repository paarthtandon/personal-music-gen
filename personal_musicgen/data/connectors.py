import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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
            scope='user-library-read',
            open_browser=False
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

def get_playlist_tracks_spotify(
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        playlist_id: str,
        limit: int = None
) -> list:
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope='playlist-read-private',
            open_browser=False
        ))
    
    offset = 0
    songs = []
    while True:
        if limit is not None and len(songs) >= limit:
            break
        results = sp.playlist_tracks(playlist_id, limit=50, offset=offset)
        if not results['items']:
            break

        for item in results['items']:
            track = item['track']
            if track:  # Check if track exists (playlist might have null tracks)
                song_name = track['name']
                artist_name = track['artists'][0]['name']
                songs.append(f'{artist_name} - {song_name}')

        offset += 50

    return songs

def get_album_tracks_spotify(
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        album_id: str,
        limit: int = None
) -> list:
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope='user-library-read',
            open_browser=False
        ))
    
    offset = 0
    songs = []
    while True:
        if limit is not None and len(songs) >= limit:
            break
        results = sp.album_tracks(album_id, limit=50, offset=offset)
        if not results['items']:
            break

        for item in results['items']:
            song_name = item['name']
            artist_name = item['artists'][0]['name']
            songs.append(f'{artist_name} - {song_name}')

        offset += 50

    return songs

def download_songs_from_youtube(
        track_strings: list,
        download_dir: str,
        # ffmpeg_loc: str
) -> None:
    if not os.path.isdir(download_dir):
        os.makedirs(download_dir)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128'
        }],
        'outtmpl': os.path.join(download_dir, '%(title)s.%(ext)s'),
        'default_search': 'ytsearch1:',
        'quiet': True,
        'no_warnings': False,
        'ignoreerrors': False,
        # 'ffmpeg_location': ffmpeg_loc
    }

    def download_song(track_string, ydl_opts):
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([track_string])

    download_count = 0

    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_track = {executor.submit(
            download_song, track, ydl_opts
        ): track for track in track_strings}

        for future in as_completed(future_to_track):
            track = future_to_track[future]
            try:
                future.result()  # Wait for each download to complete
                download_count += 1  # Increment the counter
                if download_count % 100 == 0:  # Check if the counter is a multiple of 100
                    print(f'Downloaded {download_count}/{len(track_strings)} tracks.')
            except Exception as e:
                print(f"Failed to download {track}: {e}")
