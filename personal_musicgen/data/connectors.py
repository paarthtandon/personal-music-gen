import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time

def get_saved_tracks_spotify(
        client_id: str,
        client_secret: str,
        redirect_uri: str
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
        results = sp.current_user_saved_tracks(limit=50, offset=offset)
        if not results['items']:
            break

        for item in results['items']:
            song_name = item['track']['name']
            artist_name = item['track']['artists'][0]['name']
            songs.append(f'{artist_name} - {song_name}')

        offset += 50
        time.sleep(0.5)

    return songs

