import spotipy
from spotipy.oauth2 import SpotifyOAuth

# fill this later
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    redirect_uri="http://localhost:8888/callback",
    scope="user-library-read"
))

results = sp.current_user_saved_tracks(limit=5)
for idx, item in enumerate(results['items']):
    track = item['track']
    print(f"{idx+1}. {track['artists'][0]['name']} – {track['name']}")
