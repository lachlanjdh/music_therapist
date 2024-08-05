import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from googleapiclient.discovery import build
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load NLP model and tokenizer
model = tf.keras.models.load_model('your_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Initialize YouTube API credentials
YOUTUBE_API_KEY = 'AIzaSyCZhY3TP1aelp10h83LasnYz-Gpv2eu56w'
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Function to predict emotion and suggest playlists
def suggest_playlist(text):
    # Tokenize and pad the input text
    seq = tokenizer.texts_to_sequences([text])
    pad_seq = pad_sequences(seq, maxlen=100)
    
    # Predict emotion
    pred = np.argmax(model.predict(pad_seq), axis=1)
    emotion = label_encoder.inverse_transform(pred)[0]
    
    logging.debug(f'Detected emotion: {emotion}')
    
    # Define search queries based on detected emotion
    search_queries = {
        'happy': 'happy upbeat songs playlist',
        'sadness': 'sad emotional songs playlist',
        'anger': 'angry rock songs playlist',
        'relaxed': 'relaxing chill songs playlist',
        'nervous': 'calming relaxing songs playlist',
        'worried': 'calming study songs playlist',
    }
    
    # Get search query based on emotion
    search_query = search_queries.get(emotion, 'popular songs playlist')
    
    logging.debug(f'Search query: {search_query}')
    
    # Search for playlists on YouTube
    search_response = youtube.search().list(
        q=search_query,
        part='snippet',
        type='playlist',
        maxResults=1
    ).execute()
    
    logging.debug(f'YouTube API response: {search_response}')
    
    if 'items' in search_response and len(search_response['items']) > 0:
        playlist_title = search_response['items'][0]['snippet']['title']
        playlist_url = f"https://www.youtube.com/playlist?list={search_response['items'][0]['id']['playlistId']}"
    else:
        playlist_title = 'No playlist found'
        playlist_url = '#'
    
    return emotion, playlist_title, playlist_url

# Streamlit app
st.title('MyMusicTherapist')

# User input
text = st.text_area('Enter your thoughts:', '')

if st.button('Submit'):
    if text:
        emotion, playlist_title, playlist_url = suggest_playlist(text)
        st.write(f"**Detected Emotion:** {emotion}")
        st.write(f"**Suggested Playlist:** [ {playlist_title} ]({playlist_url})")
    else:
        st.error("Please enter some text.")
