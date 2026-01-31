import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import io

# Page configuration
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .emotion-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎤 Speech Emotion Recognition</h1>', unsafe_allow_html=True)
st.markdown("### Detect emotions from audio using Deep Learning (CNN Model)")
st.markdown("---")

# Emotion labels
EMOTIONS = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_EMOJI = {
    'angry': '😠', 'calm': '😌', 'disgust': '🤢', 'fear': '😨',
    'happy': '😊', 'neutral': '😐', 'sad': '😢', 'surprise': '😲'
}

@st.cache_resource
def load_model():
    """Load the pre-trained CNN model"""
    try:
        model = keras.models.load_model('Model Files/CNN_model.h5')
        return model
    except:
        st.error("Model file not found. Please ensure 'Model Files/CNN_model.h5' exists.")
        return None

def extract_features(audio_data, sr):
    """Extract audio features: MFCC, Chroma, Mel, ZCR, Spectral Centroid"""
    features = []
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    features.extend(chroma_mean)
    
    # MFCC
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    features.extend(mfccs_mean)
    
    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    mel_mean = np.mean(mel.T, axis=0)
    features.extend(mel_mean)
    
    return np.array(features)

def plot_waveform(audio_data, sr):
    """Plot audio waveform"""
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(audio_data, sr=sr, ax=ax, color='#667eea')
    ax.set_title('Audio Waveform', fontsize=14)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    plt.tight_layout()
    return fig

def plot_mel_spectrogram(audio_data, sr):
    """Plot Mel spectrogram"""
    fig, ax = plt.subplots(figsize=(10, 3))
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel Spectrogram', fontsize=14)
    plt.tight_layout()
    return fig

# Sidebar information
with st.sidebar:
    st.header("📊 About This Project")
    st.markdown("""
    This Speech Emotion Recognition system detects **8 emotions** from audio:
    - 😠 Angry
    - 😌 Calm
    - 🤢 Disgust
    - 😨 Fear
    - 😊 Happy
    - 😐 Neutral
    - 😢 Sad
    - 😲 Surprise
    
    **Model:** CNN (Convolutional Neural Network)
    
    **Accuracy:** ~73%
    
    **Features Used:**
    - MFCC (Mel-frequency cepstral coefficients)
    - Chroma
    - Mel Spectrogram
    """)
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Rahul Raj**")
    st.markdown("[GitHub](https://github.com/rahulrajdev344-code)")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose a WAV or MP3 file",
        type=['wav', 'mp3'],
        help="Upload an audio file to analyze the emotion"
    )

if uploaded_file is not None:
    # Load audio
    with st.spinner("Loading audio..."):
        audio_data, sr = librosa.load(uploaded_file, sr=22050)
    
    st.success(f"✅ Audio loaded! Duration: {len(audio_data)/sr:.2f} seconds")
    
    # Display audio player
    st.audio(uploaded_file)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌊 Waveform")
        fig_wave = plot_waveform(audio_data, sr)
        st.pyplot(fig_wave)
    
    with col2:
        st.subheader("🎨 Mel Spectrogram")
        fig_mel = plot_mel_spectrogram(audio_data, sr)
        st.pyplot(fig_mel)
    
    st.markdown("---")
    
    # Prediction
    if st.button("🔮 Analyze Emotion", use_container_width=True):
        model = load_model()
        
        if model:
            with st.spinner("Extracting features and predicting..."):
                # Extract features
                try:
                    features = extract_features(audio_data, sr)
                    
                    # Ensure correct shape (pad or truncate to 162 features)
                    target_length = 162
                    if len(features) < target_length:
                        features = np.pad(features, (0, target_length - len(features)))
                    else:
                        features = features[:target_length]
                    
                    # Reshape for CNN
                    features = features.reshape(1, -1, 1)
                    
                    # Predict
                    predictions = model.predict(features, verbose=0)
                    predicted_idx = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_idx] * 100
                    predicted_emotion = EMOTIONS[predicted_idx]
                    
                    # Display result
                    st.markdown("---")
                    st.subheader("🎯 Prediction Result")
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        emoji = EMOTION_EMOJI.get(predicted_emotion, '🎭')
                        st.markdown(f"""
                        <div style="text-align: center; padding: 30px; 
                                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    border-radius: 15px; color: white;">
                            <h1 style="font-size: 4rem; margin: 0;">{emoji}</h1>
                            <h2 style="margin: 10px 0;">{predicted_emotion.upper()}</h2>
                            <p style="font-size: 1.2rem;">Confidence: {confidence:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence bars for all emotions
                    st.markdown("### 📊 Confidence Scores for All Emotions")
                    for i, (emotion, prob) in enumerate(zip(EMOTIONS, predictions[0])):
                        emoji = EMOTION_EMOJI.get(emotion, '🎭')
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.write(f"{emoji} {emotion.capitalize()}")
                        with col2:
                            st.progress(float(prob))
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
        else:
            st.warning("Please ensure the model file is available.")

else:
    # Show demo message when no file is uploaded
    st.info("👆 Upload an audio file to get started!")
    
    with st.expander("📖 How to use"):
        st.markdown("""
        1. **Upload** a WAV or MP3 audio file containing speech
        2. **View** the waveform and spectrogram visualizations
        3. **Click** the "Analyze Emotion" button
        4. **See** the predicted emotion with confidence scores
        
        **Tips for best results:**
        - Use clear audio recordings
        - 3-5 seconds of speech works best
        - Minimize background noise
        """)

# Footer
st.markdown("---")
st.markdown(
    """<p style="text-align: center; color: gray;">
    Built with ❤️ using Streamlit | 
    <a href="https://github.com/rahulrajdev344-code/SpeechEmotionRecognition">View on GitHub</a>
    </p>""",
    unsafe_allow_html=True
)
