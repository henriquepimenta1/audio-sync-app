import streamlit as st
import librosa
import numpy as np
import json
import matplotlib.pyplot as plt
import librosa.display
from io import BytesIO

st.set_page_config(page_title="Analisador de Música", layout="centered")
st.title("🎧 Detector de BPM, Tonalidade e Mudanças")

uploaded_file = st.file_uploader("Envie um arquivo de áudio", type=["mp3", "wav"])

if uploaded_file:
    y, sr = librosa.load(uploaded_file, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    st.audio(uploaded_file, format='audio/mp3')

    # BPM e batidas
   # BPM e batidas
try:
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    st.markdown(f"**BPM estimado:** `{round(tempo, 2)}`")
    st.markdown(f"**Batidas detectadas:** {len(beat_times)}")
except Exception as e:
    st.warning("⚠️ Não foi possível detectar o BPM.")
    beat_times = []
    tempo = None


    # Tonalidade
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    def estimate_key(chroma_mean, profile):
        return np.argmax([np.corrcoef(np.roll(profile, i), chroma_mean)[0, 1] for i in range(12)])

    major_score = estimate_key(chroma_mean, major_profile)
    minor_score = estimate_key(chroma_mean, minor_profile)

    if np.corrcoef(np.roll(major_profile, major_score), chroma_mean)[0, 1] > \
       np.corrcoef(np.roll(minor_profile, minor_score), chroma_mean)[0, 1]:
        mode = 'maior'
        tonic = major_score
    else:
        mode = 'menor'
        tonic = minor_score

    tonalidades = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#',
                   'G', 'G#', 'A', 'A#', 'B']
    key_name = f"{tonalidades[tonic]} {mode}"
    st.markdown(f"**Tonalidade estimada:** `{key_name}`")

    # Mudanças de tonalidade
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    segments = librosa.segment.agglomerative(tonnetz.T, k=6)
    boundaries = np.flatnonzero(np.diff(segments)) + 1
    tonal_changes = librosa.frames_to_time(boundaries, sr=sr).tolist()
    st.markdown(f"**Mudanças de tonalidade detectadas:** `{len(tonal_changes)}`")

    # Plot waveform
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='gray')
    ax.set(title='Waveform')
    ax.axis('off')
    st.pyplot(fig)

    # Exportar JSON
    output = {
        "bpm": round(tempo, 2),
        "tonalidade": key_name,
        "batidas": beat_times,
        "tonal_changes": tonal_changes
    }
    json_bytes = BytesIO()
    json_bytes.write(json.dumps(output, indent=2).encode())
    json_bytes.seek(0)
    st.download_button(
        label="📥 Baixar JSON de sincronização",
        data=json_bytes,
        file_name="sync_data.json",
        mime="application/json"
    )
