from flask import Flask, render_template, request, url_for
import librosa
import librosa.display
import os
from werkzeug.utils import secure_filename
import uuid
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/', methods=['GET', 'POST'])
def index():
    bpm = None
    beat_times = []
    duration = 0
    unique_name = None
    key_name = None
    waveform_name = None

    if request.method == 'POST':
        file = request.files['audio']
        analise = request.form.get('analise')
        
        if file:
            filename = secure_filename(file.filename)
            unique_name = f"preview_{uuid.uuid4().hex}.mp3"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            preview_path = os.path.join('static', unique_name)
            os.replace(filepath, preview_path)

            y, sr = librosa.load(preview_path)

            # Geração da imagem da waveform (sempre)
            waveform_path = os.path.join('static', f"waveform_{uuid.uuid4().hex}.png")
            plt.figure(figsize=(10, 2))
            librosa.display.waveshow(y, sr=sr, color='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(waveform_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            waveform_name = os.path.basename(waveform_path)

            if analise in ['bpm', 'ambos']:
                tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
                beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
                duration = librosa.get_duration(y=y, sr=sr)
                bpm = round(float(tempo), 2)

            if analise in ['tonalidade', 'ambos']:
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

                major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
                minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

                chroma_mean = np.mean(chroma, axis=1)

                def estimate_key(chroma_mean, profile):
                    return np.argmax([np.corrcoef(np.roll(profile, i), chroma_mean)[0, 1]
                                      for i in range(12)])

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

    return render_template('index.html',
                           bpm=bpm,
                           beat_times=beat_times,
                           duration=duration,
                           preview_name=unique_name,
                           key_name=key_name,
                           waveform_name=waveform_name)

if __name__ == '__main__':
    app.run(debug=True)
