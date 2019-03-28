import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

wav, sr = librosa.load("./bbad4n.wav")

plt.figure(figsize=(10, 4))
librosa.display.waveplot(wav, sr=sr)
plt.title('Audio waveplot')
plt.tight_layout()
plt.show()

D = np.abs(librosa.stft(wav))**2
S = librosa.feature.melspectrogram(S=D)
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max),y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()

stft = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(stft, y_axis='linear', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')
plt.tight_layout()
plt.show()