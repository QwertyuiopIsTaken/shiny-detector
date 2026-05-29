import numpy as np
import librosa
from scipy.spatial.distance import cosine
import os
import sys

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return relative_path

class ShinyDetector:
    def __init__(self, state):
        self.state = state
        self.sr = 44100

        # Load sound reference
        shiny_ref, _ = librosa.load(
            resource_path("assets/gleaming.wav"),
            sr=self.sr
        )

        # Convert to mono
        if shiny_ref.ndim > 1:
            shiny_ref = shiny_ref.mean(axis=1)

        # Extract MFCC
        mfcc_ref = librosa.feature.mfcc(
            y=shiny_ref,
            sr=self.sr,
            n_mfcc=20
        )[1:, :]
        self.mfcc_ref = mfcc_ref.mean(axis=1)

        # Extract Chroma
        chroma_ref = librosa.feature.chroma_stft(
            y=shiny_ref,
            sr=self.sr
        )
        self.chroma_ref = chroma_ref.mean(axis=1)

    def detect(self, chunk):
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)

        chunk = chunk.astype(np.float32)
        chunk = np.nan_to_num(chunk)

        if np.max(np.abs(chunk)) < 1e-6:
            return False
        if not np.isfinite(chunk).all():
            return False

        mfcc = librosa.feature.mfcc(
            y=chunk,
            sr=self.sr,
            n_mfcc=20
        )[1:, :]

        energy = librosa.feature.rms(y=chunk)[0]
        peak_frame = np.argmax(energy)
        mfcc_peak = mfcc[:, peak_frame]

        if np.all(mfcc_peak == 0):
            return False

        chroma = librosa.feature.chroma_stft(
            y=chunk,
            sr=self.sr
        )

        chroma_peak = chroma[:, peak_frame]

        if np.all(chroma_peak == 0):
            return False

        score_mfcc = 1 - cosine(self.mfcc_ref, mfcc_peak)
        score_chroma = 1 - cosine(self.chroma_ref, chroma_peak)

        score = (
            score_mfcc * 0.1
            + score_chroma * self.state.chromaScale.get()
        )

        self.state.score_buffer.append(score)
        if len(self.state.score_buffer) > 3:
            self.state.score_buffer.pop(0)

        smoothed = np.mean(self.state.score_buffer)

        print(
            f"MFCC = {score_mfcc:.3f}, "
            f"Chroma = {score_chroma:.3f}, "
            f"Combined = {smoothed:.3f}"
        )

        return (
            smoothed > self.state.combScale.get()
            and score_mfcc > 0.4
        )