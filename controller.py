from tkinter import messagebox
import pyautogui as ag
import numpy as np
import soundcard as sc
import soundcard.mediafoundation as mf
import time
import threading
import warnings
import torch
import torchaudio

from cnn import CNNNetwork

warnings.filterwarnings(
    "ignore",
    category=sc.mediafoundation.SoundcardRuntimeWarning
)

class DetectorController:
    def __init__(self, state, matcher):
        self.state = state
        self.matcher = matcher

        # =====================
        # MODEL CONFIG
        # =====================
        self.SAMPLE_RATE = 44100
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # load model
        self.model = CNNNetwork().to(self.DEVICE)
        self.model.load_state_dict(
            torch.load("./CNNmodel.pth", map_location=self.DEVICE)
        )
        self.model.eval()

        # mel transform (must match training)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE,
            n_fft=1024,
            hop_length=256,
            n_mels=64
        )

    # =====================
    # AUDIO to CNN
    # =====================
    def predict_chunk(self, chunk_np):
        # numpy -> torch
        waveform = torch.from_numpy(chunk_np.T).float()

        # mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # normalize
        waveform = waveform / (waveform.abs().max() + 1e-9)

        # fix length (IMPORTANT: match training)
        target_len = int(1.0 * self.SAMPLE_RATE)  # <-- adjust if needed
        if waveform.shape[1] >= target_len:
            waveform = waveform[:, :target_len]
        else:
            pad = target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        # mel
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-6)

        # batch dimension
        mel = mel.unsqueeze(0).to(self.DEVICE)

        with torch.no_grad():
            output = self.model(mel)
            prob = output.item()

        print("CNN prob:", prob)

        return prob > 0.5

    # =====================
    # MAIN LOOP
    # =====================
    def detectionThread(self):
        mics = sc.all_microphones(include_loopback=True)
        if not mics:
            messagebox.showerror(
                "Tone Detector",
                "No loopback device found!"
            )
            return

        mic = mics[0]
        print("Using:", mic)

        mf.numpy.fromstring = lambda data, dtype: np.frombuffer(data, dtype=dtype)

        # =====================
        # BUFFER SETTINGS
        # =====================
        WINDOW_SEC = 4.0   # must match model training
        STEP_SEC   = 1   # overlap step (latency control)

        window_len = int(WINDOW_SEC * self.SAMPLE_RATE)
        step_len   = int(STEP_SEC * self.SAMPLE_RATE)

        # rolling buffer (start empty)
        buffer = np.zeros((0, 2), dtype=np.float32)

        print("Listening for sound (overlapping windows)")

        while self.state.run:
            # record small chunk (step size)
            chunk = self.state.rec.record(numframes=step_len)
            chunk = np.nan_to_num(chunk)

            # append to buffer
            buffer = np.vstack((buffer, chunk))

            # keep only last WINDOW_SEC seconds
            if len(buffer) > window_len:
                buffer = buffer[-window_len:]

            # only run model when buffer is full
            if len(buffer) == window_len:
                if self.predict_chunk(buffer):
                    messagebox.showinfo(
                        "Tone Detector",
                        "Sound detected!"
                    )
                    self.state.run = False
                    return

            # fallback action
            pos = self.matcher.locate()
            if pos:
                time.sleep(2.5)
                ag.click(pos)

            time.sleep(0.005)

    # =====================
    # CONTROL
    # =====================
    def start(self):
        print("START")

        if self.state.run:
            return

        self.state.run = True
        mic = sc.all_microphones(include_loopback=True)[0]

        self.state.rec = mic.recorder(samplerate=self.SAMPLE_RATE)
        self.state.rec.__enter__()

        threading.Thread(
            target=self.detectionThread,
            daemon=True
        ).start()

    def stop(self):
        print("STOP")

        if not self.state.run:
            return

        self.state.run = False

        if self.state.rec:
            self.state.rec.__exit__(None, None, None)
            self.state.rec = None