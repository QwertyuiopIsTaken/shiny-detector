from tkinter import messagebox
import pyautogui as ag
import numpy as np
import soundcard as sc
import soundcard.mediafoundation as mf
import time
import threading
import warnings

warnings.filterwarnings(
    "ignore",
    category=sc.mediafoundation.SoundcardRuntimeWarning
)

class DetectorController:
    def __init__(self, state, detector, matcher):
        self.state = state
        self.detector = detector
        self.matcher = matcher

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

        shiny_length_sec = 0.7
        chunk_len = int(self.detector.sr * shiny_length_sec)

        print("Listening for sound")

        while self.state.run:
            audio_chunk = self.state.rec.record(numframes=chunk_len)
            audio_chunk = np.nan_to_num(audio_chunk)

            if self.detector.detect(audio_chunk):
                messagebox.showinfo(
                    "Tone Detector",
                    "Sound detected!"
                )
                self.state.run = False
                return
            else:
                pos = self.matcher.locate()
                if pos:
                    time.sleep(2.5)
                    ag.click(pos)

            time.sleep(0.005)

    def start(self):
        if self.state.run:
            return

        self.state.run = True
        mic = sc.all_microphones(include_loopback=True)[0]
        self.state.rec = mic.recorder(samplerate=self.detector.sr)
        self.state.rec.__enter__()

        threading.Thread(
            target=self.detectionThread,
            daemon=True
        ).start()

    def stop(self):
        if not self.state.run:
            return

        self.state.run = False
        if self.state.rec:
            self.state.rec.__exit__(None, None, None)
            self.state.rec = None