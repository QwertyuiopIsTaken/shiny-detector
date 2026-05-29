import pyautogui as ag
import cv2
import os
import sys
import numpy as np

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return relative_path

class ScreenMatcher:
    def __init__(self):
        scales = np.arange(1, 0.7, -0.1).tolist()
        original = cv2.imread(resource_path("assets/run.png"))

        self.images = [
            cv2.resize(
                original,
                None,
                fx=s,
                fy=s,
                interpolation=cv2.INTER_LINEAR
            )
            for s in scales
        ]

        self.bestSize = None

    def locate(self):
        if self.bestSize is not None:
            try:
                pos = ag.locateOnScreen(self.bestSize, confidence=0.9)
                if pos:
                    return pos
            except ag.ImageNotFoundException:
                self.bestSize = None

        for img in self.images:
            try:
                pos = ag.locateOnScreen(img, confidence=0.9)
                if pos:
                    self.bestSize = img
                    return pos
            except ag.ImageNotFoundException:
                pass

        return None