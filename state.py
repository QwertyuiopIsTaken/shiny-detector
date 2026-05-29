class AppState:
    def __init__(self):
        self.run = False
        self.rec = None
        self.score_buffer = [0.0]
        self.hotkeys = {"Start": "y", "Stop": "z"}

        # GUI-controlled
        self.chromaScale = None
        self.combScale = None