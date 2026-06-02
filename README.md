# shiny-detector

An AI powered sound detector. Similar to the earlier version, it automates clicking when it detects a sound.

The controller creates a sliding window of 4 seconds with steps of 1 second. In each iteration, a sample is recorded and fed into the CNN network. The network utilizes a pre-trained model that was designed to listen to a specific frequency using a mel spectrogram.

When a specific sound is detected, in this case the shiny sound effect, the automated clicking terminates, allowing users to take control.

## Installation

1. Clone the repository:
```bash
git clone git@github.com:QwertyuiopIsTaken/shiny-detector.git
```
2. Install dependencies:
```bash
pip install numpy soundcard pyautogui opencv-python keyboard torch torchaudio
```
3. Run the program:
```bash
python main.py
```

## Controls

* **Start button** or hotkey: Begin detection
* **Stop button** or hotkey: Stop detection
* **Chroma slider**: Deprecated
* **Threshold slider**: Deprecated