import torch
import torchaudio

from cnn import CNNNetwork


# =====================
# CONFIG
# =====================
MODEL_PATH = "./saved_model.pth"
SAMPLE_RATE = 44100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================
# LOAD MODEL
# =====================
model = CNNNetwork().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# =====================
# MEL TRANSFORM (same as training)
# =====================
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=256,
    n_mels=64
)

# =====================
# AUDIO PREPROCESSING
# =====================
def fix_length(waveform, seconds=5):
    target_len = int(seconds * SAMPLE_RATE)

    if waveform.shape[1] >= target_len:
        waveform = waveform[:, :target_len]  # start at 0
    else:
        pad = target_len - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))

    return waveform


def to_mel(waveform):
    mel = mel_transform(waveform)
    mel = torch.log(mel + 1e-6)
    return mel


# =====================
# PREDICTION
# =====================
def predict_chunk(chunk_np):
    # convert numpy → torch
    waveform = torch.from_numpy(chunk_np.T).float()  # (channels, samples)

    # mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # normalize
    waveform = waveform / (waveform.abs().max() + 1e-9)

    # fix length
    waveform = fix_length(waveform)

    # mel
    mel = to_mel(waveform)

    mel = mel.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(mel)
        prob = output.item()

    print("CNN prob:", prob)

    return prob > 0.5


# =====================
# RUN
# =====================
'''
if __name__ == "__main__":
    file_path = "placeholder.mp3"  # change this

    label, prob = predict(file_path)

    print(f"\nPrediction: {label}")
    print(f"Confidence: {prob:.4f}")
'''