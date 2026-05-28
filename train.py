import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from cnn import CNNNetwork
from transition_dataset import TransitionDataset

# =====================
# CONFIG
# =====================
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

DRIVE_SAVE_PATH = "C:/Users/ricky/Desktop/saved_model.pth"
DATA_PATH = "C:/Users/ricky/shiny-detector/data"

# =====================
# TRAIN LOOP
# =====================
def train_epoch(model, loader, loss_fn, optimizer, device):

    model.train()
    total_loss = 0

    for i, (x, y) in enumerate(loader):

        x = x.to(device)
        y = y.to(device).unsqueeze(1)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 2 == 0:
            print(f"Batch {i}/{len(loader)}")

    print("loss:", total_loss / len(loader))


def train(model, loader, loss_fn, optimizer, device, epochs):

    best_loss = float("inf")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        loss = train_epoch(model, loader, loss_fn, optimizer, device)

        # optional: could track best model here


    # =====================
    # SAVE MODEL TO GOOGLE DRIVE
    # =====================
    torch.save(model.state_dict(), DRIVE_SAVE_PATH)
    print(f"\n✅ Model saved to Google Drive at:\n{DRIVE_SAVE_PATH}")


# =====================
# MAIN
# =====================
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=44100,
        n_fft=1024,
        hop_length=256,
        n_mels=64
    )

    dataset = TransitionDataset(
        pre_dir = DATA_PATH + "/pre",
        with_drum_dir = DATA_PATH + "/with_drum",
        no_drum_dir = DATA_PATH + "/no_drum",
        mel_transform=mel,
        sample_rate=44100,
        device=device
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = CNNNetwork().to(device)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, loader, loss_fn, optimizer, device, EPOCHS)