import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset


class TransitionDataset(Dataset):

    def __init__(self, pre_dir, with_drum_dir, no_drum_dir,
                 mel_transform, sample_rate, device):

        self.pre_files = [os.path.join(pre_dir, f) for f in os.listdir(pre_dir)]
        self.with_drum_files = [os.path.join(with_drum_dir, f) for f in os.listdir(with_drum_dir)]
        self.no_drum_files = [os.path.join(no_drum_dir, f) for f in os.listdir(no_drum_dir)]

        self.mel_transform = mel_transform
        self.sample_rate = sample_rate
        self.device = device

        self.resamplers = {}

    def export_sample(self, idx, save_dir="/content/debug_audio"):
      os.makedirs(save_dir, exist_ok=True)

      torch.manual_seed(idx)
      random.seed(idx)

      total_seconds = 5
      total_len = int(total_seconds * self.sample_rate)

      # choose class
      if random.random() < 0.5:
          label = 1
          post_file = random.choice(self.with_drum_files)
      else:
          label = 0
          post_file = random.choice(self.no_drum_files)

      pre_file = random.choice(self.pre_files)

      # load
      pre_wave = self._load_audio(pre_file)
      post_wave = self._load_audio(post_file)

      # -------- PRE --------
      pre_seconds = random.uniform(1.0, 1.5)
      pre_len = int(pre_seconds * self.sample_rate)

      pre_wave = self._crop_or_pad(pre_wave, pre_len)

      # -------- POST --------
      post_len = total_len - pre_len

      min_post_len = int(2.5 * self.sample_rate)
      if post_len < min_post_len:
          post_len = min_post_len
          pre_len = total_len - post_len
          pre_wave = self._crop_or_pad(pre_wave, pre_len)

      # 🔥 CRITICAL: anchored crop
      post_wave = self._fix_post_length(post_wave, post_len)

      # combine
      combined = torch.cat([pre_wave, post_wave], dim=1)

      # save
      torchaudio.save(f"{save_dir}/pre_{idx}.wav", pre_wave.cpu(), self.sample_rate)
      torchaudio.save(f"{save_dir}/post_{idx}.wav", post_wave.cpu(), self.sample_rate)
      torchaudio.save(f"{save_dir}/combined_{idx}.wav", combined.cpu(), self.sample_rate)

      print(f"Saved sample {idx} (label={label})")

    # -------------------------
    # helpers
    # -------------------------

    def _get_resampler(self, orig_sr):
        if orig_sr not in self.resamplers:
            self.resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_sr, self.sample_rate
            )
        return self.resamplers[orig_sr]

    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path)

        # mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample
        if sr != self.sample_rate:
            waveform = self._get_resampler(sr)(waveform)

        return waveform

    def _random_pre_chunk(self, waveform):
        """Take random 5–10 second chunk from pre audio"""
        min_len = int(5 * self.sample_rate)
        max_len = int(10 * self.sample_rate)

        target_len = random.randint(min_len, max_len)

        if waveform.shape[1] > target_len:
            start = random.randint(0, waveform.shape[1] - target_len)
            waveform = waveform[:, start:start + target_len]
        else:
            pad = target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        return waveform

    def _to_mel(self, waveform):
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-6)
        return mel

    # -------------------------
    # main
    # -------------------------

    def __len__(self):
        return 2000  # arbitrary large number (infinite sampling)

    def _crop_or_pad(self, waveform, target_len):

      current_len = waveform.shape[1]

      if current_len >= target_len:
          # random crop (this is OK for PRE audio)
          start = random.randint(0, current_len - target_len)
          waveform = waveform[:, start:start + target_len]
      else:
          # pad if too short
          pad_total = target_len - current_len
          pad_left = random.randint(0, pad_total)
          pad_right = pad_total - pad_left
          waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right))

      return waveform

    def __getitem__(self, idx):

      total_seconds = 5
      total_len = int(total_seconds * self.sample_rate)

      # choose class
      if random.random() < 0.5:
          label = 1
          post_file = random.choice(self.with_drum_files)
      else:
          label = 0
          post_file = random.choice(self.no_drum_files)

      pre_file = random.choice(self.pre_files)

      pre_wave = self._load_audio(pre_file)
      post_wave = self._load_audio(post_file)

      # -------- PRE --------
      pre_seconds = random.uniform(1.0, 1.5)  # slightly tighter
      pre_len = int(pre_seconds * self.sample_rate)

      pre_wave = self._crop_or_pad(pre_wave, pre_len)

      # -------- POST --------
      post_len = total_len - pre_len

      # 🔥 IMPORTANT: ensure post is long enough
      min_post_len = int(2.5 * self.sample_rate)

      if post_len < min_post_len:
          post_len = min_post_len
          pre_len = total_len - post_len
          pre_wave = self._crop_or_pad(pre_wave, pre_len)

      # 🔥 CRITICAL FIX: NO RANDOM START
      post_wave = self._fix_post_length(post_wave, post_len)

      # concatenate
      waveform = torch.cat([pre_wave, post_wave], dim=1)

      mel = self._to_mel(waveform)

      return mel, torch.tensor(label, dtype=torch.float32)

    def _fix_post_length(self, waveform, target_len):

      current_len = waveform.shape[1]

      if current_len >= target_len:
          # ✅ ALWAYS start at 0
          waveform = waveform[:, :target_len]
      else:
          pad = target_len - current_len
          waveform = torch.nn.functional.pad(waveform, (0, pad))

      return waveform