import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn

from models.new_teacher import Thesis_AR_Transcriber
from models.student import Student_AR_Transcriber

N_MELS = 229
THETAS = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


class StudentPooled(Student_AR_Transcriber):
    # same student, but the frequency axis is averaged k times before the big linear layer
    def __init__(self, k):
        super().__init__(N_MELS, 256, 24)
        conv = self.acoustic_model
        conv.cnn = nn.Sequential(conv.cnn, nn.AvgPool2d((1, k), ceil_mode=True))
        freq = -(-(N_MELS // 4) // k)
        conv.fc = nn.Sequential(nn.Linear(768 * freq, 256), nn.Dropout(0.25))


def build_model(name):
    if name == 'teacher':
        return Thesis_AR_Transcriber(N_MELS, 1024, 128, 96)
    if name == 'student':
        return Student_AR_Transcriber(N_MELS, 256, 24)
    if name.startswith('student_pool'):
        return StudentPooled(int(name[len('student_pool'):]))
    raise ValueError(name)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def list_files(data_dir, split):
    return sorted(glob.glob(os.path.join(data_dir, split, '*.pt')))


def load_file(path):
    # mmap so a crop doesn't read the whole recording
    return torch.load(path, map_location='cpu', mmap=True)


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, files, seq_len=1024, freq_mask=30, time_mask=50):
        self.files = files
        self.seq_len = seq_len
        self.freq_mask = freq_mask
        self.time_mask = time_mask

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        data = load_file(self.files[i])
        n = data['mel'].shape[0]
        if n >= self.seq_len:
            start = random.randint(0, n - self.seq_len)
            mel = data['mel'][start:start + self.seq_len].float()
            label = data['label'][start:start + self.seq_len].long()
        else:
            pad = self.seq_len - n
            mel = data['mel'].float()
            label = data['label'].long()
            mel = nn.functional.pad(mel, (0, 0, 0, pad), value=float(mel.min()))
            label = nn.functional.pad(label, (0, 0, 0, pad))

        # spec augment, filled with the mean of the window
        fill = mel.mean()
        w = random.randint(0, self.freq_mask)
        f0 = random.randint(0, N_MELS - w)
        mel[:, f0:f0 + w] = fill
        w = random.randint(0, self.time_mask)
        t0 = random.randint(0, self.seq_len - w)
        mel[t0:t0 + w, :] = fill
        return mel, label


class WindowDataset(torch.utils.data.Dataset):
    # fixed non-overlapping windows, the same ones every time
    def __init__(self, files, seq_len=1024, per_file=0, max_files=0):
        if max_files and len(files) > max_files:
            files = sorted(random.Random(0).sample(files, max_files))
        self.seq_len = seq_len
        self.items = []
        for f in files:
            n = load_file(f)['mel'].shape[0] // seq_len
            starts = [i * seq_len for i in range(n)]
            if per_file and len(starts) > per_file:
                idx = np.linspace(0, len(starts) - 1, per_file).round().astype(int)
                starts = [starts[i] for i in idx]
            self.items += [(f, s) for s in starts]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, s = self.items[i]
        data = load_file(f)
        return data['mel'][s:s + self.seq_len].float(), data['label'][s:s + self.seq_len].long()


@torch.no_grad()
def predict(model, loader, device):
    # free running: the labels are never given to the model
    model.eval()
    for mel, label in loader:
        probs = torch.softmax(model(mel.to(device)), dim=-1).cpu()
        yield probs[..., 3].numpy(), label.numpy()


def peaks(p_onset, theta):
    # onset = local maximum of the onset probability in time, above theta
    left = np.pad(p_onset, ((0, 0), (1, 0), (0, 0)))[:, :-1]
    right = np.pad(p_onset, ((0, 0), (0, 1), (0, 0)))[:, 1:]
    return (p_onset >= left) & (p_onset > right) & (p_onset >= theta)


def match(pred, gt, tol):
    # greedy, each predicted onset takes the first free reference onset within tol frames
    used = np.zeros(len(gt), bool)
    hits = 0
    for p in pred:
        j = int(np.searchsorted(gt, p - tol))
        while j < len(gt) and gt[j] <= p + tol:
            if not used[j]:
                used[j] = True
                hits += 1
                break
            j += 1
    return hits


class OnsetCounter:
    def __init__(self, tol):
        self.tol = tol
        self.tp = self.fp = self.fn = 0

    def update(self, onsets, label):
        # onsets: bool (batch, time, 88), label: int (batch, time, 88), 3 = onset
        gt = label == 3
        for b in range(onsets.shape[0]):
            for p in range(88):
                pred_frames = np.flatnonzero(onsets[b, :, p])
                gt_frames = np.flatnonzero(gt[b, :, p])
                if len(pred_frames) == 0 and len(gt_frames) == 0:
                    continue
                hits = match(pred_frames, gt_frames, self.tol)
                self.tp += hits
                self.fp += len(pred_frames) - hits
                self.fn += len(gt_frames) - hits

    def result(self):
        p = self.tp / max(1, self.tp + self.fp)
        r = self.tp / max(1, self.tp + self.fn)
        return {'precision': p, 'recall': r, 'f1': 2 * p * r / max(1e-9, p + r),
                'tp': self.tp, 'fp': self.fp, 'fn': self.fn}


def sweep_thetas(model, loader, device, tol=1):
    # one pass over the data, every threshold at once
    counters = {t: OnsetCounter(tol) for t in THETAS}
    for p_onset, label in predict(model, loader, device):
        for t, c in counters.items():
            c.update(peaks(p_onset, t), label)
    return {t: c.result() for t, c in counters.items()}
