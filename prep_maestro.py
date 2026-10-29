import argparse
import csv
import io
import os
import zipfile
from multiprocessing import Pool

import librosa
import numpy as np
import pretty_midi
import torch

SR = 16000
HOP = 512
N_MELS = 229


def midi_to_label(midi, n_frames):
    # 0 silence, 1 offset, 2 sustain, 3 onset
    label = np.zeros((n_frames, 88), dtype=np.uint8)
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in sorted(inst.notes, key=lambda n: n.start):
            p = note.pitch - 21
            if not 0 <= p < 88:
                continue
            start = max(0, min(int(note.start * SR / HOP), n_frames - 1))
            end = max(0, min(int(note.end * SR / HOP), n_frames - 1))
            if start >= end:
                continue
            label[end, p] = 1
            label[start + 1:end, p] = 2
            label[start, p] = 3
    return label


def log_mel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP, center=False)
    return np.log(np.clip(mel, 1e-5, None))


zf = None


def process(job):
    global zf
    cfg, row = job
    out = os.path.join(cfg['out'], row['split'], os.path.basename(row['audio_filename'])[:-4] + '.pt')
    if os.path.exists(out):
        return out, 'skip'
    try:
        if zf is None:
            zf = zipfile.ZipFile(cfg['zip'])
        y, _ = librosa.load(io.BytesIO(zf.read(cfg['prefix'] + row['audio_filename'])), sr=SR)
        midi = pretty_midi.PrettyMIDI(io.BytesIO(zf.read(cfg['prefix'] + row['midi_filename'])))
        mel = log_mel(y)
        label = midi_to_label(midi, mel.shape[1])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save({'mel': torch.from_numpy(mel.T).half(), 'label': torch.from_numpy(label)}, out + '.tmp')
        os.replace(out + '.tmp', out)
        return out, 'ok'
    except Exception as e:
        return out, f'error {e}'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True)          # maestro-v3.0.0.zip, read without unpacking
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    z = zipfile.ZipFile(args.zip)
    csv_name = next(n for n in z.namelist() if n.endswith('maestro-v3.0.0.csv'))
    prefix = csv_name[:-len('maestro-v3.0.0.csv')]
    rows = list(csv.DictReader(io.TextIOWrapper(io.BytesIO(z.read(csv_name)), encoding='utf-8')))
    cfg = {'zip': args.zip, 'prefix': prefix, 'out': args.out}

    errors = 0
    with Pool(args.workers) as pool:
        for i, (path, status) in enumerate(pool.imap_unordered(process, [(cfg, r) for r in rows])):
            if status.startswith('error'):
                errors += 1
                print(path, status)
            if (i + 1) % 25 == 0:
                print(f'{i + 1}/{len(rows)}')
    print('done, errors:', errors)
