import argparse
import csv
import glob
import io
import os
import sys
import zipfile

import librosa
import numpy as np
import pretty_midi
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prep_maestro import SR, HOP, log_mel


def label_and_notes(midi, n_frames):
    # same rule as prep_maestro.midi_to_label, plus the notes that got an onset: (pitch, start, end, velocity, seconds)
    label = np.zeros((n_frames, 88), np.uint8)
    notes = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for n in sorted(inst.notes, key=lambda n: n.start):
            p = n.pitch - 21
            if not 0 <= p < 88:
                continue
            s = max(0, min(int(n.start * SR / HOP), n_frames - 1))
            e = max(0, min(int(n.end * SR / HOP), n_frames - 1))
            if s >= e:
                continue
            label[e, p] = 1
            label[s + 1:e, p] = 2
            label[s, p] = 3
            notes.append((p, s, e, n.velocity, n.end - n.start))
    return label, np.array(notes, float).reshape(-1, 5)


def from_smd(src, out):
    # folder with wav + mid pairs
    for wav in sorted(glob.glob(os.path.join(src, '*.wav'))):
        mid = wav[:-4] + '.mid'
        y, _ = librosa.load(wav, sr=SR)
        mel = log_mel(y)
        label, notes = label_and_notes(pretty_midi.PrettyMIDI(mid), mel.shape[1])
        torch.save({'mel': torch.from_numpy(mel.T).half(), 'label': torch.from_numpy(label),
                    'notes': torch.from_numpy(notes)}, os.path.join(out, os.path.basename(wav)[:-4] + '.pt'))


def from_maestro(pre_test, zip_path, out):
    # test files already preprocessed by prep_maestro.py, only the notes are added
    z = zipfile.ZipFile(zip_path)
    csv_name = next(n for n in z.namelist() if n.endswith('maestro-v3.0.0.csv'))
    prefix = csv_name[:-len('maestro-v3.0.0.csv')]
    rows = csv.DictReader(io.TextIOWrapper(io.BytesIO(z.read(csv_name)), encoding='utf-8'))
    midi_of = {os.path.basename(r['audio_filename'])[:-4]: r['midi_filename'] for r in rows if r['split'] == 'test'}
    for f in sorted(glob.glob(os.path.join(pre_test, '*.pt'))):
        name = os.path.basename(f)[:-3]
        data = torch.load(f)
        midi = pretty_midi.PrettyMIDI(io.BytesIO(z.read(prefix + midi_of[name])))
        label, notes = label_and_notes(midi, data['mel'].shape[0])
        assert (torch.from_numpy(label) == data['label']).all(), name
        torch.save({'mel': data['mel'], 'label': data['label'], 'notes': torch.from_numpy(notes)}, os.path.join(out, name + '.pt'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--smd')                       # folder with wav + mid
    ap.add_argument('--maestro_test')              # maestro-pre/test
    ap.add_argument('--zip')                       # maestro-v3.0.0.zip
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.smd:
        from_smd(args.smd, args.out)
    else:
        from_maestro(args.maestro_test, args.zip, args.out)
