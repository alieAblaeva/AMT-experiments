import argparse
import csv
import glob
import io
import json
import os
import sys
import zipfile

import librosa
import mir_eval
import numpy as np
import pretty_midi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import peaks

HOP, SR = 512, 16000

# note level onset f1 like in the papers: 50 ms tolerance, 50 cents, offsets are ignored.
# the predicted time is the middle of the 32 ms frame, the reference times come from the midi files
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', required=True)      # npz files of run_inference.py
    ap.add_argument('--theta', type=float, required=True)
    ap.add_argument('--zip', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    z = zipfile.ZipFile(args.zip)
    csv_name = next(n for n in z.namelist() if n.endswith('maestro-v3.0.0.csv'))
    prefix = csv_name[:-len('maestro-v3.0.0.csv')]
    rows = csv.DictReader(io.TextIOWrapper(io.BytesIO(z.read(csv_name)), encoding='utf-8'))
    midi_of = {os.path.basename(r['audio_filename'])[:-4]: r['midi_filename'] for r in rows if r['split'] == 'test'}

    scores, tp, n_pred, n_ref = [], 0, 0, 0
    for f in sorted(glob.glob(os.path.join(args.preds, '*.npz'))):
        midi = pretty_midi.PrettyMIDI(io.BytesIO(z.read(prefix + midi_of[os.path.basename(f)[:-4]])))
        ref = np.array([(n.start, n.end, n.pitch) for i in midi.instruments if not i.is_drum
                        for n in i.notes if 21 <= n.pitch <= 108], float)
        ref_int, ref_hz = ref[:, :2], librosa.midi_to_hz(ref[:, 2])

        frames, pitches = np.nonzero(peaks(np.load(f)['p_onset'].astype(np.float32)[None], args.theta)[0])
        start = (frames + 0.5) * HOP / SR
        est_int = np.stack([start, start + 0.05], 1) if len(start) else np.zeros((0, 2))
        est_hz = librosa.midi_to_hz(pitches + 21) if len(start) else np.zeros(0)

        p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_int, ref_hz, est_int, est_hz, onset_tolerance=0.05, offset_ratio=None)
        scores.append((p, r, f1))
        tp += len(mir_eval.transcription.match_notes(ref_int, ref_hz, est_int, est_hz, onset_tolerance=0.05, offset_ratio=None))
        n_pred += len(est_hz)
        n_ref += len(ref_hz)

    p, r = tp / max(1, n_pred), tp / max(1, n_ref)
    result = {'recordings': len(scores), 'theta': args.theta,
              'mean_over_recordings': dict(zip(('precision', 'recall', 'f1'), np.mean(scores, 0).tolist())),
              'pooled': {'precision': p, 'recall': r, 'f1': 2 * p * r / max(1e-9, p + r)}}
    json.dump(result, open(args.out, 'w'), indent=1)
    print(result)
