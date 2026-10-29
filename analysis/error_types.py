import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import peaks, match

OCTAVE = (-24, -12, 12, 24)
FIFTH = (-19, -7, 7, 19)
SEMITONE = (-2, -1, 1, 2)


def near(x, tol):
    # a frame counts if there is an onset within tol frames
    y = x.copy()
    for s in range(1, tol + 1):
        y[s:] |= x[:-s]
        y[:-s] |= x[s:]
    return y


def other_pitch(arr, t, p, shifts):
    return any(0 <= p + d < 88 and arr[t, p + d] for d in shifts)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--preds', required=True)
    ap.add_argument('--theta', type=float, required=True)
    ap.add_argument('--tol', type=int, default=1)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    c = dict(tp=0, fp=0, fn=0, fp_octave=0, fp_fifth=0, fp_semitone=0, fp_double=0, fp_other=0,
             fn_octave=0, fn_semitone=0, fn_nothing=0)
    reg = {r: dict(tp=0, fp=0, fn=0) for r in ('bass', 'mid', 'treble')}
    register = lambda p: 'bass' if p <= 26 else ('mid' if p <= 62 else 'treble')

    for f in sorted(glob.glob(os.path.join(args.cache, '*.pt'))):
        label = torch.load(f)['label'].numpy()
        p_onset = np.load(os.path.join(args.preds, os.path.basename(f)[:-3] + '.npz'))['p_onset'].astype(np.float32)
        pred = peaks(p_onset[None], args.theta)[0]
        gt = label == 3
        gt_near, pred_near = near(gt, args.tol), near(pred, args.tol)

        for p in range(88):
            pf = np.flatnonzero(pred[:, p])
            gf = np.flatnonzero(gt[:, p])
            hit = np.zeros(len(pf), bool)
            used = np.zeros(len(gf), bool)
            for i, t in enumerate(pf):
                j = int(np.searchsorted(gf, t - args.tol))
                while j < len(gf) and gf[j] <= t + args.tol:
                    if not used[j]:
                        used[j] = hit[i] = True
                        break
                    j += 1
            c['tp'] += hit.sum()
            reg[register(p)]['tp'] += int(hit.sum())

            for t in pf[~hit]:
                c['fp'] += 1
                reg[register(p)]['fp'] += 1
                if other_pitch(gt_near, t, p, OCTAVE):
                    c['fp_octave'] += 1
                elif other_pitch(gt_near, t, p, FIFTH):
                    c['fp_fifth'] += 1
                elif other_pitch(gt_near, t, p, SEMITONE):
                    c['fp_semitone'] += 1
                elif hit[np.abs(pf - t) <= 4].any():
                    c['fp_double'] += 1          # a second detection next to a correct one
                else:
                    c['fp_other'] += 1
            for t in gf[~used]:
                c['fn'] += 1
                reg[register(p)]['fn'] += 1
                if other_pitch(pred_near, t, p, OCTAVE):
                    c['fn_octave'] += 1
                elif other_pitch(pred_near, t, p, (-1, 1)):
                    c['fn_semitone'] += 1
                else:
                    c['fn_nothing'] += 1

    c = {k: int(v) for k, v in c.items()}
    c['precision'] = c['tp'] / max(1, c['tp'] + c['fp'])
    c['recall'] = c['tp'] / max(1, c['tp'] + c['fn'])
    for r, v in reg.items():
        c[f'{r}_recall'] = v['tp'] / max(1, v['tp'] + v['fn'])
        c[f'{r}_precision'] = v['tp'] / max(1, v['tp'] + v['fp'])
    json.dump(c, open(args.out, 'w'), indent=1)
    print(c)
