import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import peaks

# bins are inclusive on the left; the cache files need 'notes' = (pitch, start, end, velocity, duration)
REGISTER = [('bass', 21, 47), ('mid', 48, 83), ('treble', 84, 108)]
POLY = [('1', 0, 1), ('2', 2, 2), ('3-4', 3, 4), ('5-6', 5, 6), ('7+', 7, 10 ** 6)]
DURATION = [('<0.1 s', 0, 0.1), ('0.1-0.5 s', 0.1, 0.5), ('>=0.5 s', 0.5, 1e9)]
TOL = 2


def which(x, bins):
    for k, (_, lo, hi) in enumerate(bins):
        if lo <= x <= hi:
            return k
    return len(bins) - 1


def match(pred, gt):
    used = np.zeros(len(gt), bool)
    hit = np.zeros(len(pred), bool)
    for i, p in enumerate(pred):
        j = int(np.searchsorted(gt, p - TOL))
        while j < len(gt) and gt[j] <= p + TOL:
            if not used[j]:
                used[j] = hit[i] = True
                break
            j += 1
    return hit, used


def count_piece(cache_file, pred_file, theta, v_cut):
    # returns tp, fp, fn per bin for every dimension
    data = torch.load(cache_file)
    label = data['label'].numpy()
    notes = data['notes'].numpy()
    p_onset = np.load(pred_file)['p_onset'].astype(np.float32)
    onsets = peaks(p_onset[None], theta)[0]

    n = label.shape[0]
    sounding = np.zeros(n + 1, int)
    for pitch, s, e, vel, dur in notes:
        sounding[int(s)] += 1
        sounding[int(e)] -= 1
    sounding = np.cumsum(sounding)[:n]
    attr = {(int(p), int(s)): (v, d) for p, s, e, v, d in notes}

    c = {'all': np.zeros((1, 3), int), 'register': np.zeros((3, 3), int), 'poly': np.zeros((5, 3), int),
         'velocity': np.zeros((3, 3), int), 'duration': np.zeros((3, 3), int)}
    for p in range(88):
        pred = np.flatnonzero(onsets[:, p])
        gt = np.flatnonzero(label[:, p] == 3)
        hit, used = match(pred, gt)
        reg = which(p + 21, REGISTER)
        for f, h in zip(pred, hit):
            col = 0 if h else 1
            for name, k in (('all', 0), ('register', reg), ('poly', which(sounding[f], POLY))):
                c[name][k, col] += 1
        for f, u in zip(gt, used):
            if not u:
                for name, k in (('all', 0), ('register', reg), ('poly', which(sounding[f], POLY))):
                    c[name][k, 2] += 1
            if (p, f) in attr:          # velocity and duration exist only for reference notes, so only recall
                v, d = attr[(p, f)]
                col = 0 if u else 2
                c['velocity'][int(v >= v_cut[0]) + int(v >= v_cut[1]), col] += 1
                c['duration'][which(d, DURATION), col] += 1
    return c


def f1(tp, fp, fn):
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return 2 * p * r / max(1e-9, p + r)


def recall(tp, fp, fn):
    return tp / max(1, tp + fn)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--teacher', required=True)    # folders with the npz files of run_inference.py
    ap.add_argument('--student', required=True)
    ap.add_argument('--theta_teacher', type=float, required=True)
    ap.add_argument('--theta_student', type=float, required=True)
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache, '*.pt')))
    vel = np.concatenate([torch.load(f)['notes'].numpy()[:, 3] for f in files])
    v_cut = np.percentile(vel, [100 / 3, 200 / 3])

    counts = {}
    for name, folder, theta in (('teacher', args.teacher, args.theta_teacher), ('student', args.student, args.theta_student)):
        counts[name] = [count_piece(f, os.path.join(folder, os.path.basename(f)[:-3] + '.npz'), theta, v_cut) for f in files]

    names = {'all': ['all'], 'register': [b[0] for b in REGISTER], 'poly': [b[0] for b in POLY],
             'velocity': ['quiet', 'medium', 'loud'], 'duration': [b[0] for b in DURATION]}
    rng = np.random.default_rng(0)
    draws = [np.arange(len(files))] + [rng.integers(0, len(files), len(files)) for _ in range(args.boot)]

    result = {}
    for dim, labels in names.items():
        score = recall if dim in ('velocity', 'duration') else f1
        t = np.stack([c[dim] for c in counts['teacher']])
        s = np.stack([c[dim] for c in counts['student']])
        rows = []
        for k, label in enumerate(labels):
            gap = []
            for d in draws:
                gap.append(score(*t[d, k].sum(0)) - score(*s[d, k].sum(0)))
            rows.append({'bin': label, 'metric': score.__name__,
                         'teacher': score(*t[:, k].sum(0)), 'student': score(*s[:, k].sum(0)),
                         'gap': gap[0], 'ci': list(np.percentile(gap[1:], [2.5, 97.5]))})
        result[dim] = rows
    json.dump(result, open(args.out, 'w'), indent=1)
    for dim, rows in result.items():
        for r in rows:
            print(f"{dim:9s} {r['bin']:8s} teacher {r['teacher']:.3f} student {r['student']:.3f} gap {r['gap']:+.3f}")
