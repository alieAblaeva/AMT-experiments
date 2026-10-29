import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import build_model, get_device

WIN = 1024


def pad(seg):
    return torch.nn.functional.pad(seg, (0, 0, 0, WIN - seg.shape[0]))


def windows(n):
    # (start, first frame to keep), the last window is moved back so it ends at the end of the piece
    if n <= WIN:
        return [(0, 0)]
    out = [(s, s) for s in range(0, n - WIN + 1, WIN)]
    done = out[-1][0] + WIN
    if done < n:
        out.append((n - WIN, done))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cache', required=True)      # folder with .pt files (mel, label, ...)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    model = build_model(args.mode)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    model.to(device).eval()

    files = sorted(glob.glob(os.path.join(args.cache, '*.pt')))
    if args.limit:
        files = files[:args.limit]
    for i, f in enumerate(files):
        out = os.path.join(args.out, os.path.basename(f)[:-3] + '.npz')
        if os.path.exists(out):
            continue
        mel = torch.load(f)['mel'].float()
        n = mel.shape[0]
        p_onset = np.zeros((n, 88), np.float16)
        wins = windows(n)
        for b in range(0, len(wins), args.batch):
            chunk = wins[b:b + args.batch]
            x = torch.stack([pad(mel[s:s + WIN]) for s, _ in chunk])
            with torch.no_grad():
                probs = torch.softmax(model(x.to(device)), -1)[..., 3].cpu().numpy()
            for j, (s, keep) in enumerate(chunk):
                end = min(s + WIN, n)
                p_onset[keep:end] = probs[j, keep - s:end - s]
        np.savez_compressed(out, p_onset=p_onset)
        print(f'{i + 1}/{len(files)}', os.path.basename(f))
